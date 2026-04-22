"""
Comprehensive EURUSD backtester.

Usage
-----
    python backtest.py                          # default: 5m / 60d
    python backtest.py --interval 1m --period 7d
    python backtest.py --interval 15m --period 60d

Output
------
1. Top-10 leaderboard (ranked by composite score) printed to stdout.
2. backtest_results.csv — every run × every direction (sortable in Excel).
3. backtest_top10.json  — machine-readable top 10.
4. best_strategy.json   — the single best run (feed it to apply_best_strategy.py).

Strategies enumerated: EMA crossovers (± trend filter), MACD, Supertrend,
Ichimoku TK-cross, RSI mean-reversion, Bollinger (reversion & breakout),
Stochastic, EMA+ADX, London/NY session breakouts.  Each is tested as
long-only, short-only, and both-directions.

Exits: ATR-based SL/TP (1.5× / 3.0×), identical to the live bot, so ranking
reflects the bot's actual executable edge.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import backtest_data
import backtest_strategies as strat_lib
from backtest_strategies import atr

# ── Economics (must match config.py) ──────────────────────────────────────────
LOT_SIZE       = 0.05
PIP_SIZE       = 0.0001
PIP_VALUE_STD  = 10.0    # USD per pip per 1.0 std lot
ATR_PERIOD     = 14
ATR_SL_MULT    = 1.5
ATR_TP_MULT    = 3.0
SLIPPAGE_PIPS  = 0.2     # executions assumed 0.2 pip worse than signal bar
SPREAD_PIPS    = 0.5     # round-trip spread cost in pips (entry + exit combined)
MIN_TRADES     = 10      # discard runs with too few samples for stats

INTERVAL_MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("backtest")


# ── Trade / result containers ─────────────────────────────────────────────────

@dataclass
class Trade:
    side: str           # "long" | "short"
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp
    exit: float
    pnl_usd: float
    reason: str         # "TP" | "SL" | "EOD"


@dataclass
class Result:
    strategy: str
    direction: str
    params: dict
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    max_dd_usd: float = 0.0
    max_dd_pct: float = 0.0
    sharpe: float = 0.0
    composite: float = 0.0
    trade_list: list = field(default_factory=list, repr=False)


# ── Backtester core ───────────────────────────────────────────────────────────

def _pnl_long(entry, exit_):
    pips = (exit_ - entry) / PIP_SIZE
    return pips * PIP_VALUE_STD * LOT_SIZE


def _pnl_short(entry, exit_):
    pips = (entry - exit_) / PIP_SIZE
    return pips * PIP_VALUE_STD * LOT_SIZE


def run_once(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    direction: str,
) -> list[Trade]:
    """
    Walk the DataFrame bar-by-bar.  At most one open position at a time.
    Entry is at the next bar's open.  Exit is SL/TP checked on that bar's
    high/low; if both are touched, assume the worse outcome (SL).
    """
    a = atr(df, ATR_PERIOD).values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    idx = df.index

    open_trade: Optional[dict] = None
    trades: list[Trade] = []

    sl_mult = ATR_SL_MULT
    tp_mult = ATR_TP_MULT
    slip    = SLIPPAGE_PIPS * PIP_SIZE
    spread  = SPREAD_PIPS   * PIP_SIZE

    long_entries  = entries["long_entry"].values  if direction in ("long", "both") else np.zeros(len(df), dtype=bool)
    short_entries = entries["short_entry"].values if direction in ("short", "both") else np.zeros(len(df), dtype=bool)

    n = len(df)
    for i in range(n - 1):
        # 1. If we hold a position, check for exit on this bar
        if open_trade is not None:
            side = open_trade["side"]
            sl   = open_trade["sl"]
            tp   = open_trade["tp"]
            hit_sl = l[i] <= sl if side == "long" else h[i] >= sl
            hit_tp = h[i] >= tp if side == "long" else l[i] <= tp

            if hit_sl or hit_tp:
                # Pessimistic: if both within same bar, SL first
                reason = "SL" if hit_sl else "TP"
                exit_price = sl if reason == "SL" else tp
                pnl = _pnl_long(open_trade["entry"], exit_price) if side == "long" else _pnl_short(open_trade["entry"], exit_price)
                pnl -= spread / PIP_SIZE * PIP_VALUE_STD * LOT_SIZE
                trades.append(Trade(
                    side=side,
                    entry_time=open_trade["entry_time"],
                    entry=open_trade["entry"],
                    exit_time=idx[i],
                    exit=exit_price,
                    pnl_usd=pnl,
                    reason=reason,
                ))
                open_trade = None

        # 2. Check for new entries (skip if we're still in a trade)
        if open_trade is not None:
            continue

        atr_now = a[i]
        if not np.isfinite(atr_now) or atr_now <= 0:
            continue

        if long_entries[i]:
            entry = o[i + 1] + slip
            sl = entry - sl_mult * atr_now
            tp = entry + tp_mult * atr_now
            open_trade = {"side": "long", "entry": entry, "sl": sl, "tp": tp, "entry_time": idx[i + 1]}
        elif short_entries[i]:
            entry = o[i + 1] - slip
            sl = entry + sl_mult * atr_now
            tp = entry - tp_mult * atr_now
            open_trade = {"side": "short", "entry": entry, "sl": sl, "tp": tp, "entry_time": idx[i + 1]}

    # Force-close any still-open trade at the final bar's close
    if open_trade is not None:
        side = open_trade["side"]
        exit_price = df["close"].iat[-1]
        pnl = _pnl_long(open_trade["entry"], exit_price) if side == "long" else _pnl_short(open_trade["entry"], exit_price)
        pnl -= spread / PIP_SIZE * PIP_VALUE_STD * LOT_SIZE
        trades.append(Trade(
            side=side,
            entry_time=open_trade["entry_time"],
            entry=open_trade["entry"],
            exit_time=idx[-1],
            exit=exit_price,
            pnl_usd=pnl,
            reason="EOD",
        ))

    return trades


# ── Stats ─────────────────────────────────────────────────────────────────────

def _stats_from_trades(trades: list[Trade]) -> dict:
    if not trades:
        return dict(trades=0)
    pnls = np.array([t.pnl_usd for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    gross_win  = float(wins.sum())
    gross_loss = float(-losses.sum())
    net = float(pnls.sum())

    # Equity curve for drawdown (starting equity normalised to |net| scale).
    equity = np.cumsum(pnls)
    # "peak so far" then drawdown
    peak = np.maximum.accumulate(equity)
    dd_usd = float(np.min(equity - peak))      # most negative
    base = 100.0   # treat $100 as starting, express DD in % of peak+base
    dd_pct = float(((equity - peak) / (peak + base)).min() * 100.0)

    # Sharpe on trade-returns (no risk-free)
    if len(pnls) > 1 and pnls.std(ddof=1) > 0:
        sharpe = float(pnls.mean() / pnls.std(ddof=1) * math.sqrt(len(pnls)))
    else:
        sharpe = 0.0

    return dict(
        trades=int(len(trades)),
        wins=int(len(wins)),
        losses=int(len(losses)),
        gross_win=gross_win,
        gross_loss=gross_loss,
        net_profit=net,
        win_rate=float(len(wins) / len(trades) * 100.0),
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0),
        avg_trade=float(pnls.mean()),
        max_win=float(pnls.max()),
        max_loss=float(pnls.min()),
        max_dd_usd=dd_usd,
        max_dd_pct=dd_pct,
        sharpe=sharpe,
    )


def _composite(row: dict) -> float:
    """
    Simple, robust composite ranker:
        favours net profit, profit factor, Sharpe.  Penalises drawdown.
    PF is capped at 5 so a run with 2 lucky trades doesn't dominate.
    """
    if row["trades"] < MIN_TRADES:
        return -1e9
    pf = min(row["profit_factor"], 5.0) if math.isfinite(row["profit_factor"]) else 5.0
    # Normalise net profit to a 0-ish scale (USD)
    return (
        row["net_profit"] * 1.0
        + pf * 10.0
        + row["sharpe"] * 5.0
        + row["max_dd_usd"] * 0.5    # dd is negative → penalty
    )


# ── Orchestration ─────────────────────────────────────────────────────────────

def _bar_minutes(interval: str) -> int:
    return INTERVAL_MIN.get(interval, 5)


def run_all(interval: str, period: str | None):
    df = backtest_data.fetch(interval=interval, period=period)
    strategies = strat_lib.all_strategies(bar_minutes=_bar_minutes(interval))
    log.info(f"Will evaluate {len(strategies)} strategies × 3 directions = {len(strategies) * 3} runs")

    rows: list[Result] = []
    start = time.time()

    for i, strat in enumerate(strategies, 1):
        try:
            entries = strat.signals(df)
        except Exception as exc:
            log.warning(f"signal failure [{strat.name}]: {exc}")
            continue

        for direction in ("long", "short", "both"):
            trades = run_once(df, entries, direction)
            stats  = _stats_from_trades(trades)
            r = Result(strategy=strat.name, direction=direction, params=strat.params, **stats)
            r.composite = _composite(stats)
            r.trade_list = trades
            rows.append(r)

        if i % 10 == 0 or i == len(strategies):
            log.info(f"[{i:>3}/{len(strategies)}]  elapsed={time.time() - start:5.1f}s  last={strat.name}")

    log.info(f"All runs done in {time.time() - start:.1f}s")
    return df, rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def _format_top(rows: list[dict], title: str, sort_key: str, reverse: bool = True, n: int = 10):
    filtered = [r for r in rows if r["trades"] >= MIN_TRADES]
    ranked = sorted(filtered, key=lambda r: (r[sort_key] if math.isfinite(r[sort_key]) else -math.inf), reverse=reverse)[:n]
    print(f"\n{'=' * 92}")
    print(f"TOP {n} BY {title}")
    print("=" * 92)
    print(f"{'#':>2}  {'strategy':40}  {'dir':5}  {'trades':>6}  {'net$':>8}  {'PF':>5}  {'win%':>5}  {'DD$':>7}  {'Sharpe':>7}")
    print("-" * 92)
    for i, r in enumerate(ranked, 1):
        pf = "inf" if r["profit_factor"] == math.inf else f"{r['profit_factor']:5.2f}"
        print(
            f"{i:>2}  {r['strategy'][:40]:40}  {r['direction']:5}  "
            f"{r['trades']:>6}  {r['net_profit']:>8.2f}  {pf:>5}  "
            f"{r['win_rate']:>5.1f}  {r['max_dd_usd']:>7.2f}  {r['sharpe']:>7.2f}"
        )


def report(rows: list[Result], interval: str, period: str | None):
    # Convert to plain dicts for DataFrame / JSON
    plain: list[dict] = []
    for r in rows:
        d = asdict(r)
        d.pop("trade_list", None)
        plain.append(d)

    # Full CSV for manual inspection
    df_out = pd.DataFrame(plain)
    df_out = df_out.sort_values("composite", ascending=False)
    df_out.to_csv("backtest_results.csv", index=False)
    log.info(f"Wrote {len(df_out)} rows to backtest_results.csv")

    # Console leaderboards
    _format_top(plain, "COMPOSITE SCORE",   "composite")
    _format_top(plain, "NET PROFIT ($)",     "net_profit")
    _format_top(plain, "PROFIT FACTOR",      "profit_factor")
    _format_top(plain, "WIN RATE (%)",       "win_rate")
    _format_top(plain, "SHARPE RATIO",       "sharpe")
    _format_top(plain, "SMALLEST MAX DD",    "max_dd_usd")

    # Machine-readable top 10 by composite
    top10 = sorted(
        [r for r in plain if r["trades"] >= MIN_TRADES],
        key=lambda r: r["composite"],
        reverse=True,
    )[:10]
    with open("backtest_top10.json", "w") as f:
        json.dump({"interval": interval, "period": period, "top10": top10}, f, indent=2, default=str)
    log.info(f"Wrote top-10 to backtest_top10.json")

    # Best strategy — the one apply_best_strategy.py will install
    if top10:
        best = top10[0]
        with open("best_strategy.json", "w") as f:
            json.dump({"interval": interval, "period": period, "best": best}, f, indent=2, default=str)
        log.info(f"Wrote winner to best_strategy.json: {best['strategy']} ({best['direction']})")
        print("\n" + "=" * 92)
        print("WINNER (by composite):")
        print("=" * 92)
        print(json.dumps(best, indent=2, default=str))
    else:
        log.warning("No strategy produced >= MIN_TRADES trades.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="5m", choices=list(INTERVAL_MIN.keys()),
                   help="Bar interval (default 5m — best trade-off of resolution vs 60 d history)")
    p.add_argument("--period", default=None,
                   help="Period string (e.g. 7d, 60d, 730d). Default = Yahoo's max for the interval.")
    args = p.parse_args()

    df, rows = run_all(args.interval, args.period)
    report(rows, args.interval, args.period)


if __name__ == "__main__":
    main()
