"""
BacktestEngine — walk-forward backtester for XAU/USD strategies.

Expects agents/strategies.py to expose:
    STRATEGIES: list of strategy instances, each with:
        .name  (str)
        .generate_signals(df: pd.DataFrame) -> pd.DataFrame
            Returns df with new columns:
                signal    : 1 (buy), -1 (sell), 0 (flat)
                sl_price  : stop-loss price for that signal
                tp_price  : take-profit price for that signal

Walk-forward split:  Training 60% | Validation 20% | Live-sim 20%
A strategy must show positive net profit on ALL THREE periods to pass.

Account constants:
    Account size  : $10,000
    Lot size      : 0.03 lots
    Contract size : 100 oz/lot  →  unit value = $3 per $1 gold move
    Max risk/trade: $30

Outputs:
    backtest_results.csv   (all strategies with all metrics)
"""

import os
import sys
import math
import traceback

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Path: allow `from strategies import STRATEGIES` from agents/ dir ───────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_HERE)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Account / instrument constants ────────────────────────────────────────────
ACCOUNT_SIZE        = 10_000.0
LOT_SIZE            = 0.03
CONTRACT_SIZE       = 100           # oz per standard lot (XAU/USD)
UNIT_VALUE          = LOT_SIZE * CONTRACT_SIZE   # $/point → $3.00
MAX_RISK_PER_TRADE  = 30.0          # hard cap; trades exceeding 2× are skipped

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_FILE   = os.path.join(_REPO_ROOT, "xauusd_5m.csv")
OUTPUT_FILE = os.path.join(_REPO_ROOT, "backtest_results.csv")

# ── Split ratios ───────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# live-sim takes the remaining 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Data loading / splitting
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run  python agents/data_agent.py  first to download XAU/USD data."
        )
    df = pd.read_csv(path, parse_dates=["datetime"])
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def split_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n     = len(df)
    t_end = int(n * TRAIN_RATIO)
    v_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return (
        df.iloc[:t_end].reset_index(drop=True),
        df.iloc[t_end:v_end].reset_index(drop=True),
        df.iloc[v_end:].reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trade simulation
# ─────────────────────────────────────────────────────────────────────────────

_TRADE_COLS = [
    "entry_time", "exit_time", "direction",
    "entry_price", "exit_price", "sl_price", "tp_price",
    "pnl_usd", "outcome",
]


def simulate_trades(signals_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate bar-by-bar trade execution.

    Entry rule : signal on bar i  →  enter at bar i+1 open.
    Exit  rule : first bar where low ≤ sl (long) or high ≥ sl (short) → fill at sl;
                 first bar where high ≥ tp (long) or low ≤ tp (short)  → fill at tp.
                 If both triggered on same bar, SL wins (conservative).
    End-of-period: open trade closed at final bar's close.
    """
    trades: list[dict] = []

    in_trade   = False
    entry_p    = 0.0
    entry_time = None
    direction  = 0
    sl = tp    = 0.0

    n = len(price_df)

    for i in range(n - 1):
        bar = price_df.iloc[i]
        lo, hi = float(bar["low"]), float(bar["high"])

        # ── Manage open trade ──────────────────────────────────────────────
        if in_trade:
            sl_hit = (direction ==  1 and lo <= sl) or (direction == -1 and hi >= sl)
            tp_hit = (direction ==  1 and hi >= tp) or (direction == -1 and lo <= tp)

            if sl_hit or tp_hit:
                exit_p  = sl if sl_hit else tp
                outcome = "sl" if sl_hit else "tp"
                pnl     = (exit_p - entry_p) * UNIT_VALUE * direction

                trades.append({
                    "entry_time":  entry_time,
                    "exit_time":   bar["datetime"],
                    "direction":   "buy" if direction == 1 else "sell",
                    "entry_price": round(entry_p, 4),
                    "exit_price":  round(exit_p, 4),
                    "sl_price":    round(sl, 4),
                    "tp_price":    round(tp, 4),
                    "pnl_usd":     round(pnl, 2),
                    "outcome":     outcome,
                })
                in_trade = False
            else:
                continue   # still holding; skip new-entry check

        # ── Look for new entry ─────────────────────────────────────────────
        sig_row = signals_df.iloc[i]
        sig     = int(sig_row.get("signal", 0))

        if sig not in (1, -1):
            continue

        sl_p = float(sig_row.get("sl_price", 0.0))
        tp_p = float(sig_row.get("tp_price", 0.0))

        if sl_p <= 0.0 or tp_p <= 0.0:
            continue

        next_bar   = price_df.iloc[i + 1]
        entry_p    = float(next_bar["open"])
        entry_time = next_bar["datetime"]
        direction  = sig
        sl, tp     = sl_p, tp_p

        # Risk validation
        risk_usd = abs(entry_p - sl) * UNIT_VALUE
        if risk_usd <= 0.0 or risk_usd > MAX_RISK_PER_TRADE * 2.0:
            continue

        in_trade = True

    # Close end-of-period open trade
    if in_trade and n > 0:
        last   = price_df.iloc[-1]
        exit_p = float(last["close"])
        pnl    = (exit_p - entry_p) * UNIT_VALUE * direction
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   last["datetime"],
            "direction":   "buy" if direction == 1 else "sell",
            "entry_price": round(entry_p, 4),
            "exit_price":  round(exit_p, 4),
            "sl_price":    round(sl, 4),
            "tp_price":    round(tp, 4),
            "pnl_usd":     round(pnl, 2),
            "outcome":     "eop",
        })

    return pd.DataFrame(trades, columns=_TRADE_COLS) if trades else pd.DataFrame(columns=_TRADE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics calculation
# ─────────────────────────────────────────────────────────────────────────────

def _empty_metrics() -> dict:
    return dict(
        net_profit=0.0, total_trades=0, win_rate=0.0, profit_factor=0.0,
        sharpe=0.0, max_drawdown=0.0, avg_daily_pnl=0.0, std_daily_pnl=0.0,
        consistency_score=99.0,
    )


def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return _empty_metrics()

    pnls    = trades["pnl_usd"].to_numpy(dtype=float)
    winners = pnls[pnls > 0]
    losers  = pnls[pnls < 0]
    n       = len(pnls)

    net_profit    = float(pnls.sum())
    win_rate      = round(len(winners) / n * 100, 2)
    gross_profit  = float(winners.sum()) if len(winners) else 0.0
    gross_loss    = float(abs(losers.sum())) if len(losers) else 0.0
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")

    # Max drawdown via equity curve
    equity   = np.cumsum(pnls)
    peak     = np.maximum.accumulate(equity)
    max_dd   = round(float((peak - equity).max()), 2)

    # Daily aggregation — group by calendar date extracted from exit_time
    tmp = trades.copy()
    tmp["date"] = pd.to_datetime(tmp["exit_time"]).dt.strftime("%Y-%m-%d")
    daily = tmp.groupby("date")["pnl_usd"].sum()

    avg_daily = round(float(daily.mean()), 2)      if len(daily)  > 0 else 0.0
    std_daily = round(float(daily.std()),  2)      if len(daily)  > 1 else 0.0

    # Sharpe (annualised, 252 trading days)
    sharpe = round((avg_daily / std_daily) * math.sqrt(252), 3) if std_daily > 0 else 0.0

    # Consistency score: best single day / total profit (lower = better)
    if net_profit > 0 and len(daily) > 0:
        best_day          = float(daily.max())
        consistency_score = round(best_day / net_profit, 4)
    else:
        consistency_score = 99.0   # impossible value signals failure

    return dict(
        net_profit        = round(net_profit, 2),
        total_trades      = n,
        win_rate          = win_rate,
        profit_factor     = profit_factor,
        sharpe            = sharpe,
        max_drawdown      = max_dd,
        avg_daily_pnl     = avg_daily,
        std_daily_pnl     = std_daily,
        consistency_score = consistency_score,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_walkforward(
    strategy,
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_live:  pd.DataFrame,
) -> dict | None:
    """
    Run strategy on all three periods.
    Returns a dict of period → metrics, or None on exception.
    """
    period_results: dict[str, dict] = {}

    for label, period_df in (("train", df_train), ("val", df_val), ("live", df_live)):
        try:
            signals = strategy.generate_signals(period_df.copy())
        except Exception:
            print(f"      [!] Signal error on {label}:")
            traceback.print_exc()
            return None

        trades  = simulate_trades(signals, period_df)
        metrics = compute_metrics(trades)
        period_results[label] = metrics

    return period_results


def _fmt_period(label: str, m: dict) -> str:
    pf_str = f"{m['profit_factor']:.2f}" if math.isfinite(m["profit_factor"]) else "∞"
    return (
        f"    {label:10s}: profit=${m['net_profit']:>8.2f}  "
        f"win={m['win_rate']:5.1f}%  PF={pf_str}  "
        f"trades={m['total_trades']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Import strategies ──────────────────────────────────────────────────────
    try:
        from strategies import STRATEGIES   # noqa: PLC0415
    except ImportError:
        print(
            "\nERROR: Cannot import STRATEGIES from agents/strategies.py\n"
            "Make sure agents/strategies.py exists and defines a STRATEGIES list.\n"
            "Run  python agents/strategy_inventor.py  first.\n"
        )
        sys.exit(1)

    # ── Load price data ────────────────────────────────────────────────────────
    print("Loading price data …")
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df):,} bars  ({df['datetime'].iloc[0]}  →  {df['datetime'].iloc[-1]})")

    df_train, df_val, df_live = split_periods(df)
    print(
        f"Split  — train: {len(df_train):,}  "
        f"val: {len(df_val):,}  "
        f"live: {len(df_live):,} bars\n"
    )

    total     = len(STRATEGIES)
    passed    : list[str] = []
    all_rows  : list[dict] = []

    # ── Walk-forward loop ──────────────────────────────────────────────────────
    for idx, strategy in enumerate(STRATEGIES, 1):
        name = getattr(strategy, "name", f"Strategy_{idx}")
        print(f"Testing strategy {idx}/{total}: {name}")

        result = run_walkforward(strategy, df_train, df_val, df_live)

        if result is None:
            print("    FAILED (exception — see above)\n")
            row = {"strategy": name, "passed_walkforward": False}
            all_rows.append(row)
            continue

        t, v, l = result["train"], result["val"], result["live"]
        print(_fmt_period("Training",   t))
        print(_fmt_period("Validation", v))
        print(_fmt_period("Live sim",   l))

        passed_wf = t["net_profit"] > 0 and v["net_profit"] > 0 and l["net_profit"] > 0
        verdict   = "PASSED ✓" if passed_wf else "FAILED ✗"
        print(f"    ──> {verdict} walk-forward test\n")

        if passed_wf:
            passed.append(name)

        # Flatten all metrics into a single CSV row
        row: dict = {"strategy": name, "passed_walkforward": passed_wf}
        for period_label, metrics in result.items():
            for k, v_val in metrics.items():
                row[f"{period_label}_{k}"] = v_val
        all_rows.append(row)

    # ── Save results ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved → {OUTPUT_FILE}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(passed)} / {total} strategies passed walk-forward test")
    if passed:
        print("Passing strategies:")
        for s in passed:
            print(f"  • {s}")
    print("=" * 60)


if __name__ == "__main__":
    main()
