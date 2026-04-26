"""
PropFirmSimulator — 10-day withdrawal-period simulation.

Loads risk_approved.csv (from risk_guardian.py) to get approved strategy
names, imports the matching strategy objects from agents/strategies.py,
then re-runs each strategy on the most recent 10 real trading days from
xauusd_1m.csv — one day at a time — respecting hard daily caps:

    Daily target  : +$100  → stop trading for the day
    Daily cap     : +$150  → emergency stop (consistency rule)
    Daily stop    : -$75   → stop trading for the day

Scoring:
    PASS       all 10 days green, total $1 000–$1 500,
               best day < 30 % of total, ≥ 8 perfect days ($100–$150)
    BORDERLINE ≥ 7 green days, total $800–$1 500, best day < 35 % of total
    FAIL       everything else

Outputs:
    propfirm_results.csv   — one row per strategy with all 10-day stats
"""

import os
import sys
import math

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── File paths ─────────────────────────────────────────────────────────────────
APPROVED_FILE = os.path.join(_REPO_ROOT, "risk_approved.csv")
DATA_FILE     = os.path.join(_REPO_ROOT, "xauusd_1m.csv")
OUTPUT_FILE   = os.path.join(_REPO_ROOT, "propfirm_results.csv")

# ── Account / instrument constants ────────────────────────────────────────────
LOT_SIZE       = 0.03
CONTRACT_SIZE  = 100          # oz per standard lot
UNIT_VALUE     = LOT_SIZE * CONTRACT_SIZE   # $3 per $1 gold move
MAX_RISK_MULT  = 2.0          # skip trades with risk > MAX_RISK_PER_TRADE × this
MAX_RISK_USD   = 30.0

# ── Daily trading caps ─────────────────────────────────────────────────────────
DAILY_TARGET   =  100.0       # stop new entries when reached
DAILY_CAP      =  150.0       # close and halt if crossed mid-trade
DAILY_STOP     =  -75.0       # halt if reached

# ── Simulation parameters ──────────────────────────────────────────────────────
SIM_DAYS       = 10           # withdrawal period
LOOKBACK_BARS  = 400          # context bars fed to strategy for indicator warm-up


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_price_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Price data not found: {path}\n"
            "Run  python agents/data_agent.py  first."
        )
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_last_n_trading_days(df: pd.DataFrame, n: int) -> list[pd.Timestamp]:
    """Return the last *n* unique calendar dates that have bar data."""
    dates = df["datetime"].dt.date.unique()
    dates_sorted = sorted(dates)
    if len(dates_sorted) < n:
        raise ValueError(
            f"Only {len(dates_sorted)} trading days in data; need {n}."
        )
    return [pd.Timestamp(d) for d in dates_sorted[-n:]]


# ─────────────────────────────────────────────────────────────────────────────
# Single-day simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_day(
    strategy,
    context_bars: pd.DataFrame,
    day_bars: pd.DataFrame,
) -> tuple[float, list[dict]]:
    """
    Simulate one trading day with daily profit / loss caps.

    Signals are generated once over (context_bars ‖ day_bars); only bars
    that fall inside *day_bars* generate new entries.

    Returns (daily_pnl_usd, list_of_trade_dicts).
    """
    all_bars   = pd.concat([context_bars, day_bars], ignore_index=True)
    day_start  = len(context_bars)        # first index that belongs to today
    n          = len(all_bars)

    # ── Generate signals for the full window at once ───────────────────────
    try:
        signals_df = strategy.generate_signals(all_bars.copy())
    except Exception as exc:
        return 0.0, []

    # ── Bar-by-bar execution ───────────────────────────────────────────────
    trades: list[dict] = []
    daily_pnl  = 0.0
    in_trade   = False
    entry_p    = 0.0
    entry_time = None
    direction  = 0
    sl = tp    = 0.0

    for i in range(day_start, n - 1):

        # ── Daily limit check ──────────────────────────────────────────────
        if daily_pnl >= DAILY_CAP:
            break
        if daily_pnl >= DAILY_TARGET:
            break
        if daily_pnl <= DAILY_STOP:
            break

        bar = all_bars.iloc[i]
        lo  = float(bar["low"])
        hi  = float(bar["high"])

        # ── Manage open trade ──────────────────────────────────────────────
        if in_trade:
            sl_hit = (direction ==  1 and lo <= sl) or (direction == -1 and hi >= sl)
            tp_hit = (direction ==  1 and hi >= tp) or (direction == -1 and lo <= tp)

            if sl_hit or tp_hit:
                exit_p  = sl if sl_hit else tp
                outcome = "sl" if sl_hit else "tp"
                raw_pnl = (exit_p - entry_p) * UNIT_VALUE * direction

                # Clip the winning trade so daily total never exceeds the cap
                if daily_pnl + raw_pnl > DAILY_CAP:
                    raw_pnl = DAILY_CAP - daily_pnl

                daily_pnl = round(daily_pnl + raw_pnl, 2)
                trades.append({
                    "time":        bar["datetime"],
                    "direction":   "buy" if direction == 1 else "sell",
                    "entry_price": round(entry_p, 4),
                    "exit_price":  round(exit_p,  4),
                    "pnl_usd":     round(raw_pnl, 2),
                    "outcome":     outcome,
                })
                in_trade = False

                # Re-check limits right after a close
                if daily_pnl >= DAILY_TARGET or daily_pnl <= DAILY_STOP:
                    break
            else:
                continue     # still in trade; skip entry check

        # ── New entry ──────────────────────────────────────────────────────
        sig_row = signals_df.iloc[i]
        sig     = int(sig_row.get("signal", 0))

        if sig not in (1, -1):
            continue

        sl_p = float(sig_row.get("sl_price", 0.0))
        tp_p = float(sig_row.get("tp_price", 0.0))

        if sl_p <= 0.0 or tp_p <= 0.0:
            continue

        next_bar   = all_bars.iloc[i + 1]
        entry_p    = float(next_bar["open"])
        entry_time = next_bar["datetime"]
        direction  = sig
        sl, tp     = sl_p, tp_p

        risk_usd = abs(entry_p - sl) * UNIT_VALUE
        if risk_usd <= 0.0 or risk_usd > MAX_RISK_USD * MAX_RISK_MULT:
            continue

        in_trade = True

    # ── Close any open trade at end of day ─────────────────────────────────
    if in_trade:
        last      = all_bars.iloc[-1]
        exit_p    = float(last["close"])
        raw_pnl   = (exit_p - entry_p) * UNIT_VALUE * direction

        if daily_pnl + raw_pnl > DAILY_CAP:
            raw_pnl = DAILY_CAP - daily_pnl

        daily_pnl = round(daily_pnl + raw_pnl, 2)
        trades.append({
            "time":        last["datetime"],
            "direction":   "buy" if direction == 1 else "sell",
            "entry_price": round(entry_p, 4),
            "exit_price":  round(exit_p,  4),
            "pnl_usd":     round(raw_pnl, 2),
            "outcome":     "eod",
        })

    return daily_pnl, trades


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _day_label(pnl: float) -> str:
    if 100.0 <= pnl <= 150.0:
        return "✅ Perfect"
    if pnl > 150.0:
        return "⚠️  Over cap"
    if pnl > 0.0:
        return "🟡 Profit"
    return "❌ Loss day"


def score_simulation(day_pnls: list[float]) -> tuple[str, dict]:
    """Return (verdict, stats_dict) for a completed 10-day simulation."""
    total        = round(sum(day_pnls), 2)
    best_day     = max(day_pnls) if day_pnls else 0.0
    worst_day    = min(day_pnls) if day_pnls else 0.0
    green_days   = sum(1 for p in day_pnls if p > 0)
    red_days     = sum(1 for p in day_pnls if p <= 0)
    perfect_days = sum(1 for p in day_pnls if 100.0 <= p <= 150.0)
    consistency  = round(best_day / total, 4) if total > 0 else 99.0

    if (
        green_days   == SIM_DAYS
        and 1_000.0  <= total      <= 1_500.0
        and consistency             <  0.30
        and perfect_days           >= 8
    ):
        verdict = "PASS"
    elif (
        green_days   >= 7
        and 800.0    <= total      <= 1_500.0
        and consistency             <  0.35
    ):
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"

    return verdict, {
        "total_profit":   total,
        "best_day":       round(best_day,  2),
        "worst_day":      round(worst_day, 2),
        "green_days":     green_days,
        "red_days":       red_days,
        "perfect_days":   perfect_days,
        "consistency":    consistency,
        "verdict":        verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print table
# ─────────────────────────────────────────────────────────────────────────────

def print_simulation_table(name: str, day_dates: list, day_pnls: list[float], stats: dict) -> None:
    verdict = stats["verdict"]
    verdict_icon = {"PASS": "✅", "BORDERLINE": "🟡", "FAIL": "❌"}.get(verdict, "")

    print(f"\nStrategy: {name}")
    print("─" * 50)
    for i, (dt, pnl) in enumerate(zip(day_dates, day_pnls), 1):
        sign  = "+" if pnl >= 0 else ""
        label = _day_label(pnl)
        print(f"  Day {i:02d} ({dt.strftime('%Y-%m-%d')}): {sign}${pnl:.2f}  {label}")
    print("─" * 50)

    total = stats["total_profit"]
    best  = stats["best_day"]
    cs    = stats["consistency"]
    pct   = cs * 100

    sign = "+" if total >= 0 else ""
    print(f"  Total: {sign}${total:.2f}")
    print(f"  Best day: ${best:.2f}  ({pct:.1f}% of total)")
    print(f"  Consistency score: {cs:.3f}")
    print(f"  Perfect days: {stats['perfect_days']}/{SIM_DAYS}")
    print(f"  Result: {verdict} {verdict_icon}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load approved strategy names ───────────────────────────────────────────
    if not os.path.isfile(APPROVED_FILE):
        print(
            f"ERROR: {APPROVED_FILE} not found.\n"
            "Run  python agents/risk_guardian.py  first."
        )
        sys.exit(1)

    approved_df    = pd.read_csv(APPROVED_FILE)
    approved_names = set(approved_df["strategy"].astype(str).tolist())
    print(f"Loaded {len(approved_names)} approved strategies from {APPROVED_FILE}")

    # ── Import strategy objects ────────────────────────────────────────────────
    try:
        from strategies import STRATEGIES   # noqa: PLC0415
    except ImportError:
        print(
            "\nERROR: Cannot import STRATEGIES from agents/strategies.py\n"
            "Run  python agents/strategy_inventor.py  first.\n"
        )
        sys.exit(1)

    strategy_map = {
        getattr(s, "name", f"Strategy_{i}"): s
        for i, s in enumerate(STRATEGIES)
    }

    # Filter to approved names
    to_simulate = {
        name: strat
        for name, strat in strategy_map.items()
        if name in approved_names
    }

    if not to_simulate:
        print(
            "WARNING: None of the approved strategy names matched the loaded "
            "STRATEGIES list.  Check that agents/strategies.py is up to date."
        )
        sys.exit(0)

    print(f"Matched {len(to_simulate)} strategies for simulation\n")

    # ── Load price data and identify last 10 trading days ─────────────────────
    print("Loading price data …")
    price_df    = load_price_data(DATA_FILE)
    sim_days    = get_last_n_trading_days(price_df, SIM_DAYS)
    print(f"Simulating {SIM_DAYS} days:  {sim_days[0].date()}  →  {sim_days[-1].date()}\n")

    # ── Simulate each strategy ────────────────────────────────────────────────
    print("=" * 60)
    print("10-DAY PROP-FIRM WITHDRAWAL SIMULATION")
    print("=" * 60)

    all_rows: list[dict] = []
    counts = {"PASS": 0, "BORDERLINE": 0, "FAIL": 0}

    for name, strategy in to_simulate.items():
        day_pnls: list[float]     = []
        day_dates: list[pd.Timestamp] = []

        for sim_day in sim_days:
            day_date_val = sim_day.date()

            # All bars that belong to today
            day_mask = price_df["datetime"].dt.date == day_date_val
            day_bars = price_df[day_mask].reset_index(drop=True)

            if day_bars.empty:
                day_pnls.append(0.0)
                day_dates.append(sim_day)
                continue

            # Context: LOOKBACK_BARS immediately before today's first bar
            first_idx = price_df.index[day_mask][0]
            ctx_start = max(0, first_idx - LOOKBACK_BARS)
            context_bars = price_df.iloc[ctx_start:first_idx].reset_index(drop=True)

            daily_pnl, _ = simulate_day(strategy, context_bars, day_bars)
            day_pnls.append(daily_pnl)
            day_dates.append(sim_day)

        verdict, stats = score_simulation(day_pnls)
        counts[verdict] += 1

        print_simulation_table(name, day_dates, day_pnls, stats)

        # Build CSV row
        row: dict = {"strategy": name}
        row.update(stats)
        for i, (dt, pnl) in enumerate(zip(day_dates, day_pnls), 1):
            row[f"day{i:02d}_date"] = dt.strftime("%Y-%m-%d")
            row[f"day{i:02d}_pnl"]  = pnl
        all_rows.append(row)

    # ── Save results ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved → {OUTPUT_FILE}")

    # ── Final summary ──────────────────────────────────────────────────────────
    total_sim = len(all_rows)
    print("\n" + "=" * 60)
    print("PROP-FIRM SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Strategies simulated : {total_sim}")
    print(f"  PASS                 : {counts['PASS']}")
    print(f"  BORDERLINE           : {counts['BORDERLINE']}")
    print(f"  FAIL                 : {counts['FAIL']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
