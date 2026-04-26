"""
RiskGuardian — prop-firm compliance filter for backtested XAU/USD strategies.

Loads backtest_results.csv (written by backtest_engine.py), applies hard
prop-firm rules against the live-simulation period metrics, and writes
risk_approved.csv with only the strategies that pass every filter.

All filters operate on the 'live_*' columns, which represent the final
out-of-sample period — the only period that matters for real trading.

Prop-firm hard rules applied:
    ① Avg daily PnL  ∈ [$100, $150]
    ② Std daily PnL  < $50  (day-to-day consistency)
    ③ Consistency score  < 0.30  (best day < 30 % of total profit)
    ④ Max drawdown   ≤ $500  (5 % of $10 000 account)
    ⑤ Win rate       > 45 %
    ⑥ Profit factor  > 1.3
    ⑦ Trade count    ≥ 20  (statistical validity)
    ⑧ 95 % daily band  ∈ [$50, $200]  (avg ± 2×std)
    ⑨ Daily loss floor  > -$75  (avg − 3×std > −75; covers 99.7 % of days)
"""

import os
import sys
import math

import pandas as pd

# ── File paths ─────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

INPUT_FILE  = os.path.join(_REPO_ROOT, "backtest_results.csv")
OUTPUT_FILE = os.path.join(_REPO_ROOT, "risk_approved.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Individual filter functions
# Each returns None (pass) or a human-readable failure reason string.
# ─────────────────────────────────────────────────────────────────────────────

def _check_avg_daily_pnl(row: pd.Series) -> str | None:
    v = row["live_avg_daily_pnl"]
    if v < 100.0:
        return f"avg_daily_pnl ${v:.2f} is below the $100 minimum"
    if v > 150.0:
        return f"avg_daily_pnl ${v:.2f} exceeds the $150 consistency cap"
    return None


def _check_std_daily_pnl(row: pd.Series) -> str | None:
    v = row["live_std_daily_pnl"]
    if v >= 50.0:
        return f"std_daily_pnl ${v:.2f} ≥ $50 — daily results too inconsistent"
    return None


def _check_consistency_score(row: pd.Series) -> str | None:
    v = row["live_consistency_score"]
    if v >= 0.30:
        return f"consistency_score {v:.4f} ≥ 0.30 — best day is ≥ 30 % of total profit"
    return None


def _check_max_drawdown(row: pd.Series) -> str | None:
    v = row["live_max_drawdown"]
    if v > 500.0:
        return f"max_drawdown ${v:.2f} exceeds $500 account protection limit"
    return None


def _check_win_rate(row: pd.Series) -> str | None:
    v = row["live_win_rate"]
    if v <= 45.0:
        return f"win_rate {v:.1f} % ≤ 45 % minimum"
    return None


def _check_profit_factor(row: pd.Series) -> str | None:
    v = row["live_profit_factor"]
    if not math.isfinite(v):
        return None   # infinite PF means zero losses — that is fine
    if v <= 1.3:
        return f"profit_factor {v:.3f} ≤ 1.3 minimum"
    return None


def _check_trade_count(row: pd.Series) -> str | None:
    v = int(row["live_total_trades"])
    if v < 20:
        return f"only {v} trades in live period — need ≥ 20 for statistical validity"
    return None


def _check_daily_band(row: pd.Series) -> str | None:
    """95 % of days (avg ± 2·std) must fall inside [$50, $200]."""
    avg = row["live_avg_daily_pnl"]
    std = row["live_std_daily_pnl"]
    lower = avg - 2.0 * std
    upper = avg + 2.0 * std
    if lower < 50.0:
        return (
            f"95 % daily band lower bound ${lower:.2f} < $50 "
            f"(avg=${avg:.2f}, std=${std:.2f})"
        )
    if upper > 200.0:
        return (
            f"95 % daily band upper bound ${upper:.2f} > $200 "
            f"(avg=${avg:.2f}, std=${std:.2f})"
        )
    return None


def _check_loss_floor(row: pd.Series) -> str | None:
    """No single day loss should exceed $75: avg − 3·std must stay above −$75."""
    avg   = row["live_avg_daily_pnl"]
    std   = row["live_std_daily_pnl"]
    floor = avg - 3.0 * std
    if floor <= -75.0:
        return (
            f"daily loss floor ${floor:.2f} ≤ −$75 "
            f"(avg=${avg:.2f}, std=${std:.2f}) — tail-risk of large losing day"
        )
    return None


# Ordered list of (rule_name, filter_fn) — order determines which failure is reported first
FILTERS: list[tuple[str, object]] = [
    ("avg_daily_pnl in [$100,$150]", _check_avg_daily_pnl),
    ("std_daily_pnl < $50",          _check_std_daily_pnl),
    ("consistency_score < 0.30",     _check_consistency_score),
    ("max_drawdown ≤ $500",          _check_max_drawdown),
    ("win_rate > 45%",               _check_win_rate),
    ("profit_factor > 1.3",          _check_profit_factor),
    ("trade_count ≥ 20",             _check_trade_count),
    ("95% daily band $50-$200",      _check_daily_band),
    ("daily loss floor > -$75",      _check_loss_floor),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(row: pd.Series) -> tuple[bool, str | None]:
    """
    Run all filters against one strategy row.
    Returns (passed: bool, first_failure_reason: str | None).
    """
    for _rule_name, fn in FILTERS:
        reason = fn(row)
        if reason is not None:
            return False, reason
    return True, None


def _print_strategy_result(
    name: str,
    passed: bool,
    reason: str | None,
    row: pd.Series,
) -> None:
    status = "PASSED ✓" if passed else "FAILED ✗"
    avg    = row.get("live_avg_daily_pnl",     float("nan"))
    std    = row.get("live_std_daily_pnl",     float("nan"))
    cs     = row.get("live_consistency_score", float("nan"))
    dd     = row.get("live_max_drawdown",      float("nan"))
    wr     = row.get("live_win_rate",          float("nan"))

    print(f"  {status}  {name}")
    print(
        f"           avg_daily=${avg:>8.2f}  std=${std:>6.2f}  "
        f"consistency={cs:.4f}  drawdown=${dd:.2f}  win={wr:.1f}%"
    )
    if not passed and reason:
        print(f"           Reason: {reason}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load results ───────────────────────────────────────────────────────────
    if not os.path.isfile(INPUT_FILE):
        print(
            f"ERROR: {INPUT_FILE} not found.\n"
            "Run  python agents/backtest_engine.py  first."
        )
        sys.exit(1)

    df_all = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df_all):,} rows from {INPUT_FILE}")

    # Keep only strategies that passed walk-forward testing
    df = df_all[df_all["passed_walkforward"] == True].copy()   # noqa: E712
    print(f"Walk-forward passed: {len(df)} / {len(df_all)}\n")

    if df.empty:
        print("No strategies passed walk-forward testing. Nothing to filter.")
        sys.exit(0)

    # ── Apply filters ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("Applying prop-firm risk filters (live-simulation period)")
    print("=" * 70 + "\n")

    passing_rows : list[pd.Series]  = []
    failure_tally: dict[str, int]   = {}   # rule_name → count
    n_passed = 0
    n_failed = 0

    for _, row in df.iterrows():
        name           = str(row.get("strategy", "unknown"))
        passed, reason = evaluate(row)

        _print_strategy_result(name, passed, reason, row)

        if passed:
            n_passed += 1
            passing_rows.append(row)
        else:
            n_failed += 1
            # Identify the rule name from the reason prefix
            for rule_name, fn in FILTERS:
                if fn(row) == reason:
                    failure_tally[rule_name] = failure_tally.get(rule_name, 0) + 1
                    break

    # ── Save approved strategies ───────────────────────────────────────────────
    if passing_rows:
        approved_df = pd.DataFrame(passing_rows)
        approved_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Approved strategies saved → {OUTPUT_FILE}")
    else:
        print("No strategies passed all filters — risk_approved.csv not written.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RISK GUARDIAN SUMMARY")
    print("=" * 70)
    print(f"  Total strategies tested : {len(df)}")
    print(f"  Strategies passed       : {n_passed}")
    print(f"  Strategies failed       : {n_failed}")

    if failure_tally:
        top_rule, top_count = max(failure_tally.items(), key=lambda x: x[1])
        print(f"  Top reason for failure  : '{top_rule}'  ({top_count} strategies)")
        print()
        print("  Failure breakdown:")
        for rule, count in sorted(failure_tally.items(), key=lambda x: -x[1]):
            print(f"    {count:>3}×  {rule}")

    print("=" * 70)


if __name__ == "__main__":
    main()
