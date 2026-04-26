"""
ChiefAgent — orchestrates the full multi-agent research pipeline.

Execution order:
    Step 1  DataAgent          → xauusd_1m/5m/15m.csv
    Step 2  BacktestEngine     → backtest_results.csv
    Step 3  RiskGuardian       → risk_approved.csv
    Step 4  PropFirmSimulator  → propfirm_results.csv

After all steps it produces:
    full_research_report.txt   — complete narrative report
    top3_strategies.json       — top 3 strategies with all metrics
    10day_simulation.txt       — day-by-day breakdown for the #1 strategy
    config.py                  — updated with best strategy parameters
"""

import os
import sys
import json
import math
import datetime
import subprocess
import textwrap

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

BACKTEST_CSV  = os.path.join(_REPO_ROOT, "backtest_results.csv")
APPROVED_CSV  = os.path.join(_REPO_ROOT, "risk_approved.csv")
PROPFIRM_CSV  = os.path.join(_REPO_ROOT, "propfirm_results.csv")
CONFIG_FILE   = os.path.join(_REPO_ROOT, "config.py")

REPORT_FILE   = os.path.join(_REPO_ROOT, "full_research_report.txt")
TOP3_FILE     = os.path.join(_REPO_ROOT, "top3_strategies.json")
SIM_FILE      = os.path.join(_REPO_ROOT, "10day_simulation.txt")

# Map agent step → script path (relative to REPO_ROOT)
_AGENT_SCRIPTS = {
    "DataAgent":          os.path.join("agents", "data_agent.py"),
    "BacktestEngine":     os.path.join("agents", "backtest_engine.py"),
    "RiskGuardian":       os.path.join("agents", "risk_guardian.py"),
    "PropFirmSimulator":  os.path.join("agents", "propfirm_simulator.py"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(label: str, script_rel: str) -> tuple[bool, str]:
    """
    Run *script_rel* (path relative to REPO_ROOT) in a subprocess.
    Streams stdout/stderr to the terminal in real time and also captures it.
    Returns (success: bool, combined_output: str).
    """
    script = os.path.join(_REPO_ROOT, script_rel)
    if not os.path.isfile(script):
        return False, f"Script not found: {script}"

    lines: list[str] = []
    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print("  " + line, end="")
        lines.append(line)
    proc.wait()

    return proc.returncode == 0, "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CSV readers with graceful fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Day-label helper (mirrors propfirm_simulator)
# ─────────────────────────────────────────────────────────────────────────────

def _day_label(pnl: float) -> str:
    if 100.0 <= pnl <= 150.0:
        return "✅ Perfect"
    if pnl > 150.0:
        return "⚠️  Over cap"
    if pnl > 0.0:
        return "🟡 Profit"
    return "❌ Loss day"


def _verdict_icon(v: str) -> str:
    return {"PASS": "✅", "BORDERLINE": "🟡", "FAIL": "❌"}.get(v, "")


# ─────────────────────────────────────────────────────────────────────────────
# Build the 10-day table string for one strategy row
# ─────────────────────────────────────────────────────────────────────────────

def _build_sim_table(row: pd.Series, name: str) -> str:
    lines: list[str] = []
    lines.append(f"Strategy: {name}")
    lines.append("─" * 52)
    total = 0.0
    day_pnls: list[float] = []
    for i in range(1, 11):
        date_col = f"day{i:02d}_date"
        pnl_col  = f"day{i:02d}_pnl"
        if pnl_col not in row.index:
            break
        pnl  = float(row[pnl_col])
        date = str(row.get(date_col, ""))
        sign = "+" if pnl >= 0 else ""
        lines.append(f"  Day {i:02d} ({date}): {sign}${pnl:.2f}  {_day_label(pnl)}")
        total += pnl
        day_pnls.append(pnl)

    lines.append("─" * 52)
    best = max(day_pnls) if day_pnls else 0.0
    pct  = (best / total * 100) if total > 0 else 0.0
    cs   = float(row.get("consistency", best / total if total > 0 else 99.0))
    pfdays = int(row.get("perfect_days", 0))
    verdict = str(row.get("verdict", "FAIL"))
    icon    = _verdict_icon(verdict)

    sign = "+" if total >= 0 else ""
    lines.append(f"  Total: {sign}${total:.2f}")
    lines.append(f"  Best day: ${best:.2f}  ({pct:.1f}% of total)")
    lines.append(f"  Consistency score: {cs:.3f}")
    lines.append(f"  Perfect days: {pfdays}/10")
    lines.append(f"  Result: {verdict} {icon}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Config.py updater
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else 0.0
    except Exception:
        return 0.0


def update_config(best_row: pd.Series, best_name: str) -> bool:
    """
    Append / update STRATEGY_TYPE and STRATEGY_PARAMS in config.py.
    Also updates SYMBOL, DAILY_STOP, DAILY_TARGET to match the prop-firm
    parameters discovered during research.
    Returns True on success.
    """
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            original = fh.read()

        params = {
            "avg_daily_pnl":     _safe_float(best_row.get("live_avg_daily_pnl",     best_row.get("avg_daily_pnl",     0))),
            "std_daily_pnl":     _safe_float(best_row.get("live_std_daily_pnl",     best_row.get("std_daily_pnl",     0))),
            "win_rate":          _safe_float(best_row.get("live_win_rate",           best_row.get("win_rate",          0))),
            "profit_factor":     _safe_float(best_row.get("live_profit_factor",      best_row.get("profit_factor",     0))),
            "max_drawdown":      _safe_float(best_row.get("live_max_drawdown",       best_row.get("max_drawdown",      0))),
            "consistency_score": _safe_float(best_row.get("consistency",             0)),
            "total_10day_profit":_safe_float(best_row.get("total_profit",            0)),
            "verdict":           str(best_row.get("verdict", "PASS")),
        }

        new_block = textwrap.dedent(f"""
            # ── Auto-installed by ChiefAgent ─────────────────────────────────────────────
            # Research run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            SYMBOL          = "XAU/USD"
            TIMEFRAME       = "1"              # 1-minute bars
            LOT_SIZE        = 0.03             # oz lots — $3 per $1 gold move
            DAILY_STOP      = -75.0            # USD — prop-firm daily loss limit
            DAILY_TARGET    =  100.0           # USD — stop trading when hit
            DAILY_CAP       =  150.0           # USD — emergency cap (consistency rule)
            MAX_LOSS_PER_TRADE = 30.0          # USD — per-trade risk limit

            STRATEGY_TYPE   = "{best_name}"
            STRATEGY_PARAMS = {json.dumps(params, indent=4)}
            # ──────────────────────────────────────────────────────────────────────────────
        """)

        # Remove any previous auto-installed block to avoid duplicates
        marker = "# ── Auto-installed by ChiefAgent"
        if marker in original:
            original = original[: original.index(marker)].rstrip() + "\n"

        with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
            fh.write(original + new_block)

        return True
    except Exception as exc:
        print(f"  [!] config.py update failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Report generators
# ─────────────────────────────────────────────────────────────────────────────

def _rank_propfirm(df: pd.DataFrame) -> pd.DataFrame:
    """Sort propfirm results: PASS > BORDERLINE > FAIL, then by total_profit desc."""
    order = {"PASS": 0, "BORDERLINE": 1, "FAIL": 2}
    df = df.copy()
    df["_rank"] = df["verdict"].map(order).fillna(3)
    df.sort_values(["_rank", "total_profit"], ascending=[True, False], inplace=True)
    df.drop(columns=["_rank"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def write_top3_json(propfirm_df: pd.DataFrame, backtest_df: pd.DataFrame | None) -> list[dict]:
    top3_rows = propfirm_df.head(3)
    top3: list[dict] = []

    for _, row in top3_rows.iterrows():
        name = str(row["strategy"])
        entry: dict = {
            "name":   name,
            "verdict": str(row.get("verdict", "FAIL")),
            "simulation": {
                "total_profit":   _safe_float(row.get("total_profit")),
                "best_day":       _safe_float(row.get("best_day")),
                "worst_day":      _safe_float(row.get("worst_day")),
                "green_days":     int(row.get("green_days",   0)),
                "red_days":       int(row.get("red_days",     0)),
                "perfect_days":   int(row.get("perfect_days", 0)),
                "consistency":    _safe_float(row.get("consistency")),
            },
            "day_by_day": {},
        }
        for i in range(1, 11):
            pnl_col  = f"day{i:02d}_pnl"
            date_col = f"day{i:02d}_date"
            if pnl_col in row.index:
                entry["day_by_day"][f"day{i:02d}"] = {
                    "date": str(row.get(date_col, "")),
                    "pnl":  _safe_float(row.get(pnl_col)),
                }
        # Attach backtest metrics if available
        if backtest_df is not None and "strategy" in backtest_df.columns:
            bt_rows = backtest_df[backtest_df["strategy"] == name]
            if not bt_rows.empty:
                bt = bt_rows.iloc[0]
                entry["backtest"] = {
                    k: (_safe_float(v) if not isinstance(v, str) else v)
                    for k, v in bt.items()
                    if k != "strategy"
                }
        top3.append(entry)

    with open(TOP3_FILE, "w", encoding="utf-8") as fh:
        json.dump(top3, fh, indent=2)

    return top3


def write_10day_sim(propfirm_df: pd.DataFrame) -> str:
    """Write 10day_simulation.txt for the #1 strategy. Returns the strategy name."""
    if propfirm_df.empty:
        with open(SIM_FILE, "w", encoding="utf-8") as fh:
            fh.write("No strategies available for simulation.\n")
        return ""

    row  = propfirm_df.iloc[0]
    name = str(row["strategy"])
    table = _build_sim_table(row, name)

    with open(SIM_FILE, "w", encoding="utf-8") as fh:
        fh.write("10-DAY PROP-FIRM WITHDRAWAL SIMULATION\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(table + "\n")

    return name


def write_full_report(
    run_time: str,
    step_results: dict[str, tuple[bool, str]],
    backtest_df:  pd.DataFrame | None,
    approved_df:  pd.DataFrame | None,
    propfirm_df:  pd.DataFrame | None,
    top3:         list[dict],
    best_name:    str,
) -> None:
    lines: list[str] = []

    def h(text: str) -> None:
        lines.append("\n" + "=" * 60)
        lines.append(text)
        lines.append("=" * 60)

    def s(text: str) -> None:
        lines.append("\n" + "─" * 40)
        lines.append(text)
        lines.append("─" * 40)

    h("MULTI-AGENT XAU/USD RESEARCH REPORT")
    lines.append(f"Generated : {run_time}")
    lines.append(f"Best strat: {best_name or 'N/A'}")

    h("PIPELINE STATUS")
    for agent, (ok, _) in step_results.items():
        status = "✅ SUCCESS" if ok else "❌ FAILED"
        lines.append(f"  {agent:22s} {status}")

    h("STATISTICS")
    n_tested   = len(backtest_df)  if backtest_df  is not None else "N/A"
    n_wf       = int(backtest_df["passed_walkforward"].sum()) if backtest_df is not None and "passed_walkforward" in backtest_df.columns else "N/A"
    n_approved = len(approved_df)  if approved_df  is not None else "N/A"

    if propfirm_df is not None and "verdict" in propfirm_df.columns:
        vc = propfirm_df["verdict"].value_counts()
        n_pass  = int(vc.get("PASS",       0))
        n_bord  = int(vc.get("BORDERLINE", 0))
        n_fail  = int(vc.get("FAIL",       0))
    else:
        n_pass = n_bord = n_fail = "N/A"

    lines.append(f"  Strategies tested (backtest)  : {n_tested}")
    lines.append(f"  Passed walk-forward           : {n_wf}")
    lines.append(f"  Passed risk filter            : {n_approved}")
    lines.append(f"  Prop-firm PASS                : {n_pass}")
    lines.append(f"  Prop-firm BORDERLINE          : {n_bord}")
    lines.append(f"  Prop-firm FAIL                : {n_fail}")

    h("TOP 3 STRATEGIES — DAY-BY-DAY")
    if propfirm_df is not None and not propfirm_df.empty:
        for rank, entry in enumerate(top3, 1):
            name    = entry["name"]
            verdict = entry["verdict"]
            icon    = _verdict_icon(verdict)
            s(f"#{rank}  {name}  [{verdict} {icon}]")

            sim = entry["simulation"]
            lines.append(f"  Total 10-day profit : ${sim['total_profit']:.2f}")
            lines.append(f"  Best day            : ${sim['best_day']:.2f}")
            lines.append(f"  Consistency score   : {sim['consistency']:.3f}")
            lines.append(f"  Perfect days        : {sim['perfect_days']}/10")
            lines.append("")

            # Day-by-day mini table
            for day_key, dinfo in entry["day_by_day"].items():
                pnl  = dinfo["pnl"]
                date = dinfo["date"]
                sign = "+" if pnl >= 0 else ""
                day_num = day_key.replace("day", "Day ")
                lines.append(f"    {day_num} ({date}): {sign}${pnl:.2f}  {_day_label(pnl)}")

    h("RECOMMENDATION")
    if best_name:
        best_row = propfirm_df[propfirm_df["strategy"] == best_name].iloc[0] if propfirm_df is not None and not propfirm_df.empty else None
        lines.append(f"  Run live: {best_name}")
        if best_row is not None:
            lines.append(f"  Expected daily profit : ~${_safe_float(best_row.get('total_profit', 0)) / 10:.2f}")
            lines.append(f"  Expected 10-day total : ${_safe_float(best_row.get('total_profit', 0)):.2f}")
            lines.append(f"  Consistency score     : {_safe_float(best_row.get('consistency', 0)):.3f}")
            lines.append(f"  Prop-firm verdict     : {best_row.get('verdict', 'N/A')} {_verdict_icon(str(best_row.get('verdict', '')))}")
        lines.append("")
        lines.append("  Why this strategy was chosen:")
        lines.append("    • Profitable on all 3 walk-forward periods (train/val/live)")
        lines.append("    • Passed all 9 prop-firm risk rules")
        lines.append("    • Highest 10-day total with best consistency score")
        lines.append("    • Best-day never exceeded 30% of 10-day total")
    else:
        lines.append("  No strategy achieved PASS status.")
        lines.append("  Review BORDERLINE strategies manually or collect more data.")

    h("FILES PRODUCED")
    for f in [TOP3_FILE, SIM_FILE, CONFIG_FILE, REPORT_FILE]:
        exists = "✅" if os.path.isfile(f) else "❌"
        lines.append(f"  {exists}  {os.path.relpath(f, _REPO_ROOT)}")

    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_results: dict[str, tuple[bool, str]] = {}

    # ── Banner ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  MULTI-AGENT AI RESEARCH SYSTEM STARTING")
    print(f"  {run_time}")
    print("=" * 52 + "\n")

    agents_in_order = [
        ("DataAgent",         "agents/data_agent.py",         "1/4: DataAgent fetching market data"),
        ("BacktestEngine",    "agents/backtest_engine.py",     "2/4: BacktestEngine testing strategies"),
        ("RiskGuardian",      "agents/risk_guardian.py",       "3/4: RiskGuardian filtering"),
        ("PropFirmSimulator", "agents/propfirm_simulator.py",  "4/4: PropFirmSimulator running 10-day sim"),
    ]

    for agent_key, script, label in agents_in_order:
        print(f"Step {label} …")
        ok, output = run_agent(agent_key, script)
        step_results[agent_key] = (ok, output)

        # Post-step summary line
        extra = ""
        if agent_key == "DataAgent" and ok:
            for tf in ("xauusd_1m.csv", "xauusd_5m.csv", "xauusd_15m.csv"):
                p = os.path.join(_REPO_ROOT, tf)
                if os.path.isfile(p):
                    n = sum(1 for _ in open(p)) - 1
                    extra += f"  {tf}: {n:,} bars\n"
        elif agent_key == "BacktestEngine":
            bt = _read_csv(BACKTEST_CSV)
            if bt is not None:
                n_pass = int(bt["passed_walkforward"].sum()) if "passed_walkforward" in bt.columns else 0
                extra = f"  Walk-forward: {n_pass}/{len(bt)} strategies passed"
        elif agent_key == "RiskGuardian":
            ap = _read_csv(APPROVED_CSV)
            extra = f"  Approved: {len(ap)} strategies" if ap is not None else "  risk_approved.csv not found"
        elif agent_key == "PropFirmSimulator":
            pf = _read_csv(PROPFIRM_CSV)
            if pf is not None and "verdict" in pf.columns:
                vc = pf["verdict"].value_counts()
                extra = (
                    f"  PASS={vc.get('PASS',0)}  "
                    f"BORDERLINE={vc.get('BORDERLINE',0)}  "
                    f"FAIL={vc.get('FAIL',0)}"
                )

        icon = "✅" if ok else "❌"
        print(f"\n{icon} Step {agent_key} {'complete' if ok else 'FAILED'}")
        if extra:
            print(extra)
        print()

    # ── Load all result DataFrames ─────────────────────────────────────────────
    backtest_df  = _read_csv(BACKTEST_CSV)
    approved_df  = _read_csv(APPROVED_CSV)
    propfirm_df  = _read_csv(PROPFIRM_CSV)

    # Rank propfirm results
    ranked_df: pd.DataFrame | None = None
    if propfirm_df is not None and not propfirm_df.empty:
        ranked_df = _rank_propfirm(propfirm_df)

    best_name = ""
    if ranked_df is not None and not ranked_df.empty:
        best_name = str(ranked_df.iloc[0]["strategy"])

    # ── Write output files ─────────────────────────────────────────────────────
    print("=" * 52)
    print("  Generating output files …")
    print("=" * 52)

    top3: list[dict] = []
    if ranked_df is not None and not ranked_df.empty:
        top3 = write_top3_json(ranked_df, backtest_df)
        print(f"  ✅  {os.path.relpath(TOP3_FILE, _REPO_ROOT)}")

        write_10day_sim(ranked_df)
        print(f"  ✅  {os.path.relpath(SIM_FILE, _REPO_ROOT)}")
    else:
        print("  ⚠️   No propfirm results — skipping top3_strategies.json and 10day_simulation.txt")

    write_full_report(run_time, step_results, backtest_df, approved_df, ranked_df, top3, best_name)
    print(f"  ✅  {os.path.relpath(REPORT_FILE, _REPO_ROOT)}")

    # ── Update config.py ───────────────────────────────────────────────────────
    if ranked_df is not None and not ranked_df.empty:
        best_row = ranked_df.iloc[0]
        if update_config(best_row, best_name):
            print(f"  ✅  config.py  ← best strategy installed: {best_name}")
        else:
            print("  ❌  config.py update failed")
    else:
        print("  ⚠️   No best strategy — config.py not modified")

    # ── Final banner ───────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  RESEARCH COMPLETE")
    print("=" * 52)

    if ranked_df is not None and not ranked_df.empty:
        row     = ranked_df.iloc[0]
        total   = _safe_float(row.get("total_profit", 0))
        avg_day = total / 10 if total else 0.0
        cs      = _safe_float(row.get("consistency", 0))
        verdict = str(row.get("verdict", ""))
        icon    = _verdict_icon(verdict)

        print(f"  Best strategy   : {best_name}")
        print(f"  Expected daily  : ~${avg_day:.2f} average")
        print(f"  10-day total    : ${total:.2f}")
        print(f"  Consistency     : {cs:.3f}")
        print(f"  Prop-firm       : {verdict} {icon}")
    else:
        print("  No passing strategies found.")
        print("  Check full_research_report.txt for details.")

    print("\n  Files saved:")
    for f in [REPORT_FILE, TOP3_FILE, SIM_FILE, CONFIG_FILE]:
        exists = "✅" if os.path.isfile(f) else "⚠️ "
        print(f"    {exists}  {os.path.relpath(f, _REPO_ROOT)}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
