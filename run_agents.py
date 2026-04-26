"""run_agents.py — single entry point for the XAU/USD multi-agent research system."""

import os, sys, subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))

_AGENTS = [
    "agents/data_agent.py", "agents/backtest_engine.py",
    "agents/risk_guardian.py", "agents/propfirm_simulator.py",
    "agents/chief_agent.py",
]
_DATA_CSVS = [os.path.join(_ROOT, f) for f in ("xauusd_1m.csv", "xauusd_5m.csv", "xauusd_15m.csv")]


def _run(rel: str) -> bool:
    return subprocess.run([sys.executable, os.path.join(_ROOT, rel)], cwd=_ROOT).returncode == 0


def _importable(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def _check() -> None:
    bad_pkgs  = [p for p in ("requests", "pandas", "numpy") if not _importable(p)]
    bad_files = [f for f in _AGENTS if not os.path.isfile(os.path.join(_ROOT, f))]
    if bad_pkgs:
        print("Missing packages — run:  pip install " + " ".join(bad_pkgs))
    if bad_files:
        print("Missing agent files:\n  " + "\n  ".join(bad_files))
    if bad_pkgs or bad_files:
        sys.exit(1)


def _ask_fetch() -> bool:
    data_ok = all(os.path.isfile(f) for f in _DATA_CSVS)
    try:
        ans = input("Download fresh XAU/USD data from Twelve Data? (y/n): ").strip().lower()
    except EOFError:
        ans = "y"
    if ans == "y":
        return True
    if not data_ok:
        print("No local CSV files found — downloading data anyway.")
        return True
    return False


def main() -> None:
    print("\n" + "=" * 52)
    print("  XAUUSD MULTI-AGENT RESEARCH SYSTEM v1.0")
    print("  Goal  : Find best strategy for $10K prop firm")
    print("  Target: $100–$150 profit per day for 10 days")
    print("=" * 52 + "\n")

    _check()

    if _ask_fetch():
        print("\nFetching fresh market data …\n")
        if not _run("agents/data_agent.py"):
            print("WARNING: DataAgent reported errors — continuing with existing files.")

    print("\nLaunching ChiefAgent …\n")
    _run("agents/chief_agent.py")

    print("\n" + "=" * 52)
    print("  RESEARCH COMPLETE")
    print("=" * 52)
    print("  Results:")
    for f in ("full_research_report.txt", "top3_strategies.json", "10day_simulation.txt", "bot.log"):
        print(f"    - {f}")
    print("\n  To start trading with the best strategy run:")
    print("    python bot.py")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nResearch stopped by user.")
        sys.exit(0)
