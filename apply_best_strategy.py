"""
Apply the winning backtest strategy to config.py.

Reads best_strategy.json (produced by backtest.py) and rewrites the three
strategy-dispatch fields in config.py:

    STRATEGY_TYPE   = "..."
    STRATEGY_PARAMS = {...}
    DIRECTION       = "long" | "short" | "both"

Usage
-----
    python apply_best_strategy.py                       # uses best_strategy.json
    python apply_best_strategy.py path/to/result.json   # custom JSON

The JSON may be either:
  • the dict written by backtest.py, {"best": {...}, ...}
  • a bare strategy row (has keys "strategy", "direction", "params")
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path


_CONFIG = Path(__file__).parent / "config.py"


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "best" in data:
        return data["best"]
    if "top10" in data and data["top10"]:
        return data["top10"][0]
    if "strategy" in data and "params" in data:
        return data
    raise ValueError(f"Cannot find a strategy row inside {path}")


def _rewrite(config_text: str, key: str, literal: str) -> str:
    """
    Replace the value of `key = ...` in config.py with `literal`.  We capture
    up to the end of the first logical line, which means dict literals must
    fit on one line — apply_best writes them that way.
    """
    pattern = re.compile(rf"^({re.escape(key)}\s*=).*$", re.MULTILINE)
    if not pattern.search(config_text):
        raise ValueError(f"'{key}' not found in config.py — was it removed?")
    return pattern.sub(rf"\1 {literal}", config_text, count=1)


def main(argv: list[str]):
    src = Path(argv[1]) if len(argv) > 1 else Path("best_strategy.json")
    if not src.exists():
        print(f"ERROR: {src} not found — run backtest.py first")
        sys.exit(2)

    best = _load(src)
    strategy_name = best["strategy"]
    direction     = best["direction"]
    params        = best["params"]
    strat_type    = params.get("type") or best.get("type")
    if not strat_type:
        print("ERROR: strategy row has no 'type' field; cannot map to live strategy")
        sys.exit(3)

    # Strip the "type" key — config stores params WITHOUT the dispatcher tag
    live_params = {k: v for k, v in params.items() if k != "type"}

    # Backup + load current config
    backup = _CONFIG.with_suffix(".py.bak")
    shutil.copy2(_CONFIG, backup)
    text = _CONFIG.read_text()

    text = _rewrite(text, "STRATEGY_TYPE",   json.dumps(strat_type))
    text = _rewrite(text, "STRATEGY_PARAMS", json.dumps(live_params, sort_keys=True))
    text = _rewrite(text, "DIRECTION",       json.dumps(direction))
    _CONFIG.write_text(text)

    print("─" * 60)
    print(f"Applied winning strategy → config.py   (backup: {backup.name})")
    print("─" * 60)
    print(f"  STRATEGY_TYPE   = {strat_type!r}")
    print(f"  STRATEGY_PARAMS = {live_params}")
    print(f"  DIRECTION       = {direction!r}")
    print()
    print(f"  (was: {strategy_name})")
    print(
        f"  trades={best.get('trades','?')}  "
        f"net=${best.get('net_profit',0):.2f}  "
        f"PF={best.get('profit_factor',0)}  "
        f"win%={best.get('win_rate',0):.1f}  "
        f"sharpe={best.get('sharpe',0):.2f}"
    )
    print("─" * 60)
    print("Restart the bot (python bot.py) to pick up the new strategy.")


if __name__ == "__main__":
    main(sys.argv)
