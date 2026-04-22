"""
Live-trading strategy dispatcher.

Reads STRATEGY_TYPE / STRATEGY_PARAMS / DIRECTION from config.py and delegates
signal generation to the shared library in backtest_strategies.py, so the live
bot and the backtester use *exactly* the same logic.

Return value: (side, sl_price, tp_price, atr_value)
    side is "buy" | "sell" | None
    sl/tp are absolute prices derived from ATR (1.5× SL, 3.0× TP)
"""

import logging
import pandas as pd

from config import (
    STRATEGY_TYPE, STRATEGY_PARAMS, DIRECTION,
    ATR_PERIOD, ATR_SL_MULT, ATR_TP_MULT,
)
import backtest_strategies as lib

log = logging.getLogger(__name__)


_BUILDERS = {
    "ema_cross":        lambda p: lib.ema_cross(p["fast"], p["med"], p["slow"], p.get("trend_filter", False)),
    "macd":             lambda p: lib.macd_cross(p["fast"], p["slow"], p["signal"]),
    "supertrend":       lambda p: lib.supertrend_flip(p["period"], p["mult"]),
    "ichimoku":         lambda p: lib.ichimoku_tk(p["tenkan"], p["kijun"], p["senkou"]),
    "rsi_reversion":    lambda p: lib.rsi_reversion(p["period"], p["os"], p["ob"]),
    "bb_reversion":     lambda p: lib.bb_reversion(p["period"], p["k"]),
    "bb_breakout":      lambda p: lib.bb_breakout(p["period"], p["k"]),
    "stoch_reversion":  lambda p: lib.stoch_reversion(p["k"], p["d"], p["smooth"], p["os"], p["ob"]),
    "ema_adx":          lambda p: lib.ema_cross_adx(p["fast"], p["slow"], p["adx_period"], p["adx_min"]),
    "session_breakout": lambda p: lib.session_breakout(
        p["session"], p["lookback_minutes"], p["duration_minutes"], p["bar_minutes"]
    ),
}


def _build_strategy():
    try:
        return _BUILDERS[STRATEGY_TYPE](STRATEGY_PARAMS)
    except KeyError:
        raise RuntimeError(f"Unknown STRATEGY_TYPE '{STRATEGY_TYPE}' in config.py")


# ── DataFrame builder ─────────────────────────────────────────────────────────

def build_dataframe(bar_data: dict) -> pd.DataFrame | None:
    """
    Convert raw bar_data (from market_data.get_candles) to a DataFrame with a
    UTC DatetimeIndex and an ATR column.  Returns None if too few bars.
    """
    if not bar_data:
        return None

    t = bar_data.get("t", [])
    if len(t) < 60:
        log.warning(f"Only {len(t)} bars — need 60+ for reliable signals")
        return None

    df = pd.DataFrame({
        "time":   t,
        "open":   pd.to_numeric(bar_data["o"], errors="coerce"),
        "high":   pd.to_numeric(bar_data["h"], errors="coerce"),
        "low":    pd.to_numeric(bar_data["l"], errors="coerce"),
        "close":  pd.to_numeric(bar_data["c"], errors="coerce"),
        "volume": pd.to_numeric(bar_data.get("v", [0] * len(t)), errors="coerce"),
    })
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # DatetimeIndex is required by session-filter strategies
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)

    df["atr"] = lib.atr(df, ATR_PERIOD)
    return df


# ── Signal ────────────────────────────────────────────────────────────────────

def get_signal(df: pd.DataFrame) -> tuple[str | None, float, float, float]:
    if df is None or len(df) < 60:
        return None, 0.0, 0.0, 0.0

    strat = _build_strategy()
    try:
        entries = strat.signals(df)
    except Exception as exc:
        log.error(f"Strategy {STRATEGY_TYPE} raised: {exc}")
        return None, 0.0, 0.0, 0.0

    last = df.iloc[-2]            # last *completed* bar
    price   = float(last["close"])
    atr_val = float(last["atr"])

    if not (atr_val > 0):
        return None, 0.0, 0.0, 0.0

    go_long  = DIRECTION in ("long", "both")  and bool(entries["long_entry"].iloc[-2])
    go_short = DIRECTION in ("short", "both") and bool(entries["short_entry"].iloc[-2])

    if go_long:
        sl = round(price - ATR_SL_MULT * atr_val, 5)
        tp = round(price + ATR_TP_MULT * atr_val, 5)
        log.info(
            f"SIGNAL BUY [{STRATEGY_TYPE}] | price={price:.5f} ATR={atr_val:.5f} "
            f"SL={sl:.5f} TP={tp:.5f}"
        )
        return "buy", sl, tp, round(atr_val, 5)

    if go_short:
        sl = round(price + ATR_SL_MULT * atr_val, 5)
        tp = round(price - ATR_TP_MULT * atr_val, 5)
        log.info(
            f"SIGNAL SELL [{STRATEGY_TYPE}] | price={price:.5f} ATR={atr_val:.5f} "
            f"SL={sl:.5f} TP={tp:.5f}"
        )
        return "sell", sl, tp, round(atr_val, 5)

    return None, 0.0, 0.0, round(atr_val, 5)
