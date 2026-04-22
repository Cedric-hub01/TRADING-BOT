"""
EMA 8 / 21 / 50 crossover strategy with ATR-based SL/TP.

Signal rules (Long Only):
  Entry  : EMA8 crosses above EMA21 on the last *completed* bar
            AND EMA8 > EMA50  AND  EMA21 > EMA50  (trend filter)
  SL     : entry_price − ATR_SL_MULT × ATR
  TP     : entry_price + ATR_TP_MULT × ATR  (default 3:1 RR)
"""

import logging
import pandas as pd

from config import (
    EMA_FAST, EMA_MED, EMA_SLOW,
    ATR_PERIOD, ATR_SL_MULT, ATR_TP_MULT,
)

log = logging.getLogger(__name__)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's Average True Range (alpha = 1/period, same as TradingView default)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ── DataFrame builder ─────────────────────────────────────────────────────────

def build_dataframe(bar_data: dict) -> pd.DataFrame | None:
    """
    Convert the raw TradeLocker barData dict to a DataFrame with indicators.

    bar_data expected keys: "o", "h", "l", "c", "v", "t"
    Returns None if bar_data is missing or has too few rows.
    """
    if not bar_data:
        return None

    timestamps = bar_data.get("t", [])
    if len(timestamps) < EMA_SLOW + 10:
        log.warning(f"Only {len(timestamps)} bars — need {EMA_SLOW + 10}+ for reliable signals")
        return None

    df = pd.DataFrame({
        "time":   bar_data["t"],
        "open":   pd.to_numeric(bar_data["o"], errors="coerce"),
        "high":   pd.to_numeric(bar_data["h"], errors="coerce"),
        "low":    pd.to_numeric(bar_data["l"], errors="coerce"),
        "close":  pd.to_numeric(bar_data["c"], errors="coerce"),
        "volume": pd.to_numeric(bar_data.get("v", [0] * len(timestamps)), errors="coerce"),
    })

    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    df["ema_fast"] = _ema(df["close"], EMA_FAST)
    df["ema_med"]  = _ema(df["close"], EMA_MED)
    df["ema_slow"] = _ema(df["close"], EMA_SLOW)
    df["atr"]      = _wilder_atr(df["high"], df["low"], df["close"], ATR_PERIOD)

    return df


# ── Signal generation ─────────────────────────────────────────────────────────

def get_signal(df: pd.DataFrame) -> tuple[str | None, float, float, float]:
    """
    Analyse the last two completed bars.

    Returns
    -------
    (signal, sl_price, tp_price, atr_value)
    signal is "buy" or None.
    """
    if df is None or len(df) < EMA_SLOW + 5:
        return None, 0.0, 0.0, 0.0

    # df.iloc[-1] is the *forming* candle — use [-2] (last completed) and [-3] (prev)
    prev = df.iloc[-3]
    last = df.iloc[-2]

    ema8_cross_up = (prev["ema_fast"] <= prev["ema_med"]) and (last["ema_fast"] > last["ema_med"])
    trend_bullish  = (last["ema_fast"] > last["ema_slow"]) and (last["ema_med"] > last["ema_slow"])

    entry_price = last["close"]
    atr_val     = last["atr"]

    if ema8_cross_up and trend_bullish:
        sl = round(entry_price - ATR_SL_MULT * atr_val, 5)
        tp = round(entry_price + ATR_TP_MULT * atr_val, 5)
        log.info(
            f"SIGNAL BUY | price={entry_price:.5f} "
            f"EMA8={last['ema_fast']:.5f} EMA21={last['ema_med']:.5f} EMA50={last['ema_slow']:.5f} "
            f"ATR={atr_val:.5f} SL={sl:.5f} TP={tp:.5f}"
        )
        return "buy", sl, tp, round(atr_val, 5)

    return None, 0.0, 0.0, round(atr_val, 5)
