"""
Candle + tick data from Yahoo Finance  —  drop-in replacement for the broken
RisenFX history endpoint.

Exports
-------
get_candles(timeframe, count)  → bar_data dict or None
get_current_price()            → latest EURUSD mid (float) or None
"""

import logging
import yfinance as yf

log = logging.getLogger(__name__)

_YF_SYMBOL = "EURUSD=X"

# Map our numeric timeframe strings to yfinance interval codes
_INTERVAL_MAP = {
    "1":   "1m",
    "5":   "5m",
    "15":  "15m",
    "30":  "30m",
    "60":  "1h",
    "240": "4h",
    "1D":  "1d",
}

# How far back to fetch to guarantee `count` completed bars
_PERIOD_MAP = {
    "1m":  "1d",
    "5m":  "5d",
    "15m": "5d",
    "30m": "5d",
    "1h":  "60d",
    "4h":  "60d",
    "1d":  "1y",
}


# ── OHLCV bars ────────────────────────────────────────────────────────────────

def get_candles(timeframe: str = "5", count: int = 120) -> dict | None:
    """
    Fetch OHLCV candles for EURUSD from Yahoo Finance.

    Returns bar_data dict  {"t","o","h","l","c","v"}  or None on failure.
    """
    interval = _INTERVAL_MAP.get(str(timeframe), "5m")
    period   = _PERIOD_MAP.get(interval, "5d")

    try:
        ticker = yf.Ticker(_YF_SYMBOL)
        df = ticker.history(period=period, interval=interval)

        if df is None or df.empty:
            log.warning("yfinance returned empty DataFrame")
            return None

        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        # Drop the still-forming bar for intraday intervals
        if interval != "1d":
            df = df.iloc[:-1]

        df = df.tail(count)

        if len(df) < 55:
            log.warning(f"yfinance: only {len(df)} bars after filtering — need 55+")
            return None

        timestamps = [int(ts.timestamp()) for ts in df.index]

        bar_data = {
            "t": timestamps,
            "o": df["Open"].tolist(),
            "h": df["High"].tolist(),
            "l": df["Low"].tolist(),
            "c": df["Close"].tolist(),
            "v": df["Volume"].tolist(),
        }

        log.info(f"yfinance: {len(timestamps)} bars ({interval}) fetched for {_YF_SYMBOL}")
        return bar_data

    except Exception as exc:
        log.error(f"yfinance error: {exc}")
        return None


# ── Current price (for SL/TP monitor) ─────────────────────────────────────────

def get_current_price() -> float | None:
    """
    Return the latest EURUSD price for the price-monitoring SL/TP loop.

    Pulls the most recent 1-minute bar and uses its close.  Yahoo lags spot
    by a few seconds but is accurate enough for a 60 s monitor.
    """
    try:
        ticker = yf.Ticker(_YF_SYMBOL)
        df = ticker.history(period="1d", interval="1m")
        if df is None or df.empty:
            log.warning("yfinance: no 1m bars when fetching current price")
            return None
        price = float(df["Close"].dropna().iloc[-1])
        if price <= 0:
            log.warning(f"yfinance: invalid price {price}")
            return None
        return price
    except Exception as exc:
        log.error(f"get_current_price error: {exc}")
        return None
