"""
Backtest data loader — EURUSD from Yahoo Finance.

Yahoo's 1-minute feed is capped at ~7 days of history, so the backtester
supports multiple (interval, period) combinations.  Callers pick whatever
gives the best trade-off between resolution and history length.
"""

from __future__ import annotations

import logging
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_SYMBOL = "EURUSD=X"

# Maximum period Yahoo will actually serve for each interval.  Requesting more
# is silently clipped to the limit, so we just use these caps.
MAX_PERIOD = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h":  "730d",
    "1d":  "20y",
}


def fetch(interval: str = "5m", period: str | None = None) -> pd.DataFrame:
    """
    Return a DataFrame indexed by UTC timestamp with columns
    open / high / low / close / volume.
    """
    if period is None:
        period = MAX_PERIOD.get(interval, "60d")

    log.info(f"Fetching {_SYMBOL} interval={interval} period={period} …")
    df = yf.Ticker(_SYMBOL).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {interval}/{period}")

    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Normalise to UTC for session-filter strategies
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    log.info(
        f"Got {len(df)} bars  [{df.index.min()} .. {df.index.max()}]  "
        f"span={df.index.max() - df.index.min()}"
    )
    return df
