"""
Market data for the live bot.

Primary  : Twelve Data  (real-time 1-min EURUSD, free tier).
Fallback : Yahoo Finance (~15-min delay — used if TD fails, is rate-limited,
                          or the API key is missing).

Both code paths return the same `bar_data` dict the rest of the bot expects:
    {"t": [...], "o": [...], "h": [...], "l": [...], "c": [...], "v": [...]}
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import requests
import yfinance as yf

from config import TWELVE_DATA_API_KEY

log = logging.getLogger(__name__)

# ── Twelve Data config ────────────────────────────────────────────────────────
_TD_SYMBOL     = "EUR/USD"
_TD_SERIES_URL = "https://api.twelvedata.com/time_series"
_TD_PRICE_URL  = "https://api.twelvedata.com/price"
_TD_INTERVAL_MAP = {
    "1":   "1min",
    "5":   "5min",
    "15":  "15min",
    "30":  "30min",
    "60":  "1h",
    "240": "4h",
    "1D":  "1day",
}

# If Twelve Data rejects a call (rate-limit, auth, bad plan) stop hitting it for
# this long — subsequent calls go straight to Yahoo during the cool-down.
_TD_COOLDOWN_SEC = 300
_td_blocked_until: float = 0.0


def _td_usable() -> bool:
    return bool(TWELVE_DATA_API_KEY) and time.time() >= _td_blocked_until


def _td_cooldown(reason: str):
    global _td_blocked_until
    _td_blocked_until = time.time() + _TD_COOLDOWN_SEC
    log.warning(
        f"Twelve Data cooling down for {_TD_COOLDOWN_SEC}s  ({reason}) — "
        "will use Yahoo in the meantime"
    )


# ── Yahoo (fallback) config ───────────────────────────────────────────────────
_YF_SYMBOL       = "EURUSD=X"
_YF_INTERVAL_MAP = {
    "1":   "1m",
    "5":   "5m",
    "15":  "15m",
    "30":  "30m",
    "60":  "1h",
    "240": "4h",
    "1D":  "1d",
}
_YF_PERIOD_MAP = {
    "1m":  "1d",
    "5m":  "5d",
    "15m": "5d",
    "30m": "5d",
    "1h":  "60d",
    "4h":  "60d",
    "1d":  "1y",
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_candles(timeframe: str = "5", count: int = 120) -> dict | None:
    """
    Fetch OHLCV candles.  Tries Twelve Data first; falls back to Yahoo.
    """
    if _td_usable():
        data = _td_candles(timeframe, count)
        if data is not None:
            return data
        # _td_candles already set the cool-down on failure

    if not TWELVE_DATA_API_KEY:
        log.debug("TWELVE_DATA_API_KEY not set — using Yahoo Finance")

    return _yf_candles(timeframe, count)


def get_current_price() -> float | None:
    """
    Latest EURUSD price.  Tries Twelve Data /price first; falls back to Yahoo.
    """
    if _td_usable():
        price = _td_current_price()
        if price is not None:
            return price

    return _yf_current_price()


# ── Twelve Data implementations ───────────────────────────────────────────────

def _td_candles(timeframe: str, count: int) -> dict | None:
    interval = _TD_INTERVAL_MAP.get(str(timeframe), "5min")
    outputsize = max(int(count), 60)
    try:
        resp = requests.get(
            _TD_SERIES_URL,
            params={
                "symbol":     _TD_SYMBOL,
                "interval":   interval,
                "outputsize": outputsize,
                "timezone":   "UTC",
                "apikey":     TWELVE_DATA_API_KEY,
            },
            timeout=10,
        )
    except Exception as exc:
        _td_cooldown(f"request exception: {exc}")
        return None

    if resp.status_code == 429:
        _td_cooldown("HTTP 429 rate-limited")
        return None
    if resp.status_code != 200:
        _td_cooldown(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        body = resp.json()
    except Exception as exc:
        _td_cooldown(f"invalid JSON: {exc}")
        return None

    if body.get("status") != "ok":
        _td_cooldown(f"API error: code={body.get('code')} msg={body.get('message','?')[:160]}")
        return None

    values = body.get("values") or []
    if not values:
        log.warning("Twelve Data returned empty 'values'")
        return None

    # TD returns newest-first — reverse to oldest-first so downstream
    # `df.sort_values("time")` is a no-op.
    values = list(reversed(values))

    t, o, h, l, c, v = [], [], [], [], [], []
    for row in values:
        try:
            # Format: "YYYY-MM-DD HH:MM:SS"  (UTC because we passed timezone=UTC)
            dt = datetime.fromisoformat(row["datetime"].replace(" ", "T"))
            dt = dt.replace(tzinfo=timezone.utc)
            t.append(int(dt.timestamp()))
            o.append(float(row["open"]))
            h.append(float(row["high"]))
            l.append(float(row["low"]))
            c.append(float(row["close"]))
            v.append(float(row.get("volume") or 0))
        except Exception as exc:
            log.debug(f"Twelve Data row skipped: {exc}  row={row}")
            continue

    if not t:
        log.warning("Twelve Data: all rows failed to parse")
        return None

    log.info(f"Twelve Data: {len(t)} bars ({interval}) for {_TD_SYMBOL} [latest={datetime.fromtimestamp(t[-1], tz=timezone.utc)}]")
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


def _td_current_price() -> float | None:
    try:
        resp = requests.get(
            _TD_PRICE_URL,
            params={"symbol": _TD_SYMBOL, "apikey": TWELVE_DATA_API_KEY},
            timeout=8,
        )
    except Exception as exc:
        _td_cooldown(f"price request exception: {exc}")
        return None

    if resp.status_code == 429:
        _td_cooldown("HTTP 429 on /price")
        return None
    if resp.status_code != 200:
        _td_cooldown(f"HTTP {resp.status_code} on /price: {resp.text[:160]}")
        return None

    try:
        body = resp.json()
    except Exception as exc:
        _td_cooldown(f"/price invalid JSON: {exc}")
        return None

    if body.get("status") == "error":
        _td_cooldown(f"/price API error: {body.get('message','?')[:160]}")
        return None

    try:
        price = float(body["price"])
    except (KeyError, TypeError, ValueError) as exc:
        log.warning(f"Twelve Data /price missing/bad 'price' field: {exc}  body={body}")
        return None

    if price <= 0:
        log.warning(f"Twelve Data /price returned non-positive {price}")
        return None

    return price


# ── Yahoo implementations (fallback, unchanged logic) ─────────────────────────

def _yf_candles(timeframe: str, count: int) -> dict | None:
    interval = _YF_INTERVAL_MAP.get(str(timeframe), "5m")
    period   = _YF_PERIOD_MAP.get(interval, "5d")

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


def _yf_current_price() -> float | None:
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
        log.error(f"get_current_price (yfinance) error: {exc}")
        return None
