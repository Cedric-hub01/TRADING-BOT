"""
DataAgent — fetches XAU/USD OHLCV data from Twelve Data API.

Timeframes fetched:
    1m  → xauusd_1m.csv
    5m  → xauusd_5m.csv
    15m → xauusd_15m.csv

Cache policy: skip download when the file is < 1 hour old.
Rate limits: max 8 requests/min on the free tier; auto-retries after 60 s.
"""

import os
import time
import datetime
import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = "4d8062e470974365b29476f041a400b8"
BASE_URL  = "https://api.twelvedata.com/time_series"
SYMBOL    = "XAU/USD"
OUTPUTSIZE = 5000
CACHE_TTL  = 3600   # seconds — re-download if file older than this

# interval label → (api param, output filename)
TIMEFRAMES = {
    "1min":  "xauusd_1m.csv",
    "5min":  "xauusd_5m.csv",
    "15min": "xauusd_15m.csv",
}

# Free-tier hard limits
MAX_REQUESTS_PER_MIN = 8
_request_timestamps: list[float] = []


def _throttle() -> None:
    """Block until the rolling 1-minute window has room for one more request."""
    now = time.monotonic()
    # Drop timestamps older than 60 s
    cutoff = now - 60.0
    while _request_timestamps and _request_timestamps[0] < cutoff:
        _request_timestamps.pop(0)

    if len(_request_timestamps) >= MAX_REQUESTS_PER_MIN:
        wait = 60.0 - (now - _request_timestamps[0]) + 0.5
        print(f"  Rate limit reached — waiting {wait:.1f}s …")
        time.sleep(max(wait, 0))
        # Recurse to re-check after the wait
        _throttle()

    _request_timestamps.append(time.monotonic())


def _is_cache_fresh(filepath: str) -> bool:
    """Return True when the file exists and was modified less than CACHE_TTL seconds ago."""
    if not os.path.isfile(filepath):
        return False
    age = time.time() - os.path.getmtime(filepath)
    return age < CACHE_TTL


def _fetch_bars(interval: str) -> pd.DataFrame:
    """
    Call the Twelve Data time_series endpoint for SYMBOL at the given interval.
    Retries automatically on rate-limit errors (HTTP 429 or status 'error').
    Returns a DataFrame with columns: datetime, open, high, low, close, volume.
    """
    params = {
        "symbol":     SYMBOL,
        "interval":   interval,
        "outputsize": OUTPUTSIZE,
        "apikey":     API_KEY,
        "format":     "JSON",
    }

    while True:
        _throttle()
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  Network error: {exc}")
            print("  Retrying in 60 s …")
            time.sleep(60)
            continue

        data = resp.json()

        # Twelve Data signals quota exhaustion inside a 200 body
        if data.get("status") == "error":
            code = data.get("code", "")
            msg  = data.get("message", "unknown error")
            if code in (429, "429") or "limit" in msg.lower():
                print(f"  API quota hit ({msg}) — waiting 60 s …")
                time.sleep(60)
                continue
            raise RuntimeError(f"Twelve Data API error: {msg}")

        values = data.get("values")
        if not values:
            raise RuntimeError(f"No 'values' key in response for {interval}: {data}")

        df = pd.DataFrame(values)
        df.rename(columns={"datetime": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Twelve Data returns newest-first; reverse to chronological order
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df[["datetime", "open", "high", "low", "close", "volume"]]


def fetch_timeframe(interval: str, output_path: str) -> pd.DataFrame:
    """
    Fetch one timeframe, respecting the cache.
    Prints progress and saves to *output_path*.
    Returns the resulting DataFrame.
    """
    label = interval.replace("min", "m")

    if _is_cache_fresh(output_path):
        age_min = (time.time() - os.path.getmtime(output_path)) / 60
        print(f"  [{label}] Cache is {age_min:.1f} min old — using {output_path}")
        return pd.read_csv(output_path, parse_dates=["datetime"])

    print(f"  Fetching XAU/USD {label} data …")
    df = _fetch_bars(interval)
    df.to_csv(output_path, index=False)
    print(f"  Downloaded {len(df):,} bars")
    print(f"  Saved to {output_path}")
    return df


def run(output_dir: str = ".") -> dict[str, pd.DataFrame]:
    """
    Fetch all three timeframes.
    *output_dir* controls where the CSV files are written.
    Returns a dict mapping interval labels to DataFrames:
        {"1m": df_1m, "5m": df_5m, "15m": df_15m}
    """
    os.makedirs(output_dir, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for interval, filename in TIMEFRAMES.items():
        filepath = os.path.join(output_dir, filename)
        label    = interval.replace("min", "m")
        df       = fetch_timeframe(interval, filepath)
        results[label] = df
        print(f"  Done  [{label}]: {len(df):,} rows\n")

    return results


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DataAgent — XAU/USD multi-timeframe download")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    frames = run(output_dir=".")

    for label, df in frames.items():
        print(f"\n── First 5 rows of {label} ──────────────────────────────")
        print(df.head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("All timeframes fetched successfully.")
    print("=" * 60)
