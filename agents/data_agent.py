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
import sys
import time
import datetime
import requests
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = "4d8062e470974365b29476f041a400b8"
BASE_URL  = "https://api.twelvedata.com/time_series"
SYMBOL    = "XAU/USD"
OUTPUTSIZE = 5000
CACHE_TTL  = 3600   # seconds — re-download if file older than this

# How many paginated 1m requests to make (10 × 5000 ≈ 50,000 bars / ~35 days)
NUM_1M_REQUESTS = 10

# interval label → output filename  (1m handled separately via pagination)
TIMEFRAMES = {
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


def _fetch_bars(interval: str, end_date: str | None = None) -> pd.DataFrame:
    """
    Single request to Twelve Data time_series.
    *end_date* (optional) is a 'YYYY-MM-DD HH:MM:SS' string that caps the
    newest bar returned — used for backwards pagination.
    Returns a DataFrame with columns: datetime, open, high, low, close, volume.
    """
    params = {
        "symbol":     SYMBOL,
        "interval":   interval,
        "outputsize": OUTPUTSIZE,
        "apikey":     API_KEY,
        "format":     "JSON",
    }
    if end_date:
        params["end_date"] = end_date

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
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # XAU/USD on Twelve Data free tier does not include volume — default to 0
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0

        # Twelve Data returns newest-first; reverse to chronological order
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df[["datetime", "open", "high", "low", "close", "volume"]]


def _fetch_1m_paginated(output_path: str) -> pd.DataFrame:
    """
    Collect ~50,000 bars of 1m XAU/USD data via NUM_1M_REQUESTS backwards-
    paginated requests.  Each request asks for OUTPUTSIZE bars ending just
    before the earliest bar received in the previous batch.
    Deduplicates, sorts, and saves to *output_path*.
    """
    print(f"  Fetching XAU/USD 1m data "
          f"({NUM_1M_REQUESTS} requests × {OUTPUTSIZE:,} bars target) …")

    all_batches: list[pd.DataFrame] = []
    end_date: str | None = None   # first request: no cap → returns latest bars

    for req_num in range(1, NUM_1M_REQUESTS + 1):
        suffix = f" end_date={end_date}" if end_date else " (latest)"
        print(f"    Request {req_num}/{NUM_1M_REQUESTS}{suffix}")
        try:
            batch = _fetch_bars("1min", end_date=end_date)
        except RuntimeError as exc:
            print(f"    Request {req_num} failed: {exc} — stopping early")
            break

        if batch.empty:
            print("    Empty response — stopping early")
            break

        all_batches.append(batch)

        # Step backwards: cap next request 1 minute before this batch's earliest bar
        earliest = batch["datetime"].min()
        end_date = (earliest - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")

    if not all_batches:
        raise RuntimeError("All 1m paginated requests failed — no data collected")

    df = pd.concat(all_batches, ignore_index=True)
    df.drop_duplicates(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"  Combined: {len(df):,} bars  "
          f"({df['datetime'].iloc[0]}  →  {df['datetime'].iloc[-1]})")
    print(f"  Saved to {output_path}")
    return df


def fetch_timeframe(interval: str, output_path: str) -> pd.DataFrame:
    """
    Fetch one timeframe, respecting the cache.
    1m uses paginated multi-request fetch; 5m/15m use a single request.
    Prints progress and saves to *output_path*.
    Returns the resulting DataFrame.
    """
    label = interval.replace("min", "m")

    if _is_cache_fresh(output_path):
        age_min = (time.time() - os.path.getmtime(output_path)) / 60
        print(f"  [{label}] Cache is {age_min:.1f} min old — using {output_path}")
        return pd.read_csv(output_path, parse_dates=["datetime"])

    if interval == "1min":
        return _fetch_1m_paginated(output_path)

    print(f"  Fetching XAU/USD {label} data …")
    df = _fetch_bars(interval)
    df.to_csv(output_path, index=False)
    print(f"  Downloaded {len(df):,} bars")
    print(f"  Saved to {output_path}")
    return df


def run(output_dir: str = ".") -> dict[str, pd.DataFrame]:
    """
    Fetch all three timeframes.
    1m: 10 paginated requests → ~50,000 bars saved to xauusd_1m.csv.
    5m/15m: single request → xauusd_5m.csv / xauusd_15m.csv.
    Returns a dict mapping interval labels to DataFrames:
        {"1m": df_1m, "5m": df_5m, "15m": df_15m}
    """
    os.makedirs(output_dir, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    # 1m — paginated deep fetch
    path_1m = os.path.join(output_dir, "xauusd_1m.csv")
    df_1m   = fetch_timeframe("1min", path_1m)
    results["1m"] = df_1m
    print(f"  Done  [1m]: {len(df_1m):,} rows\n")

    # 5m and 15m — single request each
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
