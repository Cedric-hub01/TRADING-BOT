"""
Strategy library for the backtester.

Each strategy is a factory function `build(params) -> strategy_obj` where the
strategy object has:

    name   : str                — human label used in the leaderboard
    params : dict               — parameters used (copied into the result row)
    signals(df) -> DataFrame    — two columns of booleans:
                                    'long_entry'  : open a long on next bar
                                    'short_entry' : open a short on next bar

The backtester consumes only the boolean entry columns; it handles exits
itself (ATR-based SL/TP).  A strategy never manages exits.

Parameter grids at the bottom of this file define every combination the
backtester enumerates.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ── Indicator helpers ─────────────────────────────────────────────────────────

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    return wilder(tr, n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    diff = s.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    roll_up = wilder(up, n)
    roll_dn = wilder(dn, n)
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bbands(s: pd.Series, n=20, k=2.0):
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid - k * std, mid, mid + k * std


def stoch(df: pd.DataFrame, k=14, d=3, smooth=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    raw_k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    kline = raw_k.rolling(smooth).mean()
    dline = kline.rolling(d).mean()
    return kline, dline


def adx(df: pd.DataFrame, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm  = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - df["close"].shift(1)).abs(),
         (df["low"]  - df["close"].shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_w    = wilder(tr, n)
    plus_di  = 100 * wilder(plus_dm, n)  / atr_w.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, n) / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n), plus_di, minus_di


def supertrend(df: pd.DataFrame, n=10, mult=3.0):
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    direction = pd.Series(1, index=df.index, dtype=int)
    st = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        prev = direction.iat[i - 1]
        c = df["close"].iat[i]
        if prev == 1:
            lower.iat[i] = max(lower.iat[i], lower.iat[i - 1])
            direction.iat[i] = -1 if c < lower.iat[i - 1] else 1
        else:
            upper.iat[i] = min(upper.iat[i], upper.iat[i - 1])
            direction.iat[i] = 1 if c > upper.iat[i - 1] else -1
        st.iat[i] = lower.iat[i] if direction.iat[i] == 1 else upper.iat[i]
    return st, direction


def ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou=52):
    conv = (df["high"].rolling(tenkan).max() + df["low"].rolling(tenkan).min()) / 2
    base = (df["high"].rolling(kijun).max()  + df["low"].rolling(kijun).min())  / 2
    span_a = ((conv + base) / 2).shift(kijun)
    span_b = ((df["high"].rolling(senkou).max() + df["low"].rolling(senkou).min()) / 2).shift(kijun)
    return conv, base, span_a, span_b


# ── Strategy container ────────────────────────────────────────────────────────

@dataclass
class Strategy:
    name: str
    params: dict
    signals: Callable[[pd.DataFrame], pd.DataFrame]
    direction: str = "both"   # "long" | "short" | "both"


# ── Strategy builders ─────────────────────────────────────────────────────────

def _frame(long_e: pd.Series, short_e: pd.Series, index) -> pd.DataFrame:
    return pd.DataFrame(
        {"long_entry": long_e.reindex(index).fillna(False).astype(bool),
         "short_entry": short_e.reindex(index).fillna(False).astype(bool)},
        index=index,
    )


def ema_cross(fast: int, med: int, slow: int, use_trend: bool):
    def sig(df):
        f = ema(df["close"], fast)
        m = ema(df["close"], med)
        s = ema(df["close"], slow)
        cross_up   = (f.shift(1) <= m.shift(1)) & (f > m)
        cross_down = (f.shift(1) >= m.shift(1)) & (f < m)
        if use_trend:
            cross_up   &= (f > s) & (m > s)
            cross_down &= (f < s) & (m < s)
        return _frame(cross_up, cross_down, df.index)
    tag = f"EMA {fast}/{med}/{slow}" + (" +trend" if use_trend else "")
    return Strategy(
        name=tag,
        params={"type": "ema_cross", "fast": fast, "med": med, "slow": slow, "trend_filter": use_trend},
        signals=sig,
    )


def macd_cross(fast: int, slow: int, signal: int):
    def sig(df):
        line, sigl, _ = macd(df["close"], fast, slow, signal)
        cu = (line.shift(1) <= sigl.shift(1)) & (line > sigl)
        cd = (line.shift(1) >= sigl.shift(1)) & (line < sigl)
        return _frame(cu, cd, df.index)
    return Strategy(
        name=f"MACD {fast}/{slow}/{signal}",
        params={"type": "macd", "fast": fast, "slow": slow, "signal": signal},
        signals=sig,
    )


def supertrend_flip(period: int, mult: float):
    def sig(df):
        _, d = supertrend(df, period, mult)
        flip_up   = (d.shift(1) == -1) & (d == 1)
        flip_down = (d.shift(1) ==  1) & (d == -1)
        return _frame(flip_up, flip_down, df.index)
    return Strategy(
        name=f"Supertrend {period}×{mult}",
        params={"type": "supertrend", "period": period, "mult": mult},
        signals=sig,
    )


def ichimoku_tk(tenkan: int, kijun: int, senkou: int):
    def sig(df):
        conv, base, span_a, span_b = ichimoku(df, tenkan, kijun, senkou)
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
        cu = (conv.shift(1) <= base.shift(1)) & (conv > base) & (df["close"] > cloud_top)
        cd = (conv.shift(1) >= base.shift(1)) & (conv < base) & (df["close"] < cloud_bot)
        return _frame(cu, cd, df.index)
    return Strategy(
        name=f"Ichimoku {tenkan}/{kijun}/{senkou}",
        params={"type": "ichimoku", "tenkan": tenkan, "kijun": kijun, "senkou": senkou},
        signals=sig,
    )


def rsi_reversion(period: int, os: int, ob: int):
    def sig(df):
        r = rsi(df["close"], period)
        cu = (r.shift(1) <= os) & (r > os)     # exit oversold → buy
        cd = (r.shift(1) >= ob) & (r < ob)     # exit overbought → sell
        return _frame(cu, cd, df.index)
    return Strategy(
        name=f"RSI {period} {os}/{ob} reversion",
        params={"type": "rsi_reversion", "period": period, "os": os, "ob": ob},
        signals=sig,
    )


def bb_reversion(period: int, k: float):
    def sig(df):
        lo, _, up = bbands(df["close"], period, k)
        cu = (df["close"].shift(1) < lo.shift(1)) & (df["close"] > lo)   # bounce up
        cd = (df["close"].shift(1) > up.shift(1)) & (df["close"] < up)   # fade top
        return _frame(cu, cd, df.index)
    return Strategy(
        name=f"BB {period}×{k} reversion",
        params={"type": "bb_reversion", "period": period, "k": k},
        signals=sig,
    )


def bb_breakout(period: int, k: float):
    def sig(df):
        lo, _, up = bbands(df["close"], period, k)
        cu = (df["close"].shift(1) <= up.shift(1)) & (df["close"] > up)  # upside breakout
        cd = (df["close"].shift(1) >= lo.shift(1)) & (df["close"] < lo)  # downside breakout
        return _frame(cu, cd, df.index)
    return Strategy(
        name=f"BB {period}×{k} breakout",
        params={"type": "bb_breakout", "period": period, "k": k},
        signals=sig,
    )


def stoch_reversion(k_period: int, d_period: int, smooth: int, os: int, ob: int):
    def sig(df):
        k, d = stoch(df, k_period, d_period, smooth)
        cu = (k.shift(1) <= os) & (k > os) & (k > d)
        cd = (k.shift(1) >= ob) & (k < ob) & (k < d)
        return _frame(cu, cd, df.index)
    return Strategy(
        name=f"Stoch {k_period}/{d_period}/{smooth} {os}/{ob}",
        params={"type": "stoch_reversion", "k": k_period, "d": d_period, "smooth": smooth, "os": os, "ob": ob},
        signals=sig,
    )


def ema_cross_adx(fast: int, slow: int, adx_period: int, adx_min: float):
    def sig(df):
        f = ema(df["close"], fast)
        s = ema(df["close"], slow)
        a, _, _ = adx(df, adx_period)
        cu = (f.shift(1) <= s.shift(1)) & (f > s) & (a > adx_min)
        cd = (f.shift(1) >= s.shift(1)) & (f < s) & (a > adx_min)
        return _frame(cu, cd, df.index)
    return Strategy(
        name=f"EMA {fast}/{slow} + ADX>{adx_min}",
        params={"type": "ema_adx", "fast": fast, "slow": slow, "adx_period": adx_period, "adx_min": adx_min},
        signals=sig,
    )


def session_breakout(session: str, lookback_minutes: int, duration_minutes: int, bar_minutes: int):
    """
    'london' session : range 07:00–08:00 UTC, break in the next `duration_minutes`
    'ny'     session : range 12:30–13:30 UTC, break in the next `duration_minutes`
    """
    if session == "london":
        range_start, range_end = (7, 0), (8, 0)
    else:
        range_start, range_end = (12, 30), (13, 30)

    lookback_bars = max(1, lookback_minutes // bar_minutes)
    duration_bars = max(1, duration_minutes // bar_minutes)

    def sig(df):
        idx = df.index
        minutes = idx.hour * 60 + idx.minute
        rs = range_start[0] * 60 + range_start[1]
        re = range_end[0]   * 60 + range_end[1]
        in_range = (minutes >= rs) & (minutes < re)

        # Rolling high/low of the last `lookback_bars` bars that were inside the range
        range_high = df["high"].where(in_range).rolling(lookback_bars).max().ffill()
        range_low  = df["low"].where(in_range).rolling(lookback_bars).min().ffill()

        # Limit entries to the `duration_bars` bars immediately after the range ends
        bars_since_range = pd.Series(0, index=idx, dtype=int)
        counter = 0
        in_window = []
        for flag in in_range:
            if flag:
                counter = 0
            else:
                counter += 1
            in_window.append(0 < counter <= duration_bars)
        in_window = pd.Series(in_window, index=idx)

        cu = in_window & (df["close"] > range_high) & (df["close"].shift(1) <= range_high.shift(1))
        cd = in_window & (df["close"] < range_low)  & (df["close"].shift(1) >= range_low.shift(1))
        return _frame(cu, cd, idx)

    return Strategy(
        name=f"{session.upper()} breakout L{lookback_minutes}m D{duration_minutes}m",
        params={"type": "session_breakout", "session": session,
                "lookback_minutes": lookback_minutes, "duration_minutes": duration_minutes,
                "bar_minutes": bar_minutes},
        signals=sig,
    )


# ── Parameter grid builder ────────────────────────────────────────────────────

def all_strategies(bar_minutes: int):
    """
    Enumerate every strategy × parameter combination we want to evaluate.

    Tuning grids are intentionally kept coarse so the full sweep on 60 d of
    5-minute bars finishes in a few minutes on a laptop.
    """
    out: list[Strategy] = []

    # 1. EMA crossovers (with / without EMA-slow trend filter)
    ema_fasts = [5, 8, 10, 13, 20]
    ema_meds  = [21, 34, 50]
    ema_slows = [50, 100, 200]
    for f, m, s in itertools.product(ema_fasts, ema_meds, ema_slows):
        if f < m < s:
            for tf in (False, True):
                out.append(ema_cross(f, m, s, tf))

    # 2. MACD
    for f, s, sig in [(12, 26, 9), (8, 21, 5), (5, 35, 5), (10, 30, 9)]:
        out.append(macd_cross(f, s, sig))

    # 3. Supertrend
    for n, mult in itertools.product([7, 10, 14], [1.5, 2.0, 3.0]):
        out.append(supertrend_flip(n, mult))

    # 4. Ichimoku
    for t, k, s in [(9, 26, 52), (7, 22, 44), (20, 60, 120)]:
        out.append(ichimoku_tk(t, k, s))

    # 5. RSI mean-reversion
    for p, (os, ob) in itertools.product([7, 14, 21], [(20, 80), (25, 75), (30, 70)]):
        out.append(rsi_reversion(p, os, ob))

    # 6. Bollinger — both reversion and breakout
    for p, k in itertools.product([20, 50], [2.0, 2.5, 3.0]):
        out.append(bb_reversion(p, k))
        out.append(bb_breakout(p, k))

    # 7. Stochastic
    for (kp, dp, sm), (os, ob) in itertools.product(
        [(14, 3, 3), (21, 5, 5)], [(20, 80), (25, 75)]
    ):
        out.append(stoch_reversion(kp, dp, sm, os, ob))

    # 8. EMA + ADX filter
    for f, s, ap, am in itertools.product([8, 13, 20], [50, 100], [14], [20.0, 25.0]):
        out.append(ema_cross_adx(f, s, ap, am))

    # 9. Session breakouts
    for sess in ("london", "ny"):
        for look, dur in [(30, 120), (60, 120), (60, 180)]:
            out.append(session_breakout(sess, look, dur, bar_minutes))

    return out
