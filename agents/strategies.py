"""
agents/strategies.py — 9 XAU/USD strategy classes.

Interface consumed by backtest_engine.py and propfirm_simulator.py:
    strategy.name  (str)
    strategy.generate_signals(df: pd.DataFrame) -> pd.DataFrame
        Input  columns : datetime, open, high, low, close, volume
        Output columns : all input + signal (1=buy / -1=sell / 0=flat),
                                             sl_price, tp_price
"""

import numpy as np
import pandas as pd


# ── Shared indicator helpers ───────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _sl_tp(
    close: pd.Series, atr: pd.Series,
    longs: pd.Series, shorts: pd.Series,
    sl_m: float = 1.5, tp_m: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    sl = pd.Series(np.nan, index=close.index)
    tp = pd.Series(np.nan, index=close.index)
    sl[longs]  = close[longs]  - sl_m * atr[longs]
    tp[longs]  = close[longs]  + tp_m * atr[longs]
    sl[shorts] = close[shorts] + sl_m * atr[shorts]
    tp[shorts] = close[shorts] - tp_m * atr[shorts]
    return sl, tp


def _out(df: pd.DataFrame, sig: pd.Series, sl: pd.Series, tp: pd.Series) -> pd.DataFrame:
    r = df.copy()
    r["signal"]   = sig.fillna(0).astype(int)
    r["sl_price"] = sl
    r["tp_price"] = tp
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — EMA Triple Crossover  8 / 21 / 50
# ─────────────────────────────────────────────────────────────────────────────

class EMATripleCross:
    name = "EMA_Triple_8_21_50"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        e8   = _ema(c, 8)
        e21  = _ema(c, 21)
        e50  = _ema(c, 50)

        longs  = (e8 > e21) & (e8.shift(1) <= e21.shift(1)) & (e21 > e50)
        shorts = (e8 < e21) & (e8.shift(1) >= e21.shift(1)) & (e21 < e50)

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — MACD Histogram Zero-Cross with trend filter
# ─────────────────────────────────────────────────────────────────────────────

class MACDHistogram:
    name = "MACD_12_26_9"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        macd = _ema(c, 12) - _ema(c, 26)
        hist = macd - _ema(macd, 9)
        e50  = _ema(c, 50)

        longs  = (hist > 0) & (hist.shift(1) <= 0) & (c > e50)
        shorts = (hist < 0) & (hist.shift(1) >= 0) & (c < e50)

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 — RSI Oversold / Overbought  + EMA trend filter
# ─────────────────────────────────────────────────────────────────────────────

class RSIReversion:
    name = "RSI_Reversion_14"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        rsi  = _rsi(c, 14)
        e50  = _ema(c, 50)

        longs  = (rsi < 35) & (rsi.shift(1) >= 35) & (c > e50.shift(5))
        shorts = (rsi > 65) & (rsi.shift(1) <= 65) & (c < e50.shift(5))

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts, sl_m=1.2, tp_m=2.4)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4 — Bollinger Band  2σ  Breakout
# ─────────────────────────────────────────────────────────────────────────────

class BollingerBreakout:
    name = "Bollinger_Breakout_20"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        mid  = _sma(c, 20)
        std  = c.rolling(20).std()
        e50  = _ema(c, 50)

        upper  = mid + 2.0 * std
        lower  = mid - 2.0 * std
        longs  = (c > upper) & (c.shift(1) <= upper.shift(1)) & (mid > e50)
        shorts = (c < lower) & (c.shift(1) >= lower.shift(1)) & (mid < e50)

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts, sl_m=1.0, tp_m=2.0)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5 — Donchian Channel  20-bar  Breakout
# ─────────────────────────────────────────────────────────────────────────────

class DonchianBreakout:
    name = "Donchian_Breakout_20"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        dhi  = df["high"].rolling(20).max().shift(1)
        dlo  = df["low"].rolling(20).min().shift(1)

        longs  = (c > dhi) & (c.shift(1) <= dhi.shift(1))
        shorts = (c < dlo) & (c.shift(1) >= dlo.shift(1))

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 6 — ATR Channel  Breakout  (EMA-34 midline)
# ─────────────────────────────────────────────────────────────────────────────

class ATRChannelBreakout:
    name = "ATR_Channel_Breakout_34"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df, 20)
        mid  = _ema(c, 34)

        upper  = mid + 2.0 * atr
        lower  = mid - 2.0 * atr
        longs  = (c > upper) & (c.shift(1) <= upper.shift(1))
        shorts = (c < lower) & (c.shift(1) >= lower.shift(1))

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts, sl_m=1.0, tp_m=2.5)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 7 — Rolling VWAP  Mean-Reversion  (240-bar window ≈ 4 h)
# ─────────────────────────────────────────────────────────────────────────────

class VWAPReversion:
    name = "VWAP_Reversion_240"

    _N = 240

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        v    = df["volume"].replace(0, np.nan)
        typ  = (df["high"] + df["low"] + c) / 3
        vwap = (typ * v).rolling(self._N).sum() / v.rolling(self._N).sum()

        longs  = (c < vwap * 0.9985) & (c.shift(1) >= vwap.shift(1) * 0.9985)
        shorts = (c > vwap * 1.0015) & (c.shift(1) <= vwap.shift(1) * 1.0015)

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts, sl_m=1.2, tp_m=2.4)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 8 — Rate-of-Change  Momentum  (20-bar)
# ─────────────────────────────────────────────────────────────────────────────

class ROCMomentum:
    name = "ROC_Momentum_20"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        roc  = (c / c.shift(20) - 1) * 100
        e50  = _ema(c, 50)

        longs  = (roc >  0.10) & (roc.shift(1) <=  0.10) & (c > e50)
        shorts = (roc < -0.10) & (roc.shift(1) >= -0.10) & (c < e50)

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts, sl_m=1.5, tp_m=2.5)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 9 — Stochastic  %K/%D  Cross  + EMA trend filter
# ─────────────────────────────────────────────────────────────────────────────

class StochasticTrend:
    name = "Stochastic_EMA_14_3"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        c    = df["close"]
        atr  = _atr(df)
        e21  = _ema(c, 21)
        e50  = _ema(c, 50)

        lo14 = df["low"].rolling(14).min()
        hi14 = df["high"].rolling(14).max()
        k    = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
        d    = _sma(k, 3)

        longs  = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 50) & (e21 > e50)
        shorts = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 50) & (e21 < e50)

        sig = pd.Series(0, index=df.index)
        sig[longs]  =  1
        sig[shorts] = -1
        sl, tp = _sl_tp(c, atr, longs, shorts)
        return _out(df, sig, sl, tp)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGIES — imported by backtest_engine.py and propfirm_simulator.py
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = [
    EMATripleCross(),
    MACDHistogram(),
    RSIReversion(),
    BollingerBreakout(),
    DonchianBreakout(),
    ATRChannelBreakout(),
    VWAPReversion(),
    ROCMomentum(),
    StochasticTrend(),
]
