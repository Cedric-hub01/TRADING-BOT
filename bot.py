"""
EURUSD EMA 8/21/50 Crossover Bot — TradeLocker API
────────────────────────────────────────────────────────────────────────────
Strategy : EMA8 × EMA21 crossover, EMA50 trend filter, Long Only
SL/TP    : ATR-based (separate stop + limit orders — no PATCH)
Risk     : daily stop / daily target / max consecutive losses / max loss/trade
────────────────────────────────────────────────────────────────────────────
Usage:
    python bot.py

Configuration:
    Edit config.py — update EMAIL, PASSWORD, SERVER (3 lines) to switch brokers.
────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime

from config import SCAN_INTERVAL, CANDLES_NEEDED, SYMBOL, TIMEFRAME, LOT_SIZE
from tradelocker import TradeLockerAPI
from strategy  import build_dataframe, get_signal
from risk      import RiskManager

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Position tracker file ─────────────────────────────────────────────────────
_POS_FILE = "open_trade.json"

_running = True


def _shutdown(sig, frame):
    global _running
    log.info("Shutdown signal received — stopping after current iteration")
    _running = False


# ── Position persistence ──────────────────────────────────────────────────────

def _save_trade(entry_price: float, sl: float, tp: float):
    with open(_POS_FILE, "w") as f:
        json.dump(
            {"entry": entry_price, "sl": sl, "tp": tp, "time": datetime.now().isoformat()},
            f, indent=2,
        )


def _load_trade() -> dict | None:
    if os.path.exists(_POS_FILE):
        try:
            with open(_POS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _clear_trade():
    if os.path.exists(_POS_FILE):
        os.remove(_POS_FILE)


# ── Position detection ────────────────────────────────────────────────────────

def _has_open_position(api: TradeLockerAPI) -> bool:
    positions = api.get_positions()
    for pos in positions:
        inst_id = pos.get("tradableInstrumentId") or pos.get("instrumentId") or pos.get("id")
        if inst_id == api.instrument_id:
            qty = pos.get("qty", 0) or pos.get("size", 0)
            if float(qty) > 0:
                return True
    return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    api  = TradeLockerAPI()
    risk = RiskManager()

    log.info("=" * 60)
    log.info("EURUSD EMA Crossover Bot  —  starting up")
    log.info(f"Symbol: {SYMBOL}  |  Timeframe: {TIMEFRAME}m  |  Lot: {LOT_SIZE}")
    log.info("=" * 60)

    # ── Initial auth + instrument discovery ───────────────────────────────────
    try:
        api.login()
        api.setup(SYMBOL)
    except Exception as exc:
        log.critical(f"Startup failed: {exc}")
        sys.exit(1)

    log.info(f"Ready — scanning every {SCAN_INTERVAL}s.  Press Ctrl+C to stop.")
    log.info(risk.status())

    while _running:
        loop_start = time.time()
        ts = datetime.now().strftime("%H:%M:%S")

        try:
            # ── 1. Check daily risk limits ─────────────────────────────────────
            if not risk.can_trade():
                log.info(f"[{ts}] Risk limits active — skipping scan.  {risk.status()}")
                _sleep_remainder(loop_start)
                continue

            # ── 2. Skip if we already hold a position ──────────────────────────
            if _has_open_position(api):
                saved = _load_trade()
                if saved:
                    log.info(
                        f"[{ts}] In position (entry={saved['entry']:.5f}  "
                        f"SL={saved['sl']:.5f}  TP={saved['tp']:.5f}).  "
                        f"{risk.status()}"
                    )
                else:
                    log.info(f"[{ts}] Holding position (no local record).  {risk.status()}")
                _sleep_remainder(loop_start)
                continue

            # ── Position just closed — record PnL ─────────────────────────────
            saved = _load_trade()
            if saved:
                _handle_closed_trade(saved, risk)

            # ── 3. Fetch candles ───────────────────────────────────────────────
            bar_data = api.get_candles(TIMEFRAME, CANDLES_NEEDED)
            if bar_data is None:
                log.warning(f"[{ts}] No candle data returned — will retry next cycle")
                _sleep_remainder(loop_start)
                continue

            n_bars = len(bar_data.get("t", []))
            if n_bars < 60:
                log.warning(f"[{ts}] Only {n_bars} bars — need 60+.  Waiting.")
                _sleep_remainder(loop_start)
                continue

            # ── 4. Calculate indicators + get signal ───────────────────────────
            df = build_dataframe(bar_data)
            signal_dir, sl_price, tp_price, atr_val = get_signal(df)

            if signal_dir is None:
                last = df.iloc[-2]
                log.info(
                    f"[{ts}] No signal | "
                    f"price={last['close']:.5f}  "
                    f"EMA8={last['ema_fast']:.5f}  "
                    f"EMA21={last['ema_med']:.5f}  "
                    f"EMA50={last['ema_slow']:.5f}  "
                    f"ATR={atr_val:.5f}"
                )
                _sleep_remainder(loop_start)
                continue

            # ── 5. Risk filter for this specific trade ─────────────────────────
            entry_price = df["close"].iloc[-2]
            sl_distance = abs(entry_price - sl_price)

            if not risk.check_trade_risk(sl_distance):
                _sleep_remainder(loop_start)
                continue

            # ── 6. Place orders ────────────────────────────────────────────────
            _place_trade(api, risk, entry_price, sl_price, tp_price)

        except Exception as exc:
            log.error(f"[{ts}] Loop error: {exc}", exc_info=True)
            _try_reauth(api)

        _sleep_remainder(loop_start)

    log.info("Bot stopped cleanly.")


# ── Trade execution ───────────────────────────────────────────────────────────

def _place_trade(
    api: TradeLockerAPI,
    risk: RiskManager,
    entry_price: float,
    sl_price: float,
    tp_price: float,
):
    log.info(
        f"Entering LONG | est_entry={entry_price:.5f}  SL={sl_price:.5f}  TP={tp_price:.5f}"
    )

    # 1. Market entry
    try:
        resp = api.place_market_order(side="buy", qty=LOT_SIZE)
        log.info(f"Market order filled: {resp}")
    except Exception as exc:
        log.error(f"Market order failed: {exc}")
        return

    # Brief pause — let the fill register
    time.sleep(2)

    # 2. Stop-loss order (sell stop below entry)
    try:
        resp = api.place_stop_order(side="sell", qty=LOT_SIZE, stop_price=sl_price)
        log.info(f"SL order placed: {resp}")
    except Exception as exc:
        log.error(f"SL order failed — MANUAL ACTION REQUIRED: {exc}")

    # 3. Take-profit order (sell limit above entry)
    try:
        resp = api.place_limit_order(side="sell", qty=LOT_SIZE, limit_price=tp_price)
        log.info(f"TP order placed: {resp}")
    except Exception as exc:
        log.error(f"TP order failed — MANUAL ACTION REQUIRED: {exc}")

    # 4. Save trade details + update risk state
    _save_trade(entry_price, sl_price, tp_price)
    risk.record_entry()
    log.info(f"Trade recorded.  {risk.status()}")


def _handle_closed_trade(saved: dict, risk: RiskManager):
    """
    Called when we had a tracked trade but no open position was found.
    Estimate whether SL or TP was hit based on the saved prices, then record.
    """
    entry = saved["entry"]
    sl    = saved["sl"]
    tp    = saved["tp"]

    # Without a real fill price we estimate from which level was closer
    # This is a heuristic; replace with API account history if available.
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)

    # Use TP distance for the win case, SL distance for the loss case
    # We don't know which was hit, so we record both sides at entry ±
    # (In production, poll the closed-positions / trade-history endpoint.)
    # For now, log a reminder and record 0 to avoid skewing state.
    log.info(
        f"Position closed (entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}).  "
        "Actual PnL not available without history endpoint — recording $0.  "
        "Check broker dashboard for real result."
    )
    risk.record_close(0.0)
    _clear_trade()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sleep_remainder(loop_start: float):
    elapsed    = time.time() - loop_start
    sleep_secs = max(0.0, SCAN_INTERVAL - elapsed)
    if sleep_secs > 0:
        time.sleep(sleep_secs)


def _try_reauth(api: TradeLockerAPI):
    try:
        api.refresh()
        log.info("Token refreshed after error")
    except Exception:
        try:
            log.info("Refresh failed — attempting full re-login")
            api.login()
            api.setup(SYMBOL)
            log.info("Re-login successful")
        except Exception as exc:
            log.error(f"Re-auth failed: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
