"""
EURUSD trading bot  —  RisenFX / TradeLocker.

Strategy is pluggable via config.py (STRATEGY_TYPE / STRATEGY_PARAMS / DIRECTION);
every 60 s the bot either:

    • monitors the open position — closes via an opposite-side market order when
      the live price hits the ATR-based SL or TP (real execution, not pending
      orders), OR
    • scans for a new entry using the live strategy signal.

Why monitor-and-close?
RisenFX demo does not let us attach SL/TP to a market fill and PATCH on
positions returns 404.  Separate stop / limit orders are only PENDING — they
can fail to execute.  A live market SELL/BUY guarantees the exit fills.

Usage:
    python bot.py
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime

from config import (
    SCAN_INTERVAL, CANDLES_NEEDED, SYMBOL, TIMEFRAME, LOT_SIZE,
    STRATEGY_TYPE, DIRECTION,
)
from tradelocker import TradeLockerAPI
from strategy    import build_dataframe, get_signal
from risk        import RiskManager
import market_data

# ── Logging ───────────────────────────────────────────────────────────────────
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

# ── Persistent trade record (entry / SL / TP / side for the monitor) ──────────
_POS_FILE = "open_trade.json"

_running = True


def _shutdown(sig, frame):
    global _running
    log.info("Shutdown signal received — stopping after current iteration")
    _running = False


# ── Trade persistence ─────────────────────────────────────────────────────────

def _save_trade(side: str, entry: float, sl: float, tp: float):
    with open(_POS_FILE, "w") as f:
        json.dump(
            {
                "side":  side,                        # "buy" | "sell"
                "entry": entry,
                "sl":    sl,
                "tp":    tp,
                "time":  datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


def _load_trade() -> dict | None:
    if os.path.exists(_POS_FILE):
        try:
            with open(_POS_FILE) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _clear_trade():
    if os.path.exists(_POS_FILE):
        os.remove(_POS_FILE)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    api  = TradeLockerAPI()
    risk = RiskManager()

    log.info("=" * 60)
    log.info(f"EURUSD bot — strategy={STRATEGY_TYPE}  direction={DIRECTION}")
    log.info(f"Symbol: {SYMBOL}  |  Timeframe: {TIMEFRAME}m  |  Lot: {LOT_SIZE}")
    log.info("=" * 60)

    try:
        api.login()
        api.setup(SYMBOL)
    except Exception as exc:
        log.critical(f"Startup failed: {exc}")
        sys.exit(1)

    log.info(f"Ready — scan / SL-TP monitor every {SCAN_INTERVAL}s.  Ctrl+C to stop.")
    log.info(risk.status())

    while _running:
        loop_start = time.time()
        ts = datetime.now().strftime("%H:%M:%S")

        try:
            position = api.get_open_position()

            # ── Branch A: we already hold a position → SL/TP monitor ──────
            if position is not None:
                _monitor_position(api, risk, position, ts)

            # ── Branch B: flat → look for a new entry ─────────────────────
            else:
                saved = _load_trade()
                if saved:
                    _handle_closed_trade(saved, risk)

                if not risk.can_trade():
                    log.info(f"[{ts}] Risk limits active — skipping scan.  {risk.status()}")
                else:
                    _scan_for_entry(api, risk, ts)

        except Exception as exc:
            log.error(f"[{ts}] Loop error: {exc}", exc_info=True)
            _try_reauth(api)

        _sleep_remainder(loop_start)

    log.info("Bot stopped cleanly.")


# ── SL / TP monitor (side-aware) ──────────────────────────────────────────────

def _monitor_position(api: TradeLockerAPI, risk: RiskManager, position: dict, ts: str):
    saved = _load_trade()
    if saved is None:
        log.warning(
            f"[{ts}] Open position exists but no local SL/TP record — "
            "cannot monitor.  Close manually or restart with a trade record."
        )
        return

    side  = saved.get("side", "buy")
    entry = saved["entry"]
    sl    = saved["sl"]
    tp    = saved["tp"]

    price = market_data.get_current_price()
    if price is None:
        log.warning(f"[{ts}] No live price — skipping SL/TP check this cycle")
        return

    log.info(
        f"[{ts}] MONITOR {side.upper()} price={price:.5f}  entry={entry:.5f}  "
        f"SL={sl:.5f}  TP={tp:.5f}  {risk.status()}"
    )

    # Long: SL is below entry, TP above.  Short: inverted.
    if side == "buy":
        hit = "SL" if price <= sl else "TP" if price >= tp else None
    else:
        hit = "SL" if price >= sl else "TP" if price <= tp else None

    if hit is None:
        return

    opposite = "sell" if side == "buy" else "buy"
    log.info(f"[{ts}] {hit} HIT at {price:.5f} — closing via market {opposite.upper()}")
    try:
        resp = api.close_position(position)
        log.info(f"Close order response: {resp}")
    except Exception as exc:
        log.error(f"Close failed — will retry next cycle: {exc}")
        return

    pnl = risk.estimate_pnl(entry, price, side=side)
    risk.record_close(pnl)
    _clear_trade()


# ── Entry scan ────────────────────────────────────────────────────────────────

def _scan_for_entry(api: TradeLockerAPI, risk: RiskManager, ts: str):
    bar_data = market_data.get_candles(TIMEFRAME, CANDLES_NEEDED)
    if bar_data is None:
        log.warning(f"[{ts}] No candle data — retrying next cycle")
        return

    n_bars = len(bar_data.get("t", []))
    if n_bars < 60:
        log.warning(f"[{ts}] Only {n_bars} bars — need 60+.  Waiting.")
        return

    df = build_dataframe(bar_data)
    signal_dir, sl_price, tp_price, atr_val = get_signal(df)

    if signal_dir is None:
        last = df.iloc[-2]
        log.info(
            f"[{ts}] No signal [{STRATEGY_TYPE}] | "
            f"price={last['close']:.5f}  ATR={atr_val:.5f}"
        )
        return

    entry_price = float(df["close"].iloc[-2])
    sl_distance = abs(entry_price - sl_price)

    if not risk.check_trade_risk(sl_distance):
        return

    _place_trade(api, risk, signal_dir, entry_price, sl_price, tp_price)


# ── Order placement ───────────────────────────────────────────────────────────

def _place_trade(
    api: TradeLockerAPI,
    risk: RiskManager,
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
):
    log.info(
        f"Entering {side.upper()} | est_entry={entry_price:.5f}  "
        f"SL={sl_price:.5f}  TP={tp_price:.5f}"
    )

    try:
        resp = api.place_market_order(side=side, qty=LOT_SIZE)
        log.info(f"Market order filled: {resp}")
    except Exception as exc:
        log.error(f"Market order failed — no position opened: {exc}")
        return

    # Small pause so the fill registers before the next positions call
    time.sleep(2)

    _save_trade(side, entry_price, sl_price, tp_price)
    risk.record_entry()
    log.info(f"Trade recorded — SL/TP will be enforced by the monitor loop.  {risk.status()}")


# ── Closed-trade bookkeeping (defensive) ──────────────────────────────────────

def _handle_closed_trade(saved: dict, risk: RiskManager):
    """
    Reached when the local trade record exists but the broker reports no open
    position.  In the normal flow _monitor_position() already recorded the PnL
    and cleared the file, so this handler only runs if the position closed
    outside the monitor (manual close, server-side liquidation, etc).
    """
    log.info(
        f"Position closed out-of-band (side={saved.get('side','?')}  "
        f"entry={saved['entry']:.5f}  SL={saved['sl']:.5f}  TP={saved['tp']:.5f}).  "
        "PnL not captured — recording $0.  Check broker dashboard for real result."
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
