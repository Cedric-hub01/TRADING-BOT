"""
Daily risk state manager.

Persists state to risk_state.json so the bot survives restarts within the same
trading day.  Automatically resets at midnight (new calendar date).
"""

import json
import logging
import os
from datetime import date

from config import (
    DAILY_STOP, DAILY_TARGET, MAX_CONSEC_LOSSES,
    MAX_LOSS_PER_TRADE, LOT_SIZE, PIP_SIZE, PIP_VALUE_STD,
)

log = logging.getLogger(__name__)

_STATE_FILE = "risk_state.json"


class RiskManager:
    def __init__(self):
        self.daily_pnl      = 0.0
        self.consec_losses  = 0
        self.trades_today   = 0
        self._today         = str(date.today())
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(_STATE_FILE):
            return
        try:
            with open(_STATE_FILE) as f:
                state = json.load(f)
            if state.get("date") == str(date.today()):
                self.daily_pnl     = state.get("daily_pnl", 0.0)
                self.consec_losses = state.get("consec_losses", 0)
                self.trades_today  = state.get("trades_today", 0)
                log.info(
                    f"Risk state loaded: pnl=${self.daily_pnl:.2f}  "
                    f"consec={self.consec_losses}  trades={self.trades_today}"
                )
            else:
                log.info("New trading day — risk state reset")
                self._save()
        except Exception as exc:
            log.warning(f"Could not load risk state: {exc}")

    def _save(self):
        try:
            with open(_STATE_FILE, "w") as f:
                json.dump(
                    {
                        "date":          str(date.today()),
                        "daily_pnl":     round(self.daily_pnl, 4),
                        "consec_losses": self.consec_losses,
                        "trades_today":  self.trades_today,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            log.warning(f"Could not save risk state: {exc}")

    # ── Trade gates ───────────────────────────────────────────────────────────

    def can_trade(self) -> bool:
        if self.daily_pnl <= DAILY_STOP:
            log.warning(
                f"DAILY STOP HIT — PnL ${self.daily_pnl:.2f} <= limit ${DAILY_STOP:.2f}.  "
                "No more trades today."
            )
            return False
        if self.daily_pnl >= DAILY_TARGET:
            log.info(
                f"DAILY TARGET REACHED — PnL ${self.daily_pnl:.2f} >= target ${DAILY_TARGET:.2f}.  "
                "No more trades today."
            )
            return False
        if self.consec_losses >= MAX_CONSEC_LOSSES:
            log.warning(
                f"MAX CONSECUTIVE LOSSES ({self.consec_losses}) reached.  "
                "No more trades today."
            )
            return False
        return True

    def check_trade_risk(self, sl_distance_price: float) -> bool:
        """
        Return True if the trade's SL distance translates to an acceptable USD risk.

        sl_distance_price : |entry - sl| in price (e.g. 0.0012 for 12 pips)
        """
        sl_pips    = sl_distance_price / PIP_SIZE
        sl_usd     = sl_pips * PIP_VALUE_STD * LOT_SIZE
        if sl_usd > MAX_LOSS_PER_TRADE:
            log.warning(
                f"Trade risk ${sl_usd:.2f} exceeds max ${MAX_LOSS_PER_TRADE:.2f} — skipping"
            )
            return False
        log.info(f"Trade risk check OK: ${sl_usd:.2f} (SL={sl_pips:.1f} pips)")
        return True

    # ── State updates ─────────────────────────────────────────────────────────

    def record_entry(self):
        self.trades_today += 1
        self._save()

    def record_close(self, pnl_usd: float):
        """Call when a position closes (SL or TP hit)."""
        self.daily_pnl = round(self.daily_pnl + pnl_usd, 4)
        if pnl_usd < 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0
        self._save()
        log.info(
            f"Trade closed: pnl=${pnl_usd:+.2f}  daily=${self.daily_pnl:+.2f}  "
            f"consec_losses={self.consec_losses}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def estimate_pnl(self, entry: float, exit_price: float) -> float:
        """USD PnL for a long position closed at exit_price."""
        pips = (exit_price - entry) / PIP_SIZE
        return round(pips * PIP_VALUE_STD * LOT_SIZE, 2)

    def status(self) -> str:
        return (
            f"Daily PnL: ${self.daily_pnl:+.2f}  |  "
            f"Consec Losses: {self.consec_losses}  |  "
            f"Trades Today: {self.trades_today}"
        )
