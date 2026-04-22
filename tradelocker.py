"""
TradeLocker REST API client  —  RisenFX demo.

Account, instrument, and routes are hard-coded from config.  The client only
exposes the operations we actually use:

    • login / refresh
    • get_positions
    • place_market_order       (entry + emergency close)
    • close_position           (market SELL of the open long position)

Pending stop / limit orders are intentionally NOT used because RisenFX cannot
attach them to a market fill as real SL/TP — the bot monitors price itself and
closes via market SELL when SL or TP is breached.
"""

import logging
import requests

from config import (
    EMAIL, PASSWORD, SERVER, BASE_URL,
    ACCOUNT_ID, ACC_NUM, INSTRUMENT_ID, ROUTE_TRADE,
)

log = logging.getLogger(__name__)


class TradeLockerAPI:
    def __init__(self):
        self.access_token  = None
        self.refresh_token = None
        self.account_id    = ACCOUNT_ID
        self.account_no    = ACC_NUM
        self.instrument_id = INSTRUMENT_ID
        self.route_id      = ROUTE_TRADE
        self._session      = requests.Session()

    # ── Auth ──────────────────────────────────────────────────────────────────

    def login(self):
        resp = self._session.post(
            f"{BASE_URL}/auth/jwt/token",
            json={"email": EMAIL, "password": PASSWORD, "server": SERVER},
            timeout=15,
        )
        self._raise(resp, "login")
        body = resp.json()
        self.access_token  = body["accessToken"]
        self.refresh_token = body["refreshToken"]
        log.info("Login successful")

    def refresh(self):
        resp = self._session.post(
            f"{BASE_URL}/auth/jwt/refresh",
            json={"refreshToken": self.refresh_token},
            timeout=15,
        )
        self._raise(resp, "token refresh")
        body = resp.json()
        self.access_token = body["accessToken"]
        if "refreshToken" in body:
            self.refresh_token = body["refreshToken"]
        log.info("Access token refreshed")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type":  "application/json",
            "accNum":        str(self.account_no),
        }

    @staticmethod
    def _raise(resp, label="request"):
        if not resp.ok:
            log.error(f"{label} failed — HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()

    def _get(self, path, params=None):
        url = f"{BASE_URL}{path}"
        resp = self._session.get(url, headers=self._headers(), params=params, timeout=20)
        if resp.status_code == 401:
            log.info("401 on GET — refreshing token")
            self.refresh()
            resp = self._session.get(url, headers=self._headers(), params=params, timeout=20)
        self._raise(resp, f"GET {path}")
        return resp.json()

    def _post(self, path, body):
        url = f"{BASE_URL}{path}"
        resp = self._session.post(url, headers=self._headers(), json=body, timeout=20)
        if resp.status_code == 401:
            log.info("401 on POST — refreshing token")
            self.refresh()
            resp = self._session.post(url, headers=self._headers(), json=body, timeout=20)
        self._raise(resp, f"POST {path}")
        return resp.json()

    # ── Setup (no-op — IDs are hard-coded) ────────────────────────────────────

    def setup(self, symbol=None):
        log.info(
            f"Using hard-coded RisenFX config: account={self.account_id} "
            f"accNum={self.account_no} instrument={self.instrument_id} "
            f"route={self.route_id}"
        )

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self):
        try:
            data = self._get(f"/trade/accounts/{self.account_id}/positions")
            return data.get("d", {}).get("positions", [])
        except Exception as exc:
            log.error(f"get_positions failed: {exc}")
            return []

    def get_open_position(self):
        """Return the single open EURUSD.E position dict, or None."""
        for pos in self.get_positions():
            inst_id = (
                pos.get("tradableInstrumentId")
                or pos.get("instrumentId")
                or pos.get("id")
            )
            qty = float(pos.get("qty", 0) or pos.get("size", 0) or 0)
            if inst_id == self.instrument_id and qty > 0:
                return pos
        return None

    @staticmethod
    def position_qty(pos: dict) -> float:
        return float(pos.get("qty", 0) or pos.get("size", 0) or 0)

    @staticmethod
    def position_side(pos: dict) -> str:
        side = (pos.get("side") or "").lower()
        if side in ("buy", "long"):
            return "buy"
        if side in ("sell", "short"):
            return "sell"
        return "buy"   # RisenFX demo only trades long from this bot

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_market_order(self, side: str, qty: float) -> dict:
        body = {
            "qty":          qty,
            "instrumentId": self.instrument_id,
            "side":         side,          # "buy" | "sell"
            "type":         "market",
            "validity":     "GTC",
            "routeId":      self.route_id,
        }
        log.info(f"Market order → {body}")
        return self._post(f"/trade/accounts/{self.account_id}/orders", body)

    def close_position(self, pos: dict) -> dict:
        """
        Close the supplied open position by sending an opposite-side market
        order for its full quantity.  Because RisenFX demo does not support
        attached SL/TP, this is how the bot realises SL or TP hits.
        """
        qty  = self.position_qty(pos)
        side = self.position_side(pos)
        opposite = "sell" if side == "buy" else "buy"
        log.info(f"Closing position (orig side={side}, qty={qty}) via market {opposite}")
        return self.place_market_order(side=opposite, qty=qty)
