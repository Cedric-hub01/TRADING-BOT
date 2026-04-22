"""
TradeLocker REST API client.

Handles authentication, token refresh, account/instrument discovery,
candle history, and order placement (market + stop + limit).
"""

import time
import logging
import requests

from config import EMAIL, PASSWORD, SERVER, BASE_URL, SYMBOL

log = logging.getLogger(__name__)

# How many seconds before a 401 triggers a token refresh retry
_RETRY_ON_401 = True


class TradeLockerAPI:
    def __init__(self):
        self.access_token  = None
        self.refresh_token = None
        self.account_id    = None   # internal numeric ID
        self.account_no    = None   # accNum header value
        self.instrument_id = None
        self.route_id      = None
        self._session      = requests.Session()

    # ── Auth ─────────────────────────────────────────────────────────────────

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
        h = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type":  "application/json",
        }
        if self.account_no is not None:
            h["accNum"] = str(self.account_no)
        return h

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

    # ── Account / Instrument discovery ───────────────────────────────────────

    def get_accounts(self):
        data = self._get("/trade/accounts")
        return data["d"]["accounts"]

    def get_instruments(self):
        data = self._get(
            f"/trade/accounts/{self.account_id}/instruments",
            params={"locale": "en"},
        )
        return data["d"]["instruments"]

    def setup(self, symbol=None):
        """Resolve account_id, account_no, instrument_id, route_id."""
        symbol = symbol or SYMBOL

        accounts = self.get_accounts()
        if not accounts:
            raise RuntimeError("No trading accounts found on this server")

        # Prefer live/demo account; fallback to first
        account = accounts[0]
        self.account_id = account["id"]
        # accNum can live under several field names depending on broker
        self.account_no = (
            account.get("accNum")
            or account.get("accountNo")
            or account.get("number")
            or str(self.account_id)
        )
        log.info(f"Account: id={self.account_id}  accNum={self.account_no}")

        instruments = self.get_instruments()
        needle = symbol.replace("/", "").replace(" ", "").upper()

        for inst in instruments:
            name = (
                inst.get("name", "")
                or inst.get("description", "")
                or inst.get("symbol", "")
            )
            if needle in name.replace("/", "").replace(" ", "").upper():
                self.instrument_id = (
                    inst.get("tradableInstrumentId")
                    or inst.get("id")
                )
                routes = inst.get("routes", [])
                self.route_id = routes[0]["id"] if routes else 1
                log.info(
                    f"Instrument: {name}  id={self.instrument_id}  routeId={self.route_id}"
                )
                return

        available = [
            inst.get("name") or inst.get("description") or "?"
            for inst in instruments[:30]
        ]
        raise RuntimeError(
            f"{symbol} not found in instrument list.\n"
            f"Available (first 30): {available}"
        )

    # ── Market data ───────────────────────────────────────────────────────────

    def get_candles(self, resolution, count=120):
        """
        Fetch OHLCV bars.  Returns the raw barData dict:
            {"o": [...], "h": [...], "l": [...], "c": [...], "v": [...], "t": [...]}
        or None on failure.
        """
        now   = int(time.time())
        # resolution is in minutes for numeric strings; for "1D" assume 1440
        try:
            res_mins = int(resolution)
        except ValueError:
            res_mins = 1440

        # Request 2× the needed window to guarantee enough bars
        start = now - (count * res_mins * 60 * 2)

        params = {
            "instrumentId":   self.instrument_id,
            "resolution":     resolution,
            "startTimestamp": start,
            "endTimestamp":   now,
        }

        try:
            data = self._get("/trade/history", params=params)
            bar_data = data.get("d", {}).get("barData", {})
            n = len(bar_data.get("t", []))
            log.debug(f"Fetched {n} candles (resolution={resolution})")
            return bar_data if n > 0 else None
        except Exception as exc:
            log.error(f"get_candles failed: {exc}")
            return None

    # ── Positions / Orders ────────────────────────────────────────────────────

    def get_positions(self):
        try:
            data = self._get(f"/trade/accounts/{self.account_id}/positions")
            return data.get("d", {}).get("positions", [])
        except Exception as exc:
            log.error(f"get_positions failed: {exc}")
            return []

    def get_orders(self):
        try:
            data = self._get(f"/trade/accounts/{self.account_id}/orders")
            return data.get("d", {}).get("orders", [])
        except Exception as exc:
            log.error(f"get_orders failed: {exc}")
            return []

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

    def place_stop_order(self, side: str, qty: float, stop_price: float) -> dict:
        body = {
            "qty":          qty,
            "instrumentId": self.instrument_id,
            "side":         side,
            "type":         "stop",
            "stopPrice":    round(stop_price, 5),
            "validity":     "GTC",
            "routeId":      self.route_id,
        }
        log.info(f"Stop order → {body}")
        return self._post(f"/trade/accounts/{self.account_id}/orders", body)

    def place_limit_order(self, side: str, qty: float, limit_price: float) -> dict:
        body = {
            "qty":          qty,
            "instrumentId": self.instrument_id,
            "side":         side,
            "type":         "limit",
            "price":        round(limit_price, 5),
            "validity":     "GTC",
            "routeId":      self.route_id,
        }
        log.info(f"Limit order → {body}")
        return self._post(f"/trade/accounts/{self.account_id}/orders", body)
