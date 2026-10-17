"""Finnhub WebSocket client.

``FinnhubStreamer`` runs in a background thread and never touches Streamlit.
It pushes plain ``(kind, payload)`` tuples onto a ``queue.Queue``, and the app
drains that queue on each rerun:

    ("tick",   (datetime_utc, price))
    ("status", "Connected to OANDA:XAU_USD")
    ("error",  "WS error: ...")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from queue import Queue
from typing import Optional

from websocket import WebSocketApp

TICK = "tick"
STATUS = "status"
ERROR = "error"

WS_URL = "wss://ws.finnhub.io"


class FinnhubError(Exception):
    """Finnhub sent an error message over the socket."""


def parse_trade_message(message: str) -> list[tuple[datetime, float]]:
    """Turn one raw WebSocket message into ``(utc_timestamp, price)`` ticks.

    Messages without trades (for example pings) give an empty list.
    Raises ``FinnhubError`` for an error message from Finnhub and ``ValueError``
    for malformed JSON or a trade missing its price or timestamp.
    """
    payload = json.loads(message)  # json.JSONDecodeError is a ValueError
    if not isinstance(payload, dict):
        return []
    if payload.get("type") == "error":
        raise FinnhubError(payload.get("msg", "unknown error"))

    ticks = []
    for trade in payload.get("data") or []:
        try:
            price = float(trade["p"])
            timestamp_ms = int(trade["t"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed trade {trade!r}") from exc
        ticks.append((datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc), price))
    return ticks


class FinnhubStreamer:
    def __init__(self, token: str, symbol: str, out_queue: Queue):
        self.token = token
        self.symbol = symbol
        self.out_queue = out_queue
        self.ws: Optional[WebSocketApp] = None

    def _on_open(self, ws):
        ws.send(json.dumps({"type": "subscribe", "symbol": self.symbol}))
        self.out_queue.put((STATUS, f"Connected to {self.symbol}"))

    def _on_message(self, ws, message: str):
        try:
            for tick in parse_trade_message(message):
                self.out_queue.put((TICK, tick))
        except FinnhubError as exc:
            self.out_queue.put((ERROR, f"Finnhub: {exc}"))
        except ValueError as exc:
            self.out_queue.put((ERROR, f"Could not parse message: {exc}"))

    def _on_error(self, ws, error):
        self.out_queue.put((ERROR, f"WebSocket error: {error}"))

    def _on_close(self, ws, status_code, message):
        detail = f" {message}" if message else ""
        self.out_queue.put((STATUS, f"Closed ({status_code}){detail}"))

    def run(self):
        """Blocking: connect and read until closed. Run this in a thread."""
        self.ws = WebSocketApp(
            f"{WS_URL}?token={self.token}",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever(ping_interval=15, ping_timeout=10)

    def stop(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
