"""
WhaleWatcher — subscribes to Polymarket CLOB WebSocket channels for each
tracked whale address and emits WhaleSignal objects to a callback as fast
as possible.

Speed principles:
- Single persistent WebSocket connection (not one per whale).
- orjson for all parsing — raw bytes in, parsed dict out.
- Callback is called directly in the receive loop with zero queuing overhead
  for the critical path (the executor is expected to be fast enough).
- Reconnect with exponential back-off; subscriptions re-sent immediately.
- No logging on the hot path (inside the tight receive loop).

WebSocket protocol (Polymarket CLOB):
  Connect: wss://ws-subscriptions-clob.polymarket.com/ws/
  Subscribe user channel:
    {"type": "subscribe", "channel": "user", "userAddress": "<address>"}
  Trade event shape (subset used here):
    {
      "type": "trade",
      "userAddress": "<address>",
      "asset_id": "<token_id>",
      "market": "<market_id>",
      "side": "BUY" | "SELL",
      "price": "0.72",
      "amount": "50.00",
      "timestamp": 1700000000000
    }
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

import orjson
import websockets

from config import cfg
from whale_analyzer.models import WhaleSignal

logger = logging.getLogger(__name__)

# Type alias: async callback that receives a WhaleSignal
SignalCallback = Callable[[WhaleSignal], Coroutine[Any, Any, None]]


def _parse_signal(raw: dict) -> WhaleSignal | None:
    """
    Parse a raw WebSocket event dict into a WhaleSignal.
    Returns None if the event is not a trade or is malformed.
    Hot path — keep allocations minimal.
    """
    if raw.get("type") != "trade":
        return None
    try:
        return WhaleSignal(
            whale_address=raw["userAddress"],
            market_id=raw.get("market", raw.get("conditionId", "")),
            token_id=raw.get("asset_id", raw.get("tokenId", "")),
            side=(raw.get("side") or "BUY").upper(),
            price=float(raw["price"]),
            size_usdc=float(raw.get("amount", raw.get("size", 0))),
            detected_ts=int(time.time() * 1000),  # our clock, not server's
            raw_event=raw,
        )
    except (KeyError, ValueError, TypeError):
        return None


class WhaleWatcher:
    """
    Subscribes to a list of whale addresses on Polymarket's CLOB WebSocket
    and fires `on_signal` for every trade event.

    Usage:
        watcher = WhaleWatcher(whale_addresses, on_signal=executor.handle_signal)
        await watcher.run()
    """

    def __init__(
        self,
        whale_addresses: list[str],
        on_signal: SignalCallback,
    ) -> None:
        self._addresses = whale_addresses
        self._on_signal = on_signal
        self._running = False

    async def run(self) -> None:
        """Main loop — connects and reconnects indefinitely."""
        self._running = True
        delay = cfg.ws_reconnect_delay

        while self._running:
            try:
                await self._connect_and_listen()
                delay = cfg.ws_reconnect_delay  # reset on clean exit
            except Exception as exc:
                logger.warning(
                    "WebSocket disconnected (%s). Reconnecting in %.1fs...",
                    exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)  # exponential back-off, cap 30s

    async def stop(self) -> None:
        self._running = False

    async def _connect_and_listen(self) -> None:
        logger.info(
            "Connecting to Polymarket WebSocket: %s  (%d whales)",
            cfg.ws_url, len(self._addresses),
        )
        async with websockets.connect(
            cfg.ws_url,
            ping_interval=cfg.ws_ping_interval,
            ping_timeout=cfg.ws_ping_timeout,
            # Large read buffer to handle bursts without stalling
            max_size=2**20,
            # Disable compression — latency over bandwidth
            compression=None,
        ) as ws:
            # Subscribe to all whale addresses immediately
            await self._subscribe_all(ws)
            logger.info("Subscribed. Listening for whale trades...")

            async for raw_bytes in ws:
                # ── CRITICAL PATH: parse and fire as fast as possible ─────────
                try:
                    event = orjson.loads(raw_bytes)
                except Exception:
                    continue  # malformed JSON — skip

                # Handle both single events and batched arrays
                if isinstance(event, list):
                    for item in event:
                        await self._dispatch(item)
                elif isinstance(event, dict):
                    await self._dispatch(event)

    async def _subscribe_all(self, ws: Any) -> None:
        """Send subscription messages for all tracked whale addresses."""
        for address in self._addresses:
            msg = orjson.dumps({
                "type": "subscribe",
                "channel": "user",
                "userAddress": address,
            })
            await ws.send(msg)
        logger.debug("Sent %d subscription messages.", len(self._addresses))

    async def _dispatch(self, event: dict) -> None:
        signal = _parse_signal(event)
        if signal is not None:
            # Fire the callback — executor decides whether to trade
            await self._on_signal(signal)
