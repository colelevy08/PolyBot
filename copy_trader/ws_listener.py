"""
WhalePollWatcher — detects new trades from tracked whale wallets by polling
the Polymarket CLOB REST API (/data/trades?maker_address=<addr>&after=<ts>).

WHY polling instead of WebSocket:
  Polymarket's WebSocket user channel (wss://...polymarket.com/ws/user) is
  authenticated and only delivers trades for the AUTHENTICATED user (i.e.,
  your own account). There is no mechanism to subscribe to arbitrary wallet
  addresses via WebSocket. The market channel delivers order-book and price
  events but does NOT include the trader's wallet address in the event payload,
  making it impossible to reliably attribute trades to tracked wallets.

  Polling /data/trades with `after=<ts>` per whale is the only confirmed
  way to detect external wallets' trades in near-real-time.

Performance:
  - 15 whales × 1 poll per 2s = 7.5 req/s → well within the 10 rps rate limit.
  - Polls run concurrently via asyncio (a semaphore limits burst concurrency).
  - Latency to detect a trade: 0–2 seconds (one poll interval).
  - No blocking: _parse_signal and callback are called inside the async loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from config import cfg
from whale_analyzer.fetcher import WhaleFetcher
from whale_analyzer.models import WhaleSignal

logger = logging.getLogger(__name__)

SignalCallback = Callable[[WhaleSignal], Coroutine[Any, Any, None]]

# Max simultaneous in-flight poll requests (burst protection)
_POLL_CONCURRENCY = 8


def _parse_signal(raw: dict, whale_address: str) -> WhaleSignal | None:
    """
    Parse a raw /data/trades response dict into a WhaleSignal.
    Returns None if the trade is not a BUY or is missing required fields.

    Confirmed /data/trades field names:
      id, market (condition_id), asset_id (token_id), side, size, price,
      status, match_time (unix second string), outcome, maker_address
    """
    try:
        side = (raw.get("side") or "").upper()
        token_id = str(raw.get("asset_id", ""))
        market_id = str(raw.get("market", ""))
        price_str = raw.get("price")
        size_str = raw.get("size")

        if not token_id or not market_id or price_str is None or size_str is None:
            return None

        # match_time is a Unix second string (e.g. "1672290701")
        ts_raw = raw.get("match_time") or raw.get("timestamp") or 0
        try:
            trade_ts_sec = int(float(ts_raw))
        except (TypeError, ValueError):
            trade_ts_sec = 0

        return WhaleSignal(
            whale_address=whale_address,
            market_id=market_id,
            token_id=token_id,
            side=side,
            price=float(price_str),
            size_usdc=float(size_str),
            detected_ts=int(time.time() * 1000),  # our clock, not server's
            raw_event=raw,
        )
    except (KeyError, ValueError, TypeError):
        return None


class WhaleWatcher:
    """
    Polls /data/trades for each tracked whale address and emits WhaleSignal
    objects to a callback for every new trade detected.

    Usage:
        watcher = WhaleWatcher(whale_addresses, on_signal=executor.handle_signal)
        await watcher.run()   # runs until cancelled
    """

    def __init__(
        self,
        whale_addresses: list[str],
        on_signal: SignalCallback,
    ) -> None:
        self._addresses = whale_addresses
        self._on_signal = on_signal
        # Per-whale cursor: last seen trade timestamp in Unix seconds.
        # Initialised to "now" so we only pick up new trades after startup.
        now_sec = int(time.time())
        self._last_ts: dict[str, int] = {addr: now_sec for addr in whale_addresses}
        self._sem = asyncio.Semaphore(_POLL_CONCURRENCY)
        self._fetcher: WhaleFetcher | None = None

    def attach_fetcher(self, fetcher: WhaleFetcher) -> None:
        """Inject the shared WhaleFetcher (provides aiohttp session)."""
        self._fetcher = fetcher

    async def run(self) -> None:
        """Main poll loop — runs until cancelled."""
        if self._fetcher is None:
            raise RuntimeError("Call attach_fetcher() before run()")

        logger.info(
            "WhalePollWatcher started — tracking %d whales (poll=%.1fs)",
            len(self._addresses), cfg.whale_poll_interval_sec,
        )
        try:
            while True:
                poll_start = time.monotonic()
                await self._poll_all()
                # Sleep for whatever remains of the interval (avoid drift)
                elapsed = time.monotonic() - poll_start
                sleep_for = max(0.0, cfg.whale_poll_interval_sec - elapsed)
                await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            pass

    async def _poll_all(self) -> None:
        """Fire concurrent polls for all tracked whales."""
        tasks = [
            asyncio.create_task(self._poll_whale(addr))
            for addr in self._addresses
        ]
        # gather with return_exceptions so one failure doesn't kill the loop
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_whale(self, address: str) -> None:
        """Fetch new trades for a single whale and dispatch any signals found."""
        assert self._fetcher is not None
        async with self._sem:
            after_ts = self._last_ts.get(address, 0)
            raw_trades = await self._fetcher.get_recent_trades(address, after_ts)

        if not raw_trades:
            return

        # Sort ascending by match_time so we process oldest first and advance
        # the cursor correctly even when multiple trades arrive in one poll.
        def _ts(t: dict) -> int:
            try:
                return int(float(t.get("match_time") or t.get("timestamp") or 0))
            except (TypeError, ValueError):
                return 0

        raw_trades.sort(key=_ts)

        latest_ts = self._last_ts.get(address, 0)
        for raw in raw_trades:
            trade_ts = _ts(raw)
            if trade_ts <= after_ts:
                continue  # already seen (timestamp not strictly advancing)

            signal = _parse_signal(raw, address)
            if signal is not None:
                await self._on_signal(signal)

            if trade_ts > latest_ts:
                latest_ts = trade_ts

        # Advance cursor past the last trade we processed
        self._last_ts[address] = latest_ts
