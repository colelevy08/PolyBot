"""
PositionMonitor — periodically checks open positions against Polymarket's
Gamma API for market resolution and calls OrderManager.close_order() when
a market settles.

Design:
- Polls every cfg.position_poll_interval_sec (default 5 min).
- Uses the shared aiohttp session from WhaleFetcher (no extra TCP overhead).
- Runs as a fire-and-forget asyncio task alongside WhaleWatcher.
- Exits cleanly on CancelledError.

Without this, positions held open forever lock up capital and inflate the
open position count — making the dedup check block re-entry indefinitely.
"""
from __future__ import annotations

import asyncio
import logging

from config import cfg
from copy_trader.order_manager import OrderManager
from whale_analyzer.fetcher import WhaleFetcher

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Polls open positions for market resolution.

    Args:
        fetcher:       Shared WhaleFetcher (provides HTTP session).
        order_manager: The live OrderManager whose positions to monitor.
    """

    def __init__(self, fetcher: WhaleFetcher, order_manager: OrderManager) -> None:
        self._fetcher = fetcher
        self._order_manager = order_manager

    async def run(self) -> None:
        """Main loop — runs until cancelled."""
        logger.info(
            "PositionMonitor started (poll interval=%.0fs).",
            cfg.position_poll_interval_sec,
        )
        try:
            while True:
                await asyncio.sleep(cfg.position_poll_interval_sec)
                await self._check_all_positions()
        except asyncio.CancelledError:
            pass

    async def _check_all_positions(self) -> None:
        """Check every open position for resolution."""
        summary = self._order_manager.summary()
        open_tokens: list[str] = summary.get("open_tokens", [])

        if not open_tokens:
            return

        logger.debug("PositionMonitor checking %d open positions.", len(open_tokens))

        tasks = [
            asyncio.create_task(self._check_position(token_id))
            for token_id in open_tokens
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Persist updated state after each poll cycle
        self._order_manager.save()

    async def _check_position(self, token_id: str) -> None:
        """Fetch market status for a single token and close if resolved."""
        try:
            data = await self._fetcher._get_json(
                f"{cfg.gamma_url}/markets",
                params={"clob_token_ids": token_id},
            )
        except Exception as exc:
            logger.debug("Market status fetch failed for %s: %s", token_id, exc)
            return

        # Gamma returns a list of market objects
        markets: list[dict] = data if isinstance(data, list) else data.get("markets", [])

        if not markets:
            return

        market = markets[0]
        resolved = market.get("resolved", False)
        resolution_value = market.get("resolutionValue") or market.get("resolution")

        if not resolved:
            return  # still active

        # Determine YES/NO resolution
        resolved_yes: bool | None = None
        if resolution_value is not None:
            rv = str(resolution_value).lower()
            if rv in ("yes", "1", "true", "win"):
                resolved_yes = True
            elif rv in ("no", "0", "false", "loss"):
                resolved_yes = False

        logger.info(
            "Market resolved  token=%s  resolved_yes=%s",
            token_id, resolved_yes,
        )
        self._order_manager.close_order(
            token_id=token_id,
            exit_price=None,
            resolved_yes=resolved_yes,
        )
