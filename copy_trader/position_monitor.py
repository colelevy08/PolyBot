"""
PositionMonitor — periodically checks open positions for market resolution
and calls OrderManager.close_order() when a market settles.

Data source:
  Uses the CLOB REST API: GET /markets/{condition_id}
  Response includes a `tokens` array where each token has a `winner` boolean.
  Resolution is detected when `closed == True` and at least one token has
  `winner == True`.

  The Gamma API was NOT used here because:
    - Gamma has no `resolved` boolean field (use `closed` instead)
    - Gamma has no `resolutionValue` field
    - The `clob_token_ids` query param returned empty results in live testing
  The CLOB markets endpoint is the authoritative source for resolution state.

Design:
  - Polls every cfg.position_poll_interval_sec (default 5 min).
  - Shares the WhaleFetcher HTTP session — no extra TCP overhead.
  - Persists state after each cycle so restarts don't lose closed positions.
  - Exits cleanly on CancelledError.
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
    Polls open positions for market resolution via the CLOB /markets endpoint.
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
        pos_map = self._order_manager.open_positions_map()  # {token_id: condition_id}

        if not pos_map:
            return

        logger.debug("PositionMonitor checking %d open positions.", len(pos_map))

        tasks = [
            asyncio.create_task(self._check_position(token_id, condition_id))
            for token_id, condition_id in pos_map.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Persist updated state after each poll cycle
        self._order_manager.save()

    async def _check_position(self, token_id: str, condition_id: str) -> None:
        """
        Fetch CLOB market data for condition_id and close the position if resolved.

        CLOB /markets/{condition_id} response (confirmed fields):
          closed: bool          — True when the market has ended
          tokens: list of {
            token_id: str,
            outcome:  str       — "YES" or "NO"
            price:    str,
            winner:   bool      — True for the winning outcome token
          }
        """
        market = await self._fetcher.get_clob_market(condition_id)
        if market is None:
            return

        # Only close when the market is confirmed closed
        if not market.get("closed", False):
            return

        # Find the winning token among this market's tokens
        tokens: list[dict] = market.get("tokens", [])
        winning_token: dict | None = next(
            (t for t in tokens if t.get("winner") is True), None
        )

        if winning_token is None:
            # Market closed but no winner yet (rare race condition) — skip for now
            logger.debug(
                "Market %s is closed but no winner token found yet.", condition_id
            )
            return

        winning_outcome = str(winning_token.get("outcome", "")).upper()

        # Determine if our position (which tracks the YES token) won
        # token_id is the YES token ID for BUY positions.
        # If the winning token's id matches ours → YES won → resolved_yes=True
        winning_token_id = str(winning_token.get("token_id", ""))
        if winning_token_id:
            resolved_yes = winning_token_id == token_id
        else:
            # Fall back to outcome string
            resolved_yes = winning_outcome == "YES"

        logger.info(
            "Market resolved  condition=%s  token=%s  winner=%s  resolved_yes=%s",
            condition_id, token_id, winning_outcome, resolved_yes,
        )
        self._order_manager.close_order(
            token_id=token_id,
            exit_price=None,
            resolved_yes=resolved_yes,
        )
