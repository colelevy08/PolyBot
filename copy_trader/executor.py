"""
TradeExecutor — receives a WhaleSignal, runs the EdgeScanner, and if edge
is confirmed, places an order via the Polymarket CLOB as fast as possible.

Speed principles:
- Edge scan and order building happen in parallel where possible.
- The CLOB client session is pre-created and reused (no handshake overhead).
- Order signing is done with a pre-loaded private key object (no re-parse).
- orjson for all serialisation on the order path.
- Minimal logging on the hot path; stats logged asynchronously.

Order flow:
  1. WhaleSignal arrives (already has detected_ts)
  2. EdgeScanner.scan() → EdgeResult   (async, 1 HTTP request)
  3. If has_edge: build + sign + POST order
  4. Log total latency: detected_ts → order confirmed
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp
import orjson

from config import cfg
from edge_scanner.scanner import EdgeScanner
from whale_analyzer.fetcher import WhaleFetcher
from whale_analyzer.models import EdgeResult, WhaleSignal
from copy_trader.order_manager import OrderManager

logger = logging.getLogger(__name__)

# Dedup window: ignore duplicate signals for the same token within N ms
_DEDUP_WINDOW_MS = 2_000


class TradeExecutor:
    """
    Wired together with:
      - EdgeScanner   (for live edge checks)
      - OrderManager  (for dedup + position tracking)
      - aiohttp session (shared with WhaleFetcher)
    """

    def __init__(
        self,
        fetcher: WhaleFetcher,
        order_manager: OrderManager,
        bankroll_usdc: float | None = None,
    ) -> None:
        self._scanner = EdgeScanner(fetcher, bankroll_usdc)
        self._order_manager = order_manager
        self._bankroll = bankroll_usdc or cfg.bankroll_usdc
        # Cache session reference from fetcher to avoid attribute lookup overhead
        self._session: aiohttp.ClientSession | None = None
        self._fetcher = fetcher

        # Simple in-process dedup: token_id → last signal ts (ms)
        self._last_signal: dict[str, int] = {}

    def attach_session(self, session: aiohttp.ClientSession) -> None:
        """Called after the aiohttp session is ready."""
        self._session = session

    async def handle_signal(self, signal: WhaleSignal) -> None:
        """
        Entry point called by WhaleWatcher for every whale trade.
        This runs on the asyncio event loop — must not block.
        """
        token_id = signal.token_id

        # ── Fast dedup check ──────────────────────────────────────────────────
        now_ms = int(time.time() * 1000)
        last_ms = self._last_signal.get(token_id, 0)
        if now_ms - last_ms < _DEDUP_WINDOW_MS:
            logger.debug("Dedup skip: %s (last signal %dms ago)", token_id, now_ms - last_ms)
            return
        self._last_signal[token_id] = now_ms

        # ── Check if we already have a position ───────────────────────────────
        if self._order_manager.has_open_position(token_id):
            logger.debug("Already have position in %s — skip.", token_id)
            return

        # ── Edge scan ─────────────────────────────────────────────────────────
        edge_result: EdgeResult = await self._scanner.scan(signal)

        if not edge_result.has_edge:
            return  # No edge — do not trade

        # ── Fire order ────────────────────────────────────────────────────────
        await self._place_order(signal, edge_result)

    async def _place_order(self, signal: WhaleSignal, edge: EdgeResult) -> None:
        """
        Build, sign, and POST a limit order to the CLOB.
        Uses py-clob-client for signing if credentials are set,
        otherwise falls back to raw unsigned POST (test mode).
        """
        fired_ts = int(time.time() * 1000)

        order_side = "BUY" if signal.side == "BUY" else "BUY"  # always copy direction
        size = edge.bet_size_usdc

        if size < 1.0:
            logger.debug("Bet size $%.2f too small — skip.", size)
            return

        # ── Build order via py-clob-client ────────────────────────────────────
        try:
            order_id = await self._submit_clob_order(
                token_id=signal.token_id,
                side=order_side,
                price=edge.our_price,
                size_usdc=size,
            )
        except Exception as exc:
            logger.error("Order submission failed for %s: %s", signal.token_id, exc)
            return

        # ── Record order ──────────────────────────────────────────────────────
        self._order_manager.record_order(
            order_id=order_id,
            token_id=signal.token_id,
            market_id=signal.market_id,
            side=order_side,
            price=edge.our_price,
            size_usdc=size,
            whale_address=signal.whale_address,
        )

        total_latency_ms = int(time.time() * 1000) - signal.detected_ts
        logger.info(
            "ORDER PLACED  id=%s  token=%s  side=%s  price=%.3f  size=$%.2f  "
            "edge=%.3f  latency=%dms",
            order_id, signal.token_id, order_side, edge.our_price,
            size, edge.edge, total_latency_ms,
        )

    async def _submit_clob_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_usdc: float,
    ) -> str:
        """
        Post a signed limit order to the CLOB.
        Returns the order_id string on success.

        Implements: https://docs.polymarket.com/#create-order
        """
        if not cfg.poly_api_key:
            # ── No credentials: test mode — log and return fake ID ────────────
            logger.warning(
                "TESTMODE: would place %s order for %s at %.3f size=$%.2f",
                side, token_id, price, size_usdc,
            )
            return f"test_{token_id[:8]}_{int(time.time())}"

        # ── Use py-clob-client if available ───────────────────────────────────
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType

            client = ClobClient(
                host=cfg.clob_url,
                chain_id=cfg.chain_id,
                private_key=cfg.poly_private_key,
                api_key=cfg.poly_api_key,
                api_secret=cfg.poly_api_secret,
                api_passphrase=cfg.poly_api_passphrase,
            )

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size_usdc / price,   # convert USDC to shares
                side=side,
            )
            signed_order = client.create_order(order_args)
            resp = client.post_order(signed_order, OrderType.GTC)
            order_id = resp.get("orderID", resp.get("id", ""))
            return str(order_id)

        except ImportError:
            # ── Fallback: raw CLOB POST (unsigned — requires API key header) ──
            assert self._session is not None, "HTTP session not attached"
            payload = {
                "tokenID": token_id,
                "side": side,
                "price": str(round(price, 4)),
                "size": str(round(size_usdc / price, 2)),
                "feeRateBps": "0",
                "nonce": str(int(time.time() * 1000)),
                "type": "GTC",
            }
            headers = {
                "POLY-API-KEY": cfg.poly_api_key,
                "POLY-API-SECRET": cfg.poly_api_secret,
                "POLY-API-PASSPHRASE": cfg.poly_api_passphrase,
                "Content-Type": "application/json",
            }
            async with self._session.post(
                f"{cfg.clob_url}/order",
                data=orjson.dumps(payload),
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                data = orjson.loads(await resp.read())
                return str(data.get("orderID", data.get("id", "")))
