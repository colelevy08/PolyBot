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
from concurrent.futures import ThreadPoolExecutor

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

        # Pre-instantiate ClobClient once (not per-order) to reuse its session.
        # Falls back to None if py_clob_client is not installed.
        self._clob_client = None
        # Dedicated single-threaded executor for ALL ClobClient calls.
        # ClobClient (requests-based) is NOT thread-safe; serialising via
        # max_workers=1 prevents concurrent calls from corrupting its state.
        self._clob_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clob")
        if cfg.poly_api_key:
            try:
                from py_clob_client.client import ClobClient
                self._clob_client = ClobClient(
                    host=cfg.clob_url,
                    chain_id=cfg.chain_id,
                    private_key=cfg.poly_private_key,
                    api_key=cfg.poly_api_key,
                    api_secret=cfg.poly_api_secret,
                    api_passphrase=cfg.poly_api_passphrase,
                )
                logger.info("ClobClient initialised.")
            except ImportError:
                logger.info("py_clob_client not installed — will use raw HTTP fallback.")

    def attach_session(self, session: aiohttp.ClientSession) -> None:
        """Called after the aiohttp session is ready."""
        self._session = session

    async def sync_bankroll(self) -> float:
        """
        Fetch live USDC balance from Polymarket and update the bankroll used
        for Kelly sizing. Called at startup and refreshed every stats cycle.

        Falls back to cfg.bankroll_usdc if credentials are missing or the
        call fails, so the bot can still run in test mode without credentials.
        """
        if self._clob_client is None:
            logger.debug("No CLOB client — keeping bankroll at $%.2f", self._bankroll)
            return self._bankroll

        try:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(self._clob_executor, self._clob_client.get_balance)

            # get_balance() returns a string like "150.23" or a dict;
            # normalise to float
            if isinstance(raw, dict):
                usdc = float(raw.get("balance", raw.get("usdc", 0)))
            else:
                usdc = float(raw)

            if usdc <= 0:
                logger.warning(
                    "Balance returned $%.2f — keeping previous bankroll $%.2f",
                    usdc, self._bankroll,
                )
                return self._bankroll

            # Cap at MAX_BANKROLL_USDC — never risk more than this at once
            capped = min(usdc, cfg.max_bankroll_usdc)
            old = self._bankroll
            self._bankroll = capped
            self._scanner._bankroll = capped  # keep edge scanner in sync
            logger.info(
                "Bankroll synced: $%.2f → $%.2f (balance=$%.2f, cap=$%.2f)",
                old, capped, usdc, cfg.max_bankroll_usdc,
            )
            return usdc

        except Exception as exc:
            logger.warning("Balance fetch failed (%s) — keeping $%.2f", exc, self._bankroll)
            return self._bankroll

    async def handle_signal(self, signal: WhaleSignal) -> None:
        """
        Entry point called by WhaleWatcher for every whale trade.
        This runs on the asyncio event loop — must not block.

        Only BUY signals are copied. SELL signals would require the complementary
        NO token ID (a different contract address not present in WhaleSignal), so
        they are skipped to avoid placing orders on the wrong token.
        """
        if signal.side != "BUY":
            logger.debug("Skipping SELL signal for %s (no NO token ID available)", signal.token_id)
            return

        token_id = signal.token_id

        # ── Fast dedup check ──────────────────────────────────────────────────
        now_ms = int(time.time() * 1000)
        last_ms = self._last_signal.get(token_id, 0)
        if now_ms - last_ms < _DEDUP_WINDOW_MS:
            logger.debug("Dedup skip: %s (last signal %dms ago)", token_id, now_ms - last_ms)
            return
        self._last_signal[token_id] = now_ms

        # ── Prune stale dedup entries (prevent unbounded dict growth) ─────────
        if len(self._last_signal) > 5_000:
            cutoff = now_ms - 60_000  # drop entries older than 60s
            self._last_signal = {k: v for k, v in self._last_signal.items() if v > cutoff}

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
        Uses pre-instantiated ClobClient if available, otherwise raw HTTP fallback.
        """
        order_side = signal.side  # already validated as "BUY" in handle_signal
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

        # ── Confirm fill (async, does not block main signal path) ─────────────
        asyncio.ensure_future(
            self._confirm_fill(order_id, signal.token_id)
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

        if self._clob_client is not None:
            # ── Pre-instantiated ClobClient path ─────────────────────────────
            # client.create_order / post_order are synchronous (uses requests).
            # Run them in a thread executor so we don't block the event loop.
            from py_clob_client.clob_types import OrderArgs, OrderType

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size_usdc / price,   # convert USDC to shares
                side=side,
            )
            loop = asyncio.get_running_loop()
            # Use dedicated single-threaded executor — ClobClient is not thread-safe
            signed_order = await loop.run_in_executor(
                self._clob_executor, self._clob_client.create_order, order_args
            )
            resp = await loop.run_in_executor(
                self._clob_executor,
                lambda: self._clob_client.post_order(signed_order, OrderType.GTC),
            )
            return str(resp.get("orderID", resp.get("id", "")))

        # ── Fallback: raw CLOB POST via aiohttp (async, no blocking) ─────────
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

    async def _confirm_fill(self, order_id: str, token_id: str) -> None:
        """
        Poll the CLOB for fill confirmation after an order is placed.
        If the order is not matched within cfg.order_fill_timeout_sec, cancel
        it and unblock the position slot so future signals can re-enter.

        Runs as a fire-and-forget task (does not block signal processing).
        """
        if not cfg.poly_api_key or self._clob_client is None or order_id.startswith("test_"):
            return  # test mode — nothing to confirm

        deadline = time.time() + cfg.order_fill_timeout_sec
        poll_interval = 2.0  # seconds between status checks

        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            try:
                loop = asyncio.get_running_loop()
                order_data = await loop.run_in_executor(
                    self._clob_executor,
                    lambda: self._clob_client.get_order(order_id),
                )
                # CLOB order status values are lowercase (confirmed from API docs):
                #   "live"      → resting on book, keep polling
                #   "matched"   → filled
                #   "delayed"   → sports market 3-second delay, keep polling
                #   "unmatched" → order placed but not matched
                status = str(order_data.get("status", "")).lower() if isinstance(order_data, dict) else ""

                if status == "matched":
                    logger.info("Fill confirmed  order_id=%s  token=%s", order_id, token_id)
                    return

                if status == "unmatched":
                    logger.warning(
                        "Order %s for token %s is unmatched — unblocking position slot.",
                        order_id, token_id,
                    )
                    self._order_manager.close_order(token_id, exit_price=None, resolved_yes=None)
                    return

            except Exception as exc:
                logger.debug("Fill poll failed for %s: %s", order_id, exc)

        # Timeout — cancel the order and free the slot
        logger.warning(
            "Order %s unfilled after %.0fs — cancelling and unblocking %s.",
            order_id, cfg.order_fill_timeout_sec, token_id,
        )
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._clob_executor,
                lambda: self._clob_client.cancel(order_id),
            )
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", order_id, exc)
        self._order_manager.close_order(token_id, exit_price=None, resolved_yes=None)
