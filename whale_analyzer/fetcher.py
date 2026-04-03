"""
WhaleFetcher — pulls wallet lists and trade history from Polymarket APIs.

Speed priorities:
- Single persistent aiohttp session with a large connection pool
- Concurrent fetching of individual wallet histories (semaphore-controlled)
- orjson for all parsing
- Raw bytes → dict: no intermediate string allocation on hot path
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp
import orjson

from config import cfg
from whale_analyzer.models import TradeRecord, WalletProfile

logger = logging.getLogger(__name__)

# Max concurrent wallet-history requests (tune to API rate limits)
_CONCURRENCY = 25


class _TokenBucket:
    """
    Simple async token-bucket rate limiter.
    Limits to `rate` requests per second across all callers.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate          # tokens added per second
        self._tokens = rate        # start full
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            # Release lock before sleeping so other callers can check/refill
            await asyncio.sleep(wait)


def _parse_trade(raw: dict, address: str) -> TradeRecord | None:
    """
    Map a raw /data/trades response object to a TradeRecord.

    Confirmed field names from the CLOB /data/trades endpoint:
      id, market (condition_id), asset_id (token_id), side, size, price,
      status, match_time, outcome, maker_address, transaction_hash
    """
    try:
        # outcome field: "YES" / "NO" — resolve relative to the BUY/SELL side
        size = float(raw.get("size", 0))
        price = float(raw.get("price", 0))
        side = (raw.get("side") or "BUY").upper()

        # "outcome" is the token outcome string ("YES"/"NO"), not win/loss.
        # A BUY YES that resolves YES = WIN; BUY YES that resolves NO = LOSS.
        # A SELL YES that resolves NO = WIN (bet against); etc.
        outcome_token = str(raw.get("outcome", "")).upper()  # "YES" or "NO"
        trade_status = str(raw.get("status", "")).upper()    # CONFIRMED, MATCHED, etc.

        outcome: str
        pnl: float | None = None
        if trade_status not in ("CONFIRMED", "MATCHED", "MINED"):
            # Not yet settled — skip unresolved trades
            outcome = "UNRESOLVED"
        elif outcome_token == "YES":
            if side == "BUY":
                outcome = "WIN"
                pnl = size * (1.0 - price)
            else:
                outcome = "LOSS"
                pnl = -size * price
        elif outcome_token == "NO":
            if side == "SELL":
                outcome = "WIN"
                pnl = size * price
            else:
                outcome = "LOSS"
                pnl = -size * (1.0 - price)
        else:
            outcome = "UNRESOLVED"

        # match_time is a Unix second string
        ts_raw = raw.get("match_time") or raw.get("timestamp") or 0
        try:
            ts = int(float(ts_raw)) * 1000  # normalise to milliseconds
        except (TypeError, ValueError):
            ts = 0

        return TradeRecord(
            trade_id=str(raw.get("id", "")),
            market_id=str(raw.get("market", "")),
            token_id=str(raw.get("asset_id", "")),
            side=side,
            price=price,
            size=size,
            timestamp=ts,
            outcome=outcome,
            pnl=pnl,
            market_category=str(raw.get("category", "unknown")).lower(),
        )
    except Exception as exc:
        logger.debug("Could not parse trade for %s: %s", address, exc)
        return None


class WhaleFetcher:
    """
    Async fetcher for Polymarket wallet data.

    Usage:
        async with WhaleFetcher() as fetcher:
            addresses = await fetcher.get_top_wallet_addresses(10000)
            profiles   = await fetcher.fetch_wallet_profiles(addresses)
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._rate_limiter = _TokenBucket(cfg.rate_limit_rps)

    async def __aenter__(self) -> "WhaleFetcher":
        connector = aiohttp.TCPConnector(
            limit=cfg.connector_limit,
            limit_per_host=cfg.connector_limit_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_sec)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            # orjson-backed response reading via read() + orjson.loads is faster
            # than the default aiohttp JSON decoder
            json_serialize=lambda x: orjson.dumps(x).decode(),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_json(self, url: str, params: dict | None = None) -> Any:
        """GET with retry and rate limiting, returns parsed JSON using orjson."""
        assert self._session is not None, "Use as async context manager"
        await self._rate_limiter.acquire()
        for attempt in range(cfg.max_retries):
            try:
                async with self._session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    raw_bytes = await resp.read()
                    return orjson.loads(raw_bytes)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == cfg.max_retries - 1:
                    logger.warning("Failed GET %s after %d attempts: %s", url, cfg.max_retries, exc)
                    raise
                await asyncio.sleep(0.2 * (attempt + 1))

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_top_wallet_addresses(self, limit: int = 10_000) -> list[str]:
        """
        Fetch the top `limit` wallet addresses ranked by P&L from the
        Polymarket leaderboard API.

        Returns a flat list of EVM address strings.
        """
        logger.info("Fetching top %d wallet addresses from leaderboard...", limit)
        addresses: list[str] = []
        seen: set[str] = set()   # O(1) membership test; list search is O(n²) at 10k
        page_size = 100
        offset = 0

        while len(addresses) < limit:
            fetch_n = min(page_size, limit - len(addresses))
            try:
                # Confirmed endpoint: GET /v1/leaderboard on data-api.polymarket.com
                # Params: timePeriod (not "window"), limit, offset
                data = await self._get_json(
                    f"{cfg.leaderboard_url}/v1/leaderboard",
                    params={
                        "timePeriod": "all",
                        "limit": fetch_n,
                        "offset": offset,
                    },
                )
            except Exception:
                logger.warning("Leaderboard fetch failed at offset %d, stopping.", offset)
                break

            # Response is a plain list of objects (not wrapped in a key)
            # Address field is "proxyWallet" (confirmed from live API response)
            rows: list[dict] = []
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get("data", data.get("results", data.get("rankings", [])))

            if not rows:
                break

            for row in rows:
                # Primary field confirmed from live API: "proxyWallet"
                addr = (
                    row.get("proxyWallet")
                    or row.get("proxy_wallet_address")
                    or row.get("address")
                )
                if addr and addr not in seen:
                    addresses.append(addr)
                    seen.add(addr)

            offset += len(rows)
            if len(rows) < fetch_n:
                break   # no more pages

            logger.debug("  ... %d addresses collected so far", len(addresses))

        logger.info("Collected %d unique wallet addresses.", len(addresses))
        return addresses[:limit]

    async def _fetch_single_profile(
        self,
        address: str,
        sem: asyncio.Semaphore,
    ) -> WalletProfile | None:
        """Fetch all available trade history for one wallet."""
        async with sem:
            profile = WalletProfile(address=address)
            page_size = 500
            # /data/trades uses cursor-based pagination (not offset).
            # Initial cursor is "MA==" (base64 of "0"); empty string also works.
            cursor = "MA=="

            while True:
                try:
                    data = await self._get_json(
                        f"{cfg.clob_url}/data/trades",
                        params={
                            "maker_address": address,
                            "limit": page_size,
                            "next_cursor": cursor,
                        },
                    )
                except Exception as exc:
                    logger.debug("Trade fetch failed for %s: %s", address, exc)
                    break

                trades_raw: list[dict] = []
                next_cursor: str | None = None
                if isinstance(data, list):
                    trades_raw = data
                elif isinstance(data, dict):
                    trades_raw = data.get("data", data.get("trades", []))
                    next_cursor = data.get("next_cursor")

                if not trades_raw:
                    break

                for raw in trades_raw:
                    t = _parse_trade(raw, address)
                    if t:
                        profile.trades.append(t)

                # Advance cursor; stop when API signals last page
                if not next_cursor or next_cursor == cursor or len(trades_raw) < page_size:
                    break
                cursor = next_cursor

            if profile.trades:
                profile.total_volume_usdc = sum(t.size for t in profile.trades)
                timestamps = [t.timestamp for t in profile.trades]
                profile.first_seen_ts = min(timestamps)
                profile.last_seen_ts = max(timestamps)

            return profile

    async def fetch_wallet_profiles(
        self,
        addresses: list[str],
        concurrency: int = _CONCURRENCY,
    ) -> list[WalletProfile]:
        """
        Concurrently fetch trade histories for all addresses.
        Returns only profiles that have >= cfg.min_trade_count trades.
        """
        logger.info(
            "Fetching trade histories for %d wallets (concurrency=%d)...",
            len(addresses), concurrency,
        )
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            asyncio.create_task(self._fetch_single_profile(addr, sem))
            for addr in addresses
        ]

        results: list[WalletProfile] = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            profile = await coro
            if profile and profile.trade_count >= cfg.min_trade_count:
                results.append(profile)
            if (i + 1) % 500 == 0:
                logger.info("  ... %d / %d wallets fetched", i + 1, len(addresses))

        logger.info(
            "Profiles fetched: %d / %d met minimum trade count (%d).",
            len(results), len(addresses), cfg.min_trade_count,
        )
        return results

    async def get_current_orderbook(self, token_id: str) -> dict:
        """
        Fetch live orderbook for a token. Called by EdgeScanner — always fresh,
        never cached.
        """
        data = await self._get_json(
            f"{cfg.clob_url}/book",
            params={"token_id": token_id},
        )
        return data if isinstance(data, dict) else {}

    async def get_recent_trades(self, maker_address: str, after_ts_sec: int) -> list[dict]:
        """
        Fetch trades placed by `maker_address` after `after_ts_sec` (Unix seconds).
        Used by WhalePollWatcher to detect new whale trades in near-real-time.
        Returns a list of raw trade dicts from /data/trades.
        """
        try:
            data = await self._get_json(
                f"{cfg.clob_url}/data/trades",
                params={
                    "maker_address": maker_address,
                    "after": after_ts_sec,
                    "limit": 50,
                },
            )
        except Exception as exc:
            logger.debug("get_recent_trades failed for %s: %s", maker_address, exc)
            return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", data.get("trades", []))
        return []

    async def get_clob_market(self, condition_id: str) -> dict | None:
        """
        Fetch a single CLOB market by condition_id.
        Returns the market dict (includes tokens[].winner for resolution) or None.

        Endpoint: GET /markets?condition_id=<id>  (not /markets/{id} — that path
        doesn't exist on the CLOB). The response is a list; we return the first item.
        """
        try:
            data = await self._get_json(
                f"{cfg.clob_url}/markets",
                params={"condition_id": condition_id},
            )
            # /markets returns a paginated object: {"data": [...], "next_cursor": "..."}
            markets: list[dict] = []
            if isinstance(data, list):
                markets = data
            elif isinstance(data, dict):
                markets = data.get("data", data.get("markets", []))
            return markets[0] if markets else None
        except Exception as exc:
            logger.debug("get_clob_market failed for %s: %s", condition_id, exc)
            return None

    async def get_market_price(self, token_id: str) -> float | None:
        """
        Returns the mid-price for a token (0–1) or None on failure.
        Always fetches live — no cache.
        """
        try:
            data = await self._get_json(
                f"{cfg.clob_url}/price",
                params={"token_id": token_id, "side": "buy"},
            )
            if isinstance(data, dict):
                price_str = data.get("price")
                if price_str is not None:
                    return float(price_str)
        except Exception:
            pass
        return None
