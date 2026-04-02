"""
OrderManager — lightweight in-memory tracker for open and closed orders.

Responsibilities:
- Prevent duplicate orders on the same token (dedup at position level).
- Track P&L on closed positions.
- Expose summary stats for monitoring.

All operations are synchronous (no I/O); called from the async executor
but never awaited.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import orjson

logger = logging.getLogger(__name__)


@dataclass
class OpenOrder:
    __slots__ = (
        "order_id", "token_id", "market_id", "side",
        "price", "size_usdc", "whale_address", "placed_ts",
    )
    order_id: str
    token_id: str
    market_id: str
    side: str
    price: float
    size_usdc: float
    whale_address: str
    placed_ts: int  # unix ms


@dataclass
class ClosedOrder:
    __slots__ = (
        "order_id", "token_id", "market_id", "side",
        "entry_price", "exit_price", "size_usdc",
        "pnl_usdc", "placed_ts", "closed_ts",
    )
    order_id: str
    token_id: str
    market_id: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    size_usdc: float
    pnl_usdc: float
    placed_ts: int
    closed_ts: int


_STATE_PATH = "data/order_manager_state.json"


class OrderManager:
    """Thread-safe (via asyncio single-thread) order book."""

    def __init__(self) -> None:
        # token_id → OpenOrder
        self._open: dict[str, OpenOrder] = {}
        # Append-only closed list
        self._closed: list[ClosedOrder] = []

    def record_order(
        self,
        order_id: str,
        token_id: str,
        market_id: str,
        side: str,
        price: float,
        size_usdc: float,
        whale_address: str,
    ) -> None:
        self._open[token_id] = OpenOrder(
            order_id=order_id,
            token_id=token_id,
            market_id=market_id,
            side=side,
            price=price,
            size_usdc=size_usdc,
            whale_address=whale_address,
            placed_ts=int(time.time() * 1000),
        )
        logger.debug("Recorded open order %s for token %s", order_id, token_id)

    def has_open_position(self, token_id: str) -> bool:
        return token_id in self._open

    def close_order(
        self,
        token_id: str,
        exit_price: Optional[float],
        resolved_yes: Optional[bool],
    ) -> None:
        """
        Mark a position as closed. `resolved_yes` indicates the binary outcome.
        exit_price is used for manual sells; resolved_yes for market resolution.
        """
        order = self._open.pop(token_id, None)
        if order is None:
            logger.warning("close_order called for unknown token_id %s", token_id)
            return

        # Calculate P&L
        # size_usdc is the USDC notional spent (the bet amount).
        # For a binary market BUY at price p, you buy (size_usdc / p) shares.
        # WIN: each share pays $1 → profit = shares * (1 - p) = size_usdc * (1-p)/p
        # LOSS: shares pay $0 → loss = -size_usdc (all invested capital lost)
        if resolved_yes is not None:
            if order.side == "BUY":
                if resolved_yes:
                    pnl = order.size_usdc * (1.0 - order.price) / order.price
                else:
                    pnl = -order.size_usdc
            else:  # SELL (selling YES = buying NO equivalent)
                if not resolved_yes:
                    pnl = order.size_usdc * order.price / (1.0 - order.price)
                else:
                    pnl = -order.size_usdc
            ep = 1.0 if resolved_yes else 0.0
        elif exit_price is not None:
            # Manual exit
            if order.side == "BUY":
                pnl = order.size_usdc * (exit_price - order.price) / order.price
            else:
                pnl = order.size_usdc * (order.price - exit_price) / (1 - order.price)
            ep = exit_price
        else:
            pnl = 0.0
            ep = None

        closed = ClosedOrder(
            order_id=order.order_id,
            token_id=token_id,
            market_id=order.market_id,
            side=order.side,
            entry_price=order.price,
            exit_price=ep,
            size_usdc=order.size_usdc,
            pnl_usdc=pnl,
            placed_ts=order.placed_ts,
            closed_ts=int(time.time() * 1000),
        )
        self._closed.append(closed)
        logger.info(
            "Position closed  token=%s  pnl=$%.2f  side=%s  entry=%.3f  exit=%s",
            token_id, pnl, order.side, order.price,
            f"{ep:.3f}" if ep is not None else "N/A",
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = _STATE_PATH) -> None:
        """Persist open and closed orders to disk as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "open": [
                {
                    "order_id": o.order_id, "token_id": o.token_id,
                    "market_id": o.market_id, "side": o.side,
                    "price": o.price, "size_usdc": o.size_usdc,
                    "whale_address": o.whale_address, "placed_ts": o.placed_ts,
                }
                for o in self._open.values()
            ],
            "closed": [
                {
                    "order_id": c.order_id, "token_id": c.token_id,
                    "market_id": c.market_id, "side": c.side,
                    "entry_price": c.entry_price, "exit_price": c.exit_price,
                    "size_usdc": c.size_usdc, "pnl_usdc": c.pnl_usdc,
                    "placed_ts": c.placed_ts, "closed_ts": c.closed_ts,
                }
                for c in self._closed
            ],
        }
        with open(path, "wb") as f:
            f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2))
        logger.debug("OrderManager state saved (%d open, %d closed)", len(self._open), len(self._closed))

    def load(self, path: str = _STATE_PATH) -> None:
        """Restore state from disk. No-op if file does not exist."""
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                state = orjson.loads(f.read())
            for o in state.get("open", []):
                self._open[o["token_id"]] = OpenOrder(**o)
            for c in state.get("closed", []):
                self._closed.append(ClosedOrder(**c))
            logger.info(
                "OrderManager state loaded: %d open, %d closed positions.",
                len(self._open), len(self._closed),
            )
        except Exception as exc:
            logger.warning("Failed to load OrderManager state from %s: %s", path, exc)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def open_count(self) -> int:
        return len(self._open)

    @property
    def total_pnl(self) -> float:
        return sum(o.pnl_usdc for o in self._closed)

    @property
    def win_rate(self) -> float:
        if not self._closed:
            return 0.0
        wins = sum(1 for o in self._closed if o.pnl_usdc > 0)
        return wins / len(self._closed)

    def open_positions_map(self) -> dict[str, str]:
        """Return {token_id: market_id (condition_id)} for all open positions."""
        return {token_id: o.market_id for token_id, o in self._open.items()}

    def summary(self) -> dict:
        return {
            "open_positions": self.open_count,
            "closed_positions": len(self._closed),
            "total_pnl_usdc": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "open_tokens": list(self._open.keys()),
        }
