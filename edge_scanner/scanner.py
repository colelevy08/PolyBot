"""
EdgeScanner — evaluates whether a detected whale trade still has positive
expected value by the time WE can execute it.

Design principles:
- ALWAYS fetches live orderbook — never uses cache.
- Estimates "true probability" using a weighted combination of:
    a) The whale's implied bet direction / price
    b) Current market mid-price
    c) Orderbook depth imbalance
- Returns an EdgeResult with has_edge, bet_size, and scan_ts so the
  executor can immediately fire if edge is confirmed.

Speed:
- Uses the shared aiohttp session from WhaleFetcher (passed in).
- Single HTTP request per scan (the orderbook endpoint returns bid + ask).
- No object allocation beyond the returned EdgeResult.
"""
from __future__ import annotations

import logging
import time

from config import cfg
from whale_analyzer.fetcher import WhaleFetcher
from whale_analyzer.models import EdgeResult, WhaleSignal
from edge_scanner.kelly import kelly_size, kelly_size_no

logger = logging.getLogger(__name__)


def _depth_imbalance(bids: list[dict], asks: list[dict]) -> float:
    """
    Returns a value in [-1, 1]:
      +1 → all depth on bid side (strong buying pressure → price likely higher)
      -1 → all depth on ask side (selling pressure → price likely lower)
    """
    bid_vol = sum(float(b.get("size", 0)) for b in bids)
    ask_vol = sum(float(a.get("size", 0)) for a in asks)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def _estimate_true_prob(
    whale_price: float,
    mid_price: float,
    depth_imb: float,
    whale_side: str,
) -> float:
    """
    Weighted blend of signals to estimate true probability.

    Weights:
      - whale_price:  50%  (the whale is our signal; their price is the prior)
      - mid_price:    30%  (current market consensus)
      - depth_adj:    20%  (orderbook pressure adjustment)
    """
    depth_adj = mid_price + depth_imb * 0.05  # max ±5 cents from depth

    # If whale bought YES, their price is a lower bound on true prob
    # If whale bought NO, their implied YES prior = 1 - whale_price
    if whale_side == "BUY":
        whale_signal = whale_price
    else:
        whale_signal = 1.0 - whale_price

    true_prob = (
        0.50 * whale_signal
        + 0.30 * mid_price
        + 0.20 * depth_adj
    )
    # Clamp to [cfg.min_prob, cfg.max_prob]
    return max(cfg.min_prob, min(cfg.max_prob, true_prob))


class EdgeScanner:
    """
    Scans a WhaleSignal for edge and returns an EdgeResult.

    Requires a WhaleFetcher instance (for its live HTTP session).
    """

    def __init__(self, fetcher: WhaleFetcher, bankroll_usdc: float | None = None) -> None:
        self._fetcher = fetcher
        self._bankroll = bankroll_usdc or cfg.bankroll_usdc

    async def scan(self, signal: WhaleSignal) -> EdgeResult:
        """
        Evaluate edge for a WhaleSignal. Always fetches live orderbook.
        Returns EdgeResult with has_edge=False and bet_size=0 on any error.
        """
        scan_start_ms = int(time.time() * 1000)
        token_id = signal.token_id

        # ── Fetch live orderbook ──────────────────────────────────────────────
        try:
            book = await self._fetcher.get_current_orderbook(token_id)
        except Exception as exc:
            logger.warning("Orderbook fetch failed for %s: %s", token_id, exc)
            return self._no_edge(token_id, signal.price, scan_start_ms)

        bids: list[dict] = book.get("bids", [])
        asks: list[dict] = book.get("asks", [])

        # ── Best prices ───────────────────────────────────────────────────────
        best_ask = float(asks[0]["price"]) if asks else None
        best_bid = float(bids[0]["price"]) if bids else None

        if best_ask is None or best_bid is None:
            logger.debug("Incomplete orderbook for %s", token_id)
            return self._no_edge(token_id, signal.price, scan_start_ms)

        mid_price = (best_ask + best_bid) / 2.0
        depth_imb = _depth_imbalance(bids, asks)

        # ── True probability estimate ─────────────────────────────────────────
        true_prob = _estimate_true_prob(
            whale_price=signal.price,
            mid_price=mid_price,
            depth_imb=depth_imb,
            whale_side=signal.side,
        )

        # ── Edge calculation ──────────────────────────────────────────────────
        # We are copying the whale: if they BUY YES, we also BUY YES at best_ask.
        # Edge = true_prob - our_price  (for YES purchases)
        if signal.side == "BUY":
            our_price = best_ask
            edge = true_prob - our_price
            bet_usdc = kelly_size(true_prob, our_price, self._bankroll) if edge > 0 else 0.0
        else:
            # Whale sold YES (or bought NO); we follow buying NO at best_ask_no
            # Price of NO = 1 - YES_price
            our_price_no = 1.0 - best_bid   # best ask for NO ≈ 1 - best_bid_yes
            edge = (1.0 - true_prob) - our_price_no
            bet_usdc = kelly_size_no(true_prob, best_bid, self._bankroll) if edge > 0 else 0.0
            our_price = our_price_no

        has_edge = edge >= cfg.min_edge_threshold

        if has_edge:
            logger.info(
                "EDGE FOUND  token=%s  edge=%.3f  our_price=%.3f  "
                "true_prob=%.3f  bet=$%.2f",
                token_id, edge, our_price, true_prob, bet_usdc,
            )
        else:
            logger.debug(
                "No edge  token=%s  edge=%.3f (< %.3f threshold)",
                token_id, edge, cfg.min_edge_threshold,
            )

        return EdgeResult(
            token_id=token_id,
            has_edge=has_edge,
            our_price=our_price,
            whale_price=signal.price,
            mid_price=mid_price,
            edge=edge,
            true_prob_est=true_prob,
            kelly_fraction=bet_usdc / self._bankroll if self._bankroll > 0 else 0.0,
            bet_size_usdc=bet_usdc,
            scan_ts=int(time.time() * 1000),
        )

    @staticmethod
    def _no_edge(token_id: str, whale_price: float, scan_ts: int) -> EdgeResult:
        return EdgeResult(
            token_id=token_id,
            has_edge=False,
            our_price=0.0,
            whale_price=whale_price,
            mid_price=0.0,
            edge=0.0,
            true_prob_est=0.0,
            kelly_fraction=0.0,
            bet_size_usdc=0.0,
            scan_ts=scan_ts,
        )
