"""
WhaleReducer — iteratively re-fetches and re-scores wallet subsets, drilling
down through the reduction ladder until only `target_count` whales remain.

Ladder (configurable in config.py):
  10000 → 5000 → 2500 → 1250 → 750 → 375 → 188 → 94 → 47 → 23 → 15

At each step:
  1. Keep top N addresses from the previous scoring.
  2. Re-fetch their full trade history (fresh data, no cache).
  3. Re-score with tighter scrutiny (same algorithm; dataset just shrinks).
  4. The final cohort of 15 has been validated through every round.

We NEVER rely on stale/cached data — every reduction step fetches live.
"""
from __future__ import annotations

import logging
import time

import orjson

from config import cfg
from whale_analyzer.fetcher import WhaleFetcher
from whale_analyzer.models import WalletProfile, WalletScore
from whale_analyzer.scorer import WalletScorer

logger = logging.getLogger(__name__)


class WhaleReducer:
    """
    Orchestrates the full multi-round reduction pipeline.

    Returns a list of WalletScore objects for the final top whales,
    and writes results to cfg.whale_data_path.
    """

    def __init__(self) -> None:
        self._scorer = WalletScorer()

    async def run(self, fetcher: WhaleFetcher) -> list[WalletScore]:
        """
        Full pipeline:
          1. Fetch top 10 000 wallet addresses.
          2. Fetch their trade histories.
          3. Score and reduce through the ladder.
          4. Return final top-N WalletScore list.
        """
        ladder = cfg.reduction_ladder  # e.g. [10000, 5000, ..., 15]
        target = cfg.top_whale_count   # 15

        # ── Round 0: seed addresses ───────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("WHALE ANALYSIS — ROUND 0: fetching initial address list")
        logger.info("=" * 60)
        addresses = await fetcher.get_top_wallet_addresses(ladder[0])

        if not addresses:
            raise RuntimeError("Could not fetch any wallet addresses. Check API connectivity.")

        current_addresses = addresses
        final_scores: list[WalletScore] = []

        # ── Reduction rounds ──────────────────────────────────────────────────
        for round_idx, keep_n in enumerate(ladder[1:], start=1):
            step_start = time.perf_counter()
            logger.info("")
            logger.info("=" * 60)
            logger.info(
                "ROUND %d: analysing %d wallets → keeping top %d",
                round_idx, len(current_addresses), keep_n,
            )
            logger.info("=" * 60)

            # Always fetch fresh — no memory / cache
            profiles: list[WalletProfile] = await fetcher.fetch_wallet_profiles(
                current_addresses
            )

            if not profiles:
                logger.error("No profiles returned in round %d — aborting.", round_idx)
                break

            scores: list[WalletScore] = self._scorer.score(profiles)

            # Keep only top `keep_n` for the next round
            keep = min(keep_n, len(scores))
            top_scores = scores[:keep]
            current_addresses = [s.address for s in top_scores]
            final_scores = top_scores

            elapsed = time.perf_counter() - step_start
            logger.info(
                "Round %d complete in %.1fs — %d wallets remain.",
                round_idx, elapsed, len(current_addresses),
            )
            self._log_top5(top_scores)

            if keep <= target:
                logger.info("Reached target of %d — stopping.", target)
                break

        # ── Final cohort ──────────────────────────────────────────────────────
        final = final_scores[:target]
        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL TOP %d WHALES", len(final))
        logger.info("=" * 60)
        for score in final:
            logger.info(
                "  #%d  %s  score=%.4f  win_rate=%.1f%%  PnL=$%.2f  trades=%d",
                score.rank,
                score.address,
                score.composite_score,
                score.win_rate * 100,
                score.total_pnl_usdc,
                score.trade_count,
            )

        self._save(final)
        return final

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _log_top5(scores: list[WalletScore]) -> None:
        for s in scores[:5]:
            logger.info(
                "  top5 → %s  score=%.4f  win_rate=%.1f%%  PnL=$%.2f",
                s.address, s.composite_score, s.win_rate * 100, s.total_pnl_usdc,
            )

    @staticmethod
    def _save(scores: list[WalletScore]) -> None:
        import os
        os.makedirs("data", exist_ok=True)
        payload = [
            {
                "rank": s.rank,
                "address": s.address,
                "composite_score": s.composite_score,
                "win_rate": s.win_rate,
                "profit_factor": s.profit_factor,
                "ev_per_trade": s.expected_value_per_trade,
                "sharpe_ratio": s.sharpe_ratio,
                "trade_count": s.trade_count,
                "total_pnl_usdc": s.total_pnl_usdc,
                "recency_weight": s.recency_weight,
                "avg_edge": s.avg_edge,
                "category_breakdown": s.category_breakdown,
            }
            for s in scores
        ]
        path = cfg.whale_data_path
        with open(path, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        logger.info("Whale data saved → %s", path)

    @staticmethod
    def load_whale_addresses() -> list[str]:
        """
        Load previously saved whale addresses from disk.
        Called by main.py to seed the copy trader without re-running analysis.
        """
        import os
        path = cfg.whale_data_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No whale data found at '{path}'. "
                "Run `python run_analysis.py` first."
            )
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        addresses = [entry["address"] for entry in data]
        logger.info("Loaded %d whale addresses from %s", len(addresses), path)
        return addresses
