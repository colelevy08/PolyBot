"""
WalletScorer — computes a multi-factor composite score for each WalletProfile.

Factors (all normalised to [0, 1] before weighting):
  1. Win rate              — % of resolved trades won
  2. Profit factor         — gross_profit / gross_loss
  3. Expected value / trade — mean USDC P&L per trade
  4. Sharpe ratio          — EV / std(pnl) * sqrt(n)
  5. Recency weight        — bias towards wallets active in last 90 days
  6. Average edge          — mean (outcome - entry_price) per winning trade
  7. Trade count           — logarithmic; more trades = more confidence

Weights are intentionally conservative to avoid over-fitting on small samples.
"""
from __future__ import annotations

import logging
import math
import time
import numpy as np

from whale_analyzer.models import TradeRecord, WalletProfile, WalletScore

logger = logging.getLogger(__name__)

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
_WEIGHTS = {
    "win_rate":            0.20,
    "profit_factor":       0.20,
    "ev_per_trade":        0.20,
    "sharpe":              0.15,
    "recency":             0.10,
    "avg_edge":            0.10,
    "trade_count_log":     0.05,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

_90_DAYS_MS = 90 * 24 * 3600 * 1000


def _compute_profit_factor(trades: list[TradeRecord]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl is not None and t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl is not None and t.pnl < 0))
    if gross_loss == 0:
        # No losing trades: profit factor is effectively infinite. Return a large
        # sentinel (100.0) so min-max normalisation maps these wallets to 1.0,
        # above all wallets that have at least one loss. Returning 0.0 here would
        # incorrectly rank perfect wallets last.
        return 100.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _compute_sharpe(pnls: np.ndarray) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = pnls.mean()
    std = pnls.std()
    if std == 0:
        return 0.0
    # Annualised-ish: multiply by sqrt(n) as a scale factor
    return float((mean / std) * math.sqrt(len(pnls)))


def _compute_recency(profile: WalletProfile) -> float:
    """0–1: fraction of trades in last 90 days."""
    now_ms = int(time.time() * 1000)  # always current, not stale from import time
    recent = sum(1 for t in profile.trades if (now_ms - t.timestamp) < _90_DAYS_MS)
    total = max(profile.trade_count, 1)
    return recent / total


def _compute_avg_edge(trades: list[TradeRecord]) -> float:
    """
    Edge per winning BUY trade = (1.0 - entry_price).
    We compare the entry price to 1.0 (full resolution) to estimate how much
    value was captured. This is a proxy for implied edge at entry.
    For SELL trades, edge = entry_price (selling at high probability).
    """
    edges = []
    for t in trades:
        if t.outcome == "WIN":
            if t.side == "BUY":
                edges.append(1.0 - t.price)
            else:
                edges.append(t.price)
    if not edges:
        return 0.0
    return float(np.mean(edges))


def _compute_category_breakdown(trades: list[TradeRecord]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in trades:
        counts[t.market_category] = counts.get(t.market_category, 0) + 1
    total = max(sum(counts.values()), 1)
    return {cat: cnt / total for cat, cnt in counts.items()}


def _score_single(profile: WalletProfile) -> WalletScore | None:
    """Compute raw (un-normalised) metrics for one wallet."""
    resolved = profile.resolved_trades
    if not resolved:
        return None

    pnls = np.array([t.pnl for t in resolved if t.pnl is not None], dtype=np.float64)
    if len(pnls) == 0:
        return None

    win_rate = profile.win_rate
    profit_factor = _compute_profit_factor(resolved)
    ev_per_trade = float(pnls.mean())
    sharpe = _compute_sharpe(pnls)
    recency = _compute_recency(profile)
    avg_edge = _compute_avg_edge(resolved)
    trade_count_log = math.log1p(profile.trade_count)
    category_breakdown = _compute_category_breakdown(profile.trades)

    return WalletScore(
        address=profile.address,
        composite_score=0.0,    # filled in after normalisation
        win_rate=win_rate,
        profit_factor=profit_factor,
        expected_value_per_trade=ev_per_trade,
        sharpe_ratio=sharpe,
        trade_count=profile.trade_count,
        total_pnl_usdc=profile.total_pnl,
        recency_weight=recency,
        avg_edge=avg_edge,
        category_breakdown=category_breakdown,
        rank=0,
    )


def _minmax_normalise(values: list[float]) -> list[float]:
    """Normalise a list to [0, 1]. Returns zeros if all values are identical."""
    arr = np.array(values, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [0.5] * len(values)
    return list((arr - mn) / (mx - mn))


class WalletScorer:
    """
    Scores a batch of WalletProfile objects and returns WalletScore objects
    sorted by composite_score descending.
    """

    def score(self, profiles: list[WalletProfile]) -> list[WalletScore]:
        logger.info("Scoring %d wallet profiles...", len(profiles))

        raw_scores: list[WalletScore] = []
        for p in profiles:
            s = _score_single(p)
            if s is not None:
                raw_scores.append(s)

        if not raw_scores:
            return []

        # Collect raw metric arrays for normalisation
        def col(attr: str) -> list[float]:
            return [getattr(s, attr) for s in raw_scores]

        norm = {
            "win_rate":         _minmax_normalise(col("win_rate")),
            "profit_factor":    _minmax_normalise(col("profit_factor")),
            "ev_per_trade":     _minmax_normalise(col("expected_value_per_trade")),
            "sharpe":           _minmax_normalise(col("sharpe_ratio")),
            "recency":          _minmax_normalise(col("recency_weight")),
            "avg_edge":         _minmax_normalise(col("avg_edge")),
            # log trade count already relative; just normalise again
            "trade_count_log":  _minmax_normalise(
                [math.log1p(s.trade_count) for s in raw_scores]
            ),
        }

        for i, score in enumerate(raw_scores):
            composite = (
                _WEIGHTS["win_rate"]        * norm["win_rate"][i]
                + _WEIGHTS["profit_factor"] * norm["profit_factor"][i]
                + _WEIGHTS["ev_per_trade"]  * norm["ev_per_trade"][i]
                + _WEIGHTS["sharpe"]        * norm["sharpe"][i]
                + _WEIGHTS["recency"]       * norm["recency"][i]
                + _WEIGHTS["avg_edge"]      * norm["avg_edge"][i]
                + _WEIGHTS["trade_count_log"] * norm["trade_count_log"][i]
            )
            score.composite_score = composite

        # Sort by composite score descending and assign ranks
        raw_scores.sort(key=lambda s: s.composite_score, reverse=True)
        for rank, score in enumerate(raw_scores, start=1):
            score.rank = rank

        logger.info(
            "Scoring complete. Top wallet: %s (score=%.4f, win_rate=%.2f%%, PnL=%.2f)",
            raw_scores[0].address,
            raw_scores[0].composite_score,
            raw_scores[0].win_rate * 100,
            raw_scores[0].total_pnl_usdc,
        )
        return raw_scores
