"""
Data models for wallet analysis.
Using __slots__ on hot-path dataclasses to minimize memory overhead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TradeRecord:
    """Single resolved trade for a wallet."""
    __slots__ = (
        "trade_id", "market_id", "token_id", "side",
        "price", "size", "timestamp", "outcome",
        "pnl", "market_category",
    )
    trade_id: str
    market_id: str
    token_id: str
    side: str           # "BUY" | "SELL"
    price: float        # 0–1
    size: float         # USDC notional
    timestamp: int      # unix ms
    outcome: Optional[str]   # "WIN" | "LOSS" | "UNRESOLVED"
    pnl: Optional[float]     # USDC profit/loss on this trade
    market_category: str     # "crypto" | "politics" | "sports" | etc.


@dataclass
class WalletProfile:
    """Raw data collected for a single wallet address."""
    address: str
    trades: list[TradeRecord] = field(default_factory=list)
    total_volume_usdc: float = 0.0
    first_seen_ts: int = 0
    last_seen_ts: int = 0
    fetch_ts: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def resolved_trades(self) -> list[TradeRecord]:
        return [t for t in self.trades if t.outcome in ("WIN", "LOSS")]

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.resolved_trades if t.outcome == "WIN")

    @property
    def win_rate(self) -> float:
        rt = self.resolved_trades
        if not rt:
            return 0.0
        return self.win_count / len(rt)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl is not None)


@dataclass
class WalletScore:
    """
    Composite score for ranking wallets.
    Higher is better across all metrics.
    """
    __slots__ = (
        "address",
        "composite_score",
        "win_rate",
        "profit_factor",
        "expected_value_per_trade",
        "sharpe_ratio",
        "trade_count",
        "total_pnl_usdc",
        "recency_weight",
        "avg_edge",
        "category_breakdown",
        "rank",
    )
    address: str
    composite_score: float      # Final ranking score (higher = better)
    win_rate: float             # % of resolved trades that were winners
    profit_factor: float        # gross_profit / gross_loss (>1 = profitable)
    expected_value_per_trade: float   # average USDC per trade
    sharpe_ratio: float         # risk-adjusted return
    trade_count: int            # total trades analysed
    total_pnl_usdc: float       # cumulative P&L in USDC
    recency_weight: float       # how active recently (0–1)
    avg_edge: float             # average implied edge per trade
    category_breakdown: dict    # {"crypto": 0.4, "politics": 0.3, ...}
    rank: int = 0               # set after sorting


@dataclass
class WhaleSignal:
    """
    A trade detected from a tracked whale, ready for the copy engine.
    This object is created ONCE and passed through the pipeline with
    zero copies of its underlying byte data where possible.
    """
    __slots__ = (
        "whale_address",
        "market_id",
        "token_id",
        "side",
        "price",
        "size_usdc",
        "detected_ts",          # unix ms when we first saw this
        "raw_event",            # reference to raw dict — no copy
    )
    whale_address: str
    market_id: str
    token_id: str
    side: str           # "BUY" | "SELL"
    price: float
    size_usdc: float
    detected_ts: int    # unix ms
    raw_event: dict     # kept for debugging; not serialised


@dataclass
class EdgeResult:
    """Output from EdgeScanner for a given WhaleSignal."""
    __slots__ = (
        "token_id",
        "has_edge",
        "our_price",        # best price we can get right now
        "whale_price",      # price whale paid
        "mid_price",        # (best_ask + best_bid) / 2
        "edge",             # our_price vs true_prob estimate
        "true_prob_est",    # estimated true probability
        "kelly_fraction",   # raw kelly fraction
        "bet_size_usdc",    # kelly-sized bet in USDC
        "scan_ts",          # unix ms when scan completed
    )
    token_id: str
    has_edge: bool
    our_price: float
    whale_price: float
    mid_price: float
    edge: float
    true_prob_est: float
    kelly_fraction: float
    bet_size_usdc: float
    scan_ts: int
