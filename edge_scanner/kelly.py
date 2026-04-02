"""
Kelly criterion position sizing.

Full Kelly: f* = (b*p - q) / b
  where b = net odds (payout per unit risked)
        p = estimated win probability
        q = 1 - p

We ALWAYS apply a fractional multiplier (cfg.kelly_fraction, default 0.25)
and cap at cfg.max_position_fraction of bankroll.

Polymarket is a binary market (prices are probabilities, payout = $1 per share):
  - BUY YES at price p:  net odds b = (1-p)/p
  - BUY NO  at price p:  net odds b = p/(1-p)  (price of NO = 1-p_yes)
"""
from __future__ import annotations

import logging

from config import cfg

logger = logging.getLogger(__name__)


def kelly_size(
    true_prob: float,
    market_price: float,
    bankroll_usdc: float | None = None,
) -> float:
    """
    Compute Kelly-sized bet in USDC for buying YES at `market_price`
    given an estimated `true_prob`.

    Args:
        true_prob:     Estimated true probability of YES (0–1).
        market_price:  Current market price for YES (0–1).
        bankroll_usdc: Current bankroll. Defaults to cfg.bankroll_usdc.

    Returns:
        Bet size in USDC (0 if no edge or calculation invalid).
    """
    bankroll = bankroll_usdc if bankroll_usdc is not None else cfg.bankroll_usdc

    if market_price <= 0 or market_price >= 1:
        return 0.0
    if true_prob <= market_price:
        return 0.0  # No edge

    # Binary market: buying YES at price p
    # Net odds b = (1 - p) / p  (win (1-p), risk p per share, normalised per unit risked)
    p = true_prob
    q = 1.0 - p
    b = (1.0 - market_price) / market_price

    raw_kelly = (b * p - q) / b

    if raw_kelly <= 0:
        return 0.0

    # Apply fractional Kelly
    fractional = raw_kelly * cfg.kelly_fraction

    # Cap at max_position_fraction
    capped = min(fractional, cfg.max_position_fraction)

    bet_usdc = capped * bankroll

    logger.debug(
        "Kelly: true_prob=%.3f market=%.3f b=%.3f raw_f=%.4f frac_f=%.4f "
        "capped_f=%.4f bet=$%.2f",
        true_prob, market_price, b, raw_kelly, fractional, capped, bet_usdc,
    )
    return bet_usdc


def kelly_size_no(
    true_prob_yes: float,
    market_price_yes: float,
    bankroll_usdc: float | None = None,
) -> float:
    """
    Kelly size for buying NO (equivalent to betting against YES).

    true_prob_yes:    Estimated probability that YES resolves.
    market_price_yes: Current YES price.
    Returns bet size in USDC for buying NO.
    """
    # NO price = 1 - YES price; true prob of NO = 1 - true_prob_yes
    return kelly_size(
        true_prob=1.0 - true_prob_yes,
        market_price=1.0 - market_price_yes,
        bankroll_usdc=bankroll_usdc,
    )
