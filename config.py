"""
Central configuration — loaded once at startup from environment / .env file.
All modules import from here; no raw os.getenv calls elsewhere.
"""
from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # ── Credentials ──────────────────────────────────────────────────────────
    poly_private_key: str = Field(default="", env="POLY_PRIVATE_KEY")
    poly_api_key: str = Field(default="", env="POLY_API_KEY")
    poly_api_secret: str = Field(default="", env="POLY_API_SECRET")
    poly_api_passphrase: str = Field(default="", env="POLY_API_PASSPHRASE")
    polygon_rpc: str = Field(
        default="https://polygon-rpc.com", env="POLYGON_RPC"
    )

    # ── API endpoints (no trailing slash) ────────────────────────────────────
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    # data-api hosts the leaderboard and supplementary trade data
    data_url: str = "https://data-api.polymarket.com"
    # leaderboard-api.polymarket.com refuses connections; data-api hosts the leaderboard
    leaderboard_url: str = "https://data-api.polymarket.com"

    # ── Whale polling ─────────────────────────────────────────────────────────
    # Polymarket's WebSocket user channel is authenticated and only surfaces
    # the OWNER's own trades — you cannot subscribe to arbitrary wallet addresses.
    # The only reliable way to detect other wallets' trades in near-real-time is
    # to poll /data/trades?maker_address=<whale>&after=<ts>.
    # With 15 whales and a 2s interval this sends ~7.5 req/s — within rate limits.
    whale_poll_interval_sec: float = Field(default=2.0, env="WHALE_POLL_INTERVAL_SEC")

    # ── Trading params ────────────────────────────────────────────────────────
    bankroll_usdc: float = Field(default=1000.0, env="BANKROLL_USDC")
    kelly_fraction: float = Field(default=0.25, env="KELLY_FRACTION")
    max_position_fraction: float = Field(
        default=0.10, env="MAX_POSITION_FRACTION"
    )
    max_bankroll_usdc: float = Field(default=1000.0, env="MAX_BANKROLL_USDC")
    min_edge_threshold: float = Field(default=0.03, env="MIN_EDGE_THRESHOLD")
    min_prob: float = Field(default=0.05, env="MIN_PROB")
    max_prob: float = Field(default=0.95, env="MAX_PROB")

    # ── Analysis params ───────────────────────────────────────────────────────
    initial_wallet_count: int = Field(default=10000, env="INITIAL_WALLET_COUNT")
    min_trade_count: int = Field(default=50, env="MIN_TRADE_COUNT")
    top_whale_count: int = Field(default=15, env="TOP_WHALE_COUNT")
    whale_data_path: str = Field(default="data/whales.json", env="WHALE_DATA_PATH")

    # Reduction ladder: each step keeps this fraction of remaining wallets
    # 10000 → 5000 → 2500 → 1250 → 750 → 375 → 188 → 94 → 47 → 23 → 15
    reduction_ladder: list[int] = [
        10000, 5000, 2500, 1250, 750, 375, 188, 94, 47, 23, 15
    ]

    # ── HTTP tuning ───────────────────────────────────────────────────────────
    # Connections per host in the aiohttp pool
    connector_limit_per_host: int = 20
    connector_limit: int = 100
    request_timeout_sec: float = 10.0
    max_retries: int = 3

    # ── WebSocket tuning ──────────────────────────────────────────────────────
    ws_ping_interval: float = 20.0   # seconds
    ws_ping_timeout: float = 10.0
    ws_reconnect_delay: float = 0.5  # initial back-off (doubles on failure)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # Max requests per second to Polymarket APIs during whale analysis
    rate_limit_rps: float = Field(default=10.0, env="RATE_LIMIT_RPS")

    # ── Position monitoring ───────────────────────────────────────────────────
    # How often to poll open positions for market resolution (seconds)
    position_poll_interval_sec: float = Field(default=300.0, env="POSITION_POLL_INTERVAL_SEC")
    # How long to wait for an order to fill before cancelling (seconds)
    order_fill_timeout_sec: float = Field(default=30.0, env="ORDER_FILL_TIMEOUT_SEC")

    # ── Whale data freshness ──────────────────────────────────────────────────
    # Trigger re-analysis if whales.json is older than this many hours
    whale_data_max_age_hours: float = Field(default=24.0, env="WHALE_DATA_MAX_AGE_HOURS")

    # ── Misc ──────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    chain_id: int = 137  # Polygon

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton for convenience imports
cfg = get_settings()
