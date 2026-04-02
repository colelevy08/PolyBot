# PolyBot

A high-speed Polymarket copy-trading bot that identifies the top 15 most profitable whale wallets through a rigorous multi-round analysis pipeline, then mirrors their trades in real time with edge verification and Kelly-sized position sizing.

---

## Architecture

```
PolyBot/
├── run_analysis.py          # Step 1: identify top 15 whale wallets
├── main.py                  # Step 2: run the live copy-trading engine
├── config.py                # Centralised settings (reads .env)
├── whale_analyzer/
│   ├── fetcher.py           # Async API client (aiohttp + orjson)
│   ├── scorer.py            # Multi-factor wallet scoring
│   ├── reducer.py           # 10 000 → 15 iterative reduction
│   └── models.py            # Dataclasses (slots for speed)
├── copy_trader/
│   ├── ws_listener.py       # Persistent WebSocket watcher
│   ├── executor.py          # Signal → edge check → order fire
│   └── order_manager.py     # In-memory position book + P&L
└── edge_scanner/
    ├── scanner.py           # Live orderbook edge calculation
    └── kelly.py             # Kelly criterion sizing
```

---

## Step 1 — Whale Analysis

The analysis pipeline drills through the entire Polymarket leaderboard using a reduction ladder:

```
10 000 → 5 000 → 2 500 → 1 250 → 750 → 375 → 188 → 94 → 47 → 23 → 15
```

At every round, **fresh data is fetched** (no cache). Each wallet is scored on:

| Factor | Weight |
|--------|--------|
| Win rate | 20% |
| Profit factor (gross profit / gross loss) | 20% |
| Expected value per trade (avg USDC P&L) | 20% |
| Sharpe ratio | 15% |
| Recency (last 90 days activity) | 10% |
| Average entry edge | 10% |
| Trade count (log-scaled; statistical confidence) | 5% |

Results are saved to `data/whales.json`.

---

## Step 2 — Copy Trading Engine

1. **WebSocket listener** — single persistent connection to `wss://ws-subscriptions-clob.polymarket.com/ws/`, subscribed to all 15 whale user-channels simultaneously.
2. **Edge scanner** — on every whale trade event, fetches the **live orderbook** (no cache), estimates true probability from whale signal + mid-price + depth imbalance, and calculates the edge we'd capture at the current ask.
3. **Kelly sizing** — fractional Kelly (default ¼ Kelly) with a max-position cap.
4. **Order executor** — places a GTC limit order via the CLOB API using `py-clob-client`. Total latency from WebSocket receipt to order placed is logged for every trade.

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/colelevy08/PolyBot.git
cd PolyBot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — add your POLY_PRIVATE_KEY, API keys, bankroll, etc.

# 3. Run whale analysis (takes 5–30 min depending on API rate limits)
python run_analysis.py

# 4. Run the copy trader
python main.py
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `POLY_PRIVATE_KEY` | — | Your Polymarket EOA private key |
| `POLY_API_KEY` | — | CLOB API key |
| `BANKROLL_USDC` | 1000 | Total capital for Kelly sizing |
| `KELLY_FRACTION` | 0.25 | Fractional Kelly multiplier |
| `MAX_POSITION_FRACTION` | 0.10 | Max single bet as % of bankroll |
| `MIN_EDGE_THRESHOLD` | 0.03 | Min edge (3 cents) required to copy |
| `INITIAL_WALLET_COUNT` | 10000 | Wallets in first analysis round |
| `TOP_WHALE_COUNT` | 15 | Final number of whales to track |
| `LOG_LEVEL` | INFO | Logging verbosity |

---

## Performance Notes

- **uvloop** is used automatically on macOS/Linux for a faster asyncio event loop.
- **orjson** handles all JSON parsing (2–5× faster than stdlib).
- The aiohttp session uses a **persistent connection pool** (no per-request TCP handshakes).
- WebSocket compression is **disabled** to minimise latency over bandwidth.
- Edge scan is a single HTTP request; total copy latency target is **< 100ms**.

---

## Risk Warning

Trading prediction markets involves significant financial risk. This bot is provided for educational purposes. Always test with small amounts before deploying real capital. The edge scanner is a heuristic, not a guarantee of profit.
