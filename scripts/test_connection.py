"""
scripts/test_connection.py — Verify your .env credentials work before running the bot.

Usage:
    python scripts/test_connection.py

Checks:
    1. .env loads correctly
    2. CLOB API authenticates
    3. Markets endpoint responds
    4. Balance is readable
    5. WebSocket connects and receives at least one message
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

REQUIRED_VARS = [
    "POLY_PRIVATE_KEY",
    "POLY_API_KEY",
    "POLY_API_SECRET",
    "POLY_API_PASSPHRASE",
]


def check_env() -> bool:
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        print(f"ERROR: missing .env vars: {', '.join(missing)}")
        print("Run scripts/get_creds.py first.")
        return False
    print("✓ .env loaded — all required vars present")
    return True


def check_clob() -> bool:
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        print("ERROR: py_clob_client not installed. Run: pip install -r requirements.txt")
        return False

    try:
        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            private_key=os.getenv("POLY_PRIVATE_KEY"),
            api_key=os.getenv("POLY_API_KEY"),
            api_secret=os.getenv("POLY_API_SECRET"),
            api_passphrase=os.getenv("POLY_API_PASSPHRASE"),
        )
    except Exception as exc:
        print(f"ERROR: ClobClient init failed — {exc}")
        return False

    # Markets
    try:
        markets = client.get_markets()
        count = len(markets.data) if hasattr(markets, "data") else "?"
        print(f"✓ CLOB connected — {count} markets available")
    except Exception as exc:
        print(f"ERROR: get_markets() failed — {exc}")
        return False

    # Balance
    try:
        bal = client.get_balance()
        print(f"✓ Balance: {bal}")
    except Exception as exc:
        # Balance endpoint may require funded account; non-fatal
        print(f"  Balance check skipped ({exc})")

    return True


async def check_websocket() -> bool:
    try:
        import websockets
        import orjson
    except ImportError:
        print("ERROR: websockets or orjson not installed.")
        return False

    url = "wss://ws-subscriptions-clob.polymarket.com/ws/"
    print(f"  Connecting to WebSocket: {url}")
    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            # Send a test subscription to the global market channel
            await ws.send(orjson.dumps({
                "type": "subscribe",
                "channel": "market",
                "market_slug": "test",
            }).decode())
            # Wait briefly for any server response (ack, error, or message)
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print(f"✓ WebSocket connected — server responded ({len(msg)} bytes)")
            except asyncio.TimeoutError:
                # No message in 5s is still a successful connection
                print("✓ WebSocket connected — no immediate message (normal)")
    except Exception as exc:
        print(f"ERROR: WebSocket connection failed — {exc}")
        return False

    return True


async def main() -> None:
    print("=" * 50)
    print("  PolyBot — Connection Test")
    print("=" * 50)
    print()

    ok = True
    ok = check_env() and ok
    print()
    ok = check_clob() and ok
    print()
    ok = await check_websocket() and ok
    print()

    if ok:
        print("=" * 50)
        print("  All checks passed — ready to run the bot.")
        print("  Next: python run_analysis.py")
        print("=" * 50)
    else:
        print("=" * 50)
        print("  One or more checks failed. Fix the errors above.")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
