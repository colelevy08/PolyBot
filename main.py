"""
main.py — PolyBot copy-trading engine.

Run:  python main.py

Prerequisites:
  - python run_analysis.py must have been run first (produces data/whales.json)
  - .env must be populated with CLOB credentials

Pipeline:
  1. Load top whale addresses from data/whales.json.
  2. Initialise aiohttp session, EdgeScanner, TradeExecutor, OrderManager.
  3. WhaleWatcher opens a persistent WebSocket to Polymarket CLOB.
  4. On each whale trade → EdgeScanner → Kelly size → CLOB order.
  5. Periodic stats logged every 60 seconds.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("uvloop enabled — faster event loop active.")
except ImportError:
    pass

from rich.logging import RichHandler
from rich.console import Console

from config import cfg
from whale_analyzer.fetcher import WhaleFetcher
from whale_analyzer.reducer import WhaleReducer
from copy_trader.ws_listener import WhaleWatcher
from copy_trader.executor import TradeExecutor
from copy_trader.order_manager import OrderManager

console = Console()

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


async def _stats_loop(order_manager: OrderManager, interval: float = 60.0) -> None:
    """Logs position summary every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        summary = order_manager.summary()
        logger.info(
            "[STATS] open=%d closed=%d pnl=$%.2f win_rate=%.1f%%",
            summary["open_positions"],
            summary["closed_positions"],
            summary["total_pnl_usdc"],
            summary["win_rate"] * 100,
        )


async def _main() -> None:
    console.rule("[bold blue]PolyBot — Copy Trading Engine")

    # ── Load whale addresses ──────────────────────────────────────────────────
    try:
        whale_addresses = WhaleReducer.load_whale_addresses()
    except FileNotFoundError as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        sys.exit(1)

    console.print(f"[green]Tracking {len(whale_addresses)} whale wallets.[/]")

    # ── Initialise components ─────────────────────────────────────────────────
    order_manager = OrderManager()

    async with WhaleFetcher() as fetcher:
        executor = TradeExecutor(
            fetcher=fetcher,
            order_manager=order_manager,
            bankroll_usdc=cfg.bankroll_usdc,
        )
        # Give executor a reference to the shared HTTP session
        executor.attach_session(fetcher._session)

        watcher = WhaleWatcher(
            whale_addresses=whale_addresses,
            on_signal=executor.handle_signal,
        )

        # ── Graceful shutdown ─────────────────────────────────────────────────
        loop = asyncio.get_running_loop()

        def _shutdown() -> None:
            console.print("\n[yellow]Shutdown signal received.[/]")
            asyncio.create_task(watcher.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

        # ── Run ───────────────────────────────────────────────────────────────
        console.rule("[bold green]Bot is LIVE — listening for whale trades")
        await asyncio.gather(
            watcher.run(),
            _stats_loop(order_manager),
        )


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")
        sys.exit(0)
    except Exception:
        console.print_exception()
        sys.exit(1)
