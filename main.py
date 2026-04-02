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


async def _stats_loop(
    order_manager: OrderManager,
    executor: TradeExecutor,
    interval: float = 60.0,
) -> None:
    """
    Logs position summary and refreshes live bankroll every `interval` seconds.
    Exits cleanly on cancel.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            bankroll = await executor.sync_bankroll()
            summary = order_manager.summary()
            logger.info(
                "[STATS] bankroll=$%.2f  open=%d  closed=%d  pnl=$%.2f  win_rate=%.1f%%",
                bankroll,
                summary["open_positions"],
                summary["closed_positions"],
                summary["total_pnl_usdc"],
                summary["win_rate"] * 100,
            )
    except asyncio.CancelledError:
        pass


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

        # Fetch live balance and set bankroll before the bot starts trading
        opening_balance = await executor.sync_bankroll()
        console.print(f"[green]Polymarket balance: ${opening_balance:,.2f} USDC[/]")

        watcher = WhaleWatcher(
            whale_addresses=whale_addresses,
            on_signal=executor.handle_signal,
        )

        # ── Graceful shutdown ─────────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        main_task: asyncio.Task | None = None

        def _shutdown() -> None:
            console.print("\n[yellow]Shutdown signal received.[/]")
            if main_task is not None:
                # Cancelling the gather task propagates CancelledError to both
                # watcher.run() and _stats_loop(), unwinding them cleanly so
                # WhaleFetcher.__aexit__ can close the HTTP session.
                main_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

        # ── Run ───────────────────────────────────────────────────────────────
        console.rule("[bold green]Bot is LIVE — listening for whale trades")
        main_task = asyncio.ensure_future(
            asyncio.gather(
                watcher.run(),
                _stats_loop(order_manager, executor),
                return_exceptions=True,
            )
        )
        try:
            await main_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")
        sys.exit(0)
    except Exception:
        console.print_exception()
        sys.exit(1)
