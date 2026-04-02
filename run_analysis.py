"""
run_analysis.py — standalone script to execute the whale-identification pipeline.

Run:  python run_analysis.py

This will:
  1. Fetch the top 10 000 wallets from the Polymarket leaderboard.
  2. Iteratively reduce to the top 15 most profitable traders.
  3. Save results to data/whales.json.

Must be completed before running main.py (the copy trader).
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass  # uvloop not available (Windows); use default

from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

from config import cfg
from whale_analyzer.fetcher import WhaleFetcher
from whale_analyzer.reducer import WhaleReducer

console = Console()

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def _print_results_table(scores: list) -> None:
    table = Table(title=f"Top {len(scores)} Polymarket Whales", show_lines=True)
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Address", style="green", width=42)
    table.add_column("Score", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("PnL $", justify="right", style="bold green")
    table.add_column("Trades", justify="right")
    table.add_column("Profit Factor", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Avg Edge", justify="right")

    for s in scores:
        table.add_row(
            str(s.rank),
            s.address,
            f"{s.composite_score:.4f}",
            f"{s.win_rate * 100:.1f}%",
            f"${s.total_pnl_usdc:,.2f}",
            str(s.trade_count),
            f"{s.profit_factor:.2f}",
            f"{s.sharpe_ratio:.2f}",
            f"{s.avg_edge:.3f}",
        )

    console.print(table)
    console.print(f"\n[bold]Results saved to:[/] {cfg.whale_data_path}\n")


async def _main() -> None:
    start = time.perf_counter()
    console.rule("[bold blue]PolyBot — Whale Analysis Pipeline")

    reducer = WhaleReducer()

    async with WhaleFetcher() as fetcher:
        top_whales = await reducer.run(fetcher)

    elapsed = time.perf_counter() - start
    console.rule(f"[bold green]Analysis complete in {elapsed:.1f}s")
    _print_results_table(top_whales)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/]")
        sys.exit(0)
    except Exception:
        console.print_exception()
        sys.exit(1)
