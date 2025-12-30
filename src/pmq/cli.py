"""Polymarket Quant Lab CLI.

Commands:
    pmq sync        - Fetch and cache market data from Gamma API
    pmq scan        - Scan for arbitrage and stat-arb signals
    pmq paper run   - Run paper trading strategy loop
    pmq report      - Generate PnL and trading report
"""

import time
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from pmq import __version__
from pmq.config import get_settings
from pmq.gamma_client import GammaClient
from pmq.logging import setup_logging
from pmq.storage import DAO
from pmq.strategies import ArbitrageScanner, PaperLedger, StatArbScanner
from pmq.strategies.paper import SafetyError

app = typer.Typer(
    name="pmq",
    help="Polymarket Quant Lab - Market data and paper trading platform",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        rprint(f"Polymarket Quant Lab v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Polymarket Quant Lab - Market data and paper trading."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)


# =============================================================================
# Sync Command
# =============================================================================


@app.command()
def sync(
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Number of markets to fetch"),
    ] = 200,
    clear_cache: Annotated[
        bool,
        typer.Option("--clear-cache", help="Clear cache before syncing"),
    ] = False,
) -> None:
    """Fetch and cache market data from Gamma API.

    This command fetches market metadata and prices from the Polymarket
    Gamma API and stores them in the local SQLite database.

    Example:
        pmq sync --limit 100
    """
    with console.status("[bold green]Syncing market data..."):
        client = GammaClient()

        if clear_cache:
            cleared = client.clear_cache()
            console.print(f"[yellow]Cleared {cleared} cache files[/yellow]")

        try:
            # Fetch markets
            console.print(f"[cyan]Fetching up to {limit} markets from Gamma API...[/cyan]")
            markets = client.list_markets(limit=limit)

            if not markets:
                console.print("[red]No markets returned from API[/red]")
                raise typer.Exit(1)

            # Store in database
            dao = DAO()
            count = dao.upsert_markets(markets)

            console.print(f"[green]✓ Synced {count} markets[/green]")

            # Show summary
            active_count = sum(1 for m in markets if m.active and not m.closed)
            total_liquidity = sum(m.liquidity for m in markets)
            total_volume = sum(m.volume24hr for m in markets)

            table = Table(title="Sync Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Markets", str(count))
            table.add_row("Active Markets", str(active_count))
            table.add_row("Total Liquidity", f"${total_liquidity:,.0f}")
            table.add_row("24h Volume", f"${total_volume:,.0f}")

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error syncing: {e}[/red]")
            raise typer.Exit(1) from e
        finally:
            client.close()


# =============================================================================
# Scan Command
# =============================================================================


@app.command()
def scan(
    top: Annotated[
        int,
        typer.Option("--top", "-n", help="Number of top signals to show"),
    ] = 20,
    arb_threshold: Annotated[
        float,
        typer.Option("--arb-threshold", help="Arbitrage threshold (YES+NO < threshold)"),
    ] = 0.99,
    min_liquidity: Annotated[
        float,
        typer.Option("--min-liquidity", help="Minimum liquidity in USD"),
    ] = 100.0,
    from_api: Annotated[
        bool,
        typer.Option("--from-api", help="Fetch fresh data from API instead of database"),
    ] = False,
) -> None:
    """Scan markets for arbitrage and stat-arb signals.

    Analyzes cached market data for trading opportunities:
    - Arbitrage: YES + NO price < threshold
    - Stat-arb: Spread divergence between correlated pairs

    Example:
        pmq scan --top 10 --arb-threshold 0.98
    """
    with console.status("[bold green]Scanning for signals..."):
        dao = DAO()

        if from_api:
            # Fetch fresh data
            console.print("[cyan]Fetching fresh market data...[/cyan]")
            with GammaClient() as client:
                markets = client.list_markets(limit=500)

            arb_scanner = ArbitrageScanner()
            arb_signals = arb_scanner.scan_markets(markets, top_n=top)
        else:
            # Use cached data
            markets_data = dao.get_active_markets(limit=500)

            if not markets_data:
                console.print("[yellow]No cached markets found. Run 'pmq sync' first.[/yellow]")
                raise typer.Exit(1)

            console.print(f"[cyan]Scanning {len(markets_data)} cached markets...[/cyan]")

            arb_scanner = ArbitrageScanner()
            arb_scanner._config.threshold = arb_threshold
            arb_scanner._config.min_liquidity = min_liquidity
            arb_signals = arb_scanner.scan_from_db(markets_data, top_n=top)

        # Display arbitrage signals
        if arb_signals:
            table = Table(title=f"Top {len(arb_signals)} Arbitrage Signals")
            table.add_column("#", style="dim")
            table.add_column("Market", style="cyan", max_width=50)
            table.add_column("YES", justify="right", style="green")
            table.add_column("NO", justify="right", style="green")
            table.add_column("Sum", justify="right", style="yellow")
            table.add_column("Profit %", justify="right", style="bold green")
            table.add_column("Liquidity", justify="right")

            for i, sig in enumerate(arb_signals, 1):
                question = (
                    sig.market_question[:47] + "..."
                    if len(sig.market_question) > 50
                    else sig.market_question
                )
                table.add_row(
                    str(i),
                    question,
                    f"{sig.yes_price:.4f}",
                    f"{sig.no_price:.4f}",
                    f"{sig.combined_price:.4f}",
                    f"{sig.profit_potential * 100:.2f}%",
                    f"${sig.liquidity:,.0f}",
                )

            console.print(table)

            # Save signals to database
            for sig in arb_signals:
                dao.save_arb_signal(sig)
            console.print(f"[green]✓ Saved {len(arb_signals)} signals to database[/green]")
        else:
            console.print("[yellow]No arbitrage signals found[/yellow]")

        # Scan stat-arb (if pairs configured)
        statarb_scanner = StatArbScanner()
        if statarb_scanner.pairs:
            console.print(f"\n[cyan]Scanning {len(statarb_scanner.pairs)} stat-arb pairs...[/cyan]")
            if from_api:
                statarb_signals = statarb_scanner.scan_pairs(markets)
            else:
                statarb_signals = statarb_scanner.scan_from_db(markets_data)

            if statarb_signals:
                table = Table(title="Stat-Arb Signals")
                table.add_column("Pair", style="cyan")
                table.add_column("Spread", justify="right", style="yellow")
                table.add_column("Direction", style="green")

                for statarb_sig in statarb_signals[:top]:
                    table.add_row(
                        f"{statarb_sig.market_a_id[:8]}.. / {statarb_sig.market_b_id[:8]}..",
                        f"{statarb_sig.spread:.4f}",
                        statarb_sig.direction,
                    )

                console.print(table)
        else:
            console.print(
                "\n[dim]No stat-arb pairs configured. " "Add pairs to config/pairs.yml[/dim]"
            )


# =============================================================================
# Paper Trading Commands
# =============================================================================

paper_app = typer.Typer(help="Paper trading commands")
app.add_typer(paper_app, name="paper")


@paper_app.command("run")
def paper_run(
    minutes: Annotated[
        int,
        typer.Option("--minutes", "-m", help="Duration to run in minutes"),
    ] = 5,
    quantity: Annotated[
        float,
        typer.Option("--quantity", "-q", help="Trade quantity per signal"),
    ] = 10.0,
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Scan interval in seconds"),
    ] = 30,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Detect signals but don't execute trades"),
    ] = False,
) -> None:
    """Run paper trading strategy loop.

    Continuously scans for arbitrage signals and executes paper trades.
    All trades are simulated - no real orders are placed.

    Example:
        pmq paper run --minutes 10 --quantity 5
    """
    settings = get_settings()

    if settings.safety.kill_switch:
        console.print("[red]Kill switch is active. Trading halted.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Starting paper trading loop[/bold green]\n"
        f"Duration: {minutes} minutes\n"
        f"Quantity: {quantity} per signal\n"
        f"Interval: {interval} seconds\n"
        f"Dry run: {dry_run}\n"
    )

    dao = DAO()
    ledger = PaperLedger(dao=dao)
    arb_scanner = ArbitrageScanner()
    client = GammaClient()

    start_time = time.time()
    end_time = start_time + (minutes * 60)
    trades_executed = 0
    signals_detected = 0

    try:
        while time.time() < end_time:
            remaining = int((end_time - time.time()) / 60)
            console.print(f"\n[cyan]Scanning... ({remaining}m remaining)[/cyan]")

            # Fetch fresh data
            markets = client.list_markets(limit=200)
            dao.upsert_markets(markets)

            # Scan for signals
            signals = arb_scanner.scan_markets(markets, top_n=5)
            signals_detected += len(signals)

            if signals:
                console.print(f"[green]Found {len(signals)} signals[/green]")

                for signal in signals:
                    if dry_run:
                        console.print(
                            f"[dim]DRY RUN: Would trade {signal.market_id[:16]}... "
                            f"(profit: {signal.profit_potential*100:.2f}%)[/dim]"
                        )
                    else:
                        try:
                            yes_trade, no_trade = ledger.execute_arb_trade(
                                signal, quantity=quantity
                            )
                            trades_executed += 2
                            console.print(
                                f"[green]✓ Executed arb trade on {signal.market_id[:16]}...[/green]"
                            )
                        except SafetyError as e:
                            console.print(f"[yellow]Safety blocked: {e}[/yellow]")
                        except Exception as e:
                            console.print(f"[red]Trade error: {e}[/red]")
            else:
                console.print("[dim]No signals found[/dim]")

            # Wait for next interval
            if time.time() < end_time:
                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        client.close()

    # Print summary
    console.print("\n[bold]Paper Trading Summary[/bold]")
    console.print(f"Duration: {int((time.time() - start_time) / 60)} minutes")
    console.print(f"Signals detected: {signals_detected}")
    console.print(f"Trades executed: {trades_executed}")

    # Show final stats
    stats = ledger.get_stats()
    console.print(f"\n[cyan]Total trades: {stats['total_trades']}[/cyan]")
    console.print(f"[cyan]Total notional: ${stats['total_notional']:,.2f}[/cyan]")
    console.print(f"[cyan]Open positions: {stats['open_positions']}[/cyan]")


@paper_app.command("positions")
def paper_positions() -> None:
    """Show current paper positions."""
    ledger = PaperLedger()
    positions = ledger.get_all_positions()

    if not positions:
        console.print("[dim]No open positions[/dim]")
        return

    table = Table(title="Paper Positions")
    table.add_column("Market", style="cyan", max_width=40)
    table.add_column("YES Qty", justify="right")
    table.add_column("NO Qty", justify="right")
    table.add_column("Avg YES", justify="right")
    table.add_column("Avg NO", justify="right")
    table.add_column("Realized PnL", justify="right", style="green")

    for pos in positions:
        question = (
            pos.market_question[:37] + "..."
            if len(pos.market_question) > 40
            else pos.market_question
        )
        table.add_row(
            question,
            f"{pos.yes_quantity:.2f}",
            f"{pos.no_quantity:.2f}",
            f"{pos.avg_price_yes:.4f}",
            f"{pos.avg_price_no:.4f}",
            f"${pos.realized_pnl:.2f}",
        )

    console.print(table)


@paper_app.command("trades")
def paper_trades(
    limit: Annotated[int, typer.Option("--limit", "-l")] = 20,
    strategy: Annotated[str | None, typer.Option("--strategy", "-s")] = None,
) -> None:
    """Show recent paper trades."""
    ledger = PaperLedger()
    trades = ledger.get_trades(strategy=strategy, limit=limit)

    if not trades:
        console.print("[dim]No trades found[/dim]")
        return

    table = Table(title=f"Recent Paper Trades (limit={limit})")
    table.add_column("ID", style="dim")
    table.add_column("Strategy")
    table.add_column("Market", max_width=30)
    table.add_column("Side")
    table.add_column("Outcome")
    table.add_column("Price", justify="right")
    table.add_column("Qty", justify="right")
    table.add_column("Notional", justify="right")

    for trade in trades:
        question = (
            trade.market_question[:27] + "..."
            if len(trade.market_question) > 30
            else trade.market_question
        )
        table.add_row(
            str(trade.id),
            trade.strategy,
            question,
            trade.side.value,
            trade.outcome.value,
            f"{trade.price:.4f}",
            f"{trade.quantity:.1f}",
            f"${trade.notional:.2f}",
        )

    console.print(table)


# =============================================================================
# Report Command
# =============================================================================


@app.command()
def report() -> None:
    """Generate summary report of paper trading performance.

    Shows PnL breakdown, position summary, and trading statistics.

    Example:
        pmq report
    """
    ledger = PaperLedger()
    dao = DAO()

    # Get trading stats
    stats = ledger.get_stats()

    # Get PnL
    markets_data = dao.get_active_markets(limit=500)
    markets_lookup = {m["id"]: m for m in markets_data}
    pnl = ledger.calculate_pnl(markets_data=markets_lookup)

    # Main stats table
    console.print("\n[bold]Polymarket Quant Lab Report[/bold]\n")

    stats_table = Table(title="Trading Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Trades", str(stats["total_trades"]))
    stats_table.add_row("Total Notional", f"${stats['total_notional']:,.2f}")
    stats_table.add_row("Unique Markets Traded", str(stats["unique_markets"]))
    stats_table.add_row("Open Positions", str(stats["open_positions"]))
    stats_table.add_row("Total Signals Detected", str(stats["total_signals"]))

    console.print(stats_table)

    # PnL table
    pnl_table = Table(title="PnL Summary")
    pnl_table.add_column("Metric", style="cyan")
    pnl_table.add_column("Value", justify="right")

    pnl_color = "green" if pnl["total_pnl"] >= 0 else "red"
    pnl_table.add_row("Realized PnL", f"${pnl['total_realized_pnl']:,.2f}")
    pnl_table.add_row("Unrealized PnL", f"${pnl['total_unrealized_pnl']:,.2f}")
    pnl_table.add_row(
        "Total PnL",
        f"[{pnl_color}]${pnl['total_pnl']:,.2f}[/{pnl_color}]",
    )

    console.print(pnl_table)

    # Position details
    if pnl["positions"]:
        pos_table = Table(title="Position Details")
        pos_table.add_column("Market", style="cyan", max_width=50)
        pos_table.add_column("YES", justify="right")
        pos_table.add_column("NO", justify="right")
        pos_table.add_column("PnL", justify="right")

        for pos in pnl["positions"][:10]:  # Top 10 positions
            color = "green" if pos["total_pnl"] >= 0 else "red"
            pos_table.add_row(
                pos["market_question"],
                f"{pos['yes_qty']:.1f}",
                f"{pos['no_qty']:.1f}",
                f"[{color}]${pos['total_pnl']:.2f}[/{color}]",
            )

        console.print(pos_table)

    # Recent signals
    recent_signals = dao.get_recent_signals(limit=5)
    if recent_signals:
        sig_table = Table(title="Recent Signals")
        sig_table.add_column("Type", style="cyan")
        sig_table.add_column("Market(s)", max_width=40)
        sig_table.add_column("Profit Potential", justify="right")
        sig_table.add_column("Time")

        for sig in recent_signals:
            market_ids = ", ".join(m[:12] + "..." for m in sig["market_ids"])
            sig_table.add_row(
                sig["type"],
                market_ids,
                f"{sig['profit_potential']*100:.2f}%",
                sig["created_at"][:19],
            )

        console.print(sig_table)

    console.print("\n[dim]Note: All trades are simulated. No real money involved.[/dim]")


if __name__ == "__main__":
    app()
