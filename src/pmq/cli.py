"""Polymarket Quant Lab CLI.

Commands:
    pmq sync        - Fetch and cache market data from Gamma API
    pmq scan        - Scan for arbitrage and stat-arb signals
    pmq paper run   - Run paper trading strategy loop
    pmq report      - Generate PnL and trading report
    pmq serve       - Start local operator web console
    pmq run         - Run continuous operator loop (sync+scan)
    pmq export      - Export data to CSV files
"""

import csv
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from pmq import __version__
from pmq.config import get_settings
from pmq.gamma_client import GammaClient, GammaClientError
from pmq.logging import get_logger, setup_logging
from pmq.models import SignalType
from pmq.storage import DAO
from pmq.strategies import ArbitrageScanner, PaperLedger, StatArbScanner
from pmq.strategies.paper import SafetyError

logger = get_logger("cli")

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
    snapshot: Annotated[
        bool,
        typer.Option("--snapshot", help="Save immutable snapshot for backtesting"),
    ] = False,
) -> None:
    """Fetch and cache market data from Gamma API.

    This command fetches market metadata and prices from the Polymarket
    Gamma API and stores them in the local SQLite database.

    Use --snapshot to save time-series data for backtesting.

    Example:
        pmq sync --limit 100
        pmq sync --snapshot  # Save snapshot for backtesting
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

            # Save snapshot if requested
            if snapshot:
                snapshot_time = datetime.now(UTC).isoformat()
                snapshot_count = dao.save_snapshots_bulk(markets, snapshot_time)
                console.print(
                    f"[green]✓ Saved {snapshot_count} snapshots at {snapshot_time[:19]}[/green]"
                )

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
            if snapshot:
                table.add_row("Snapshots Saved", str(snapshot_count))

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
                "\n[dim]No stat-arb pairs configured. Add pairs to config/pairs.yml[/dim]"
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
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Strategy name (must be approved)"),
    ] = "arb",
    override_unsafe: Annotated[
        bool,
        typer.Option(
            "--override-unsafe",
            help="Bypass approval check (NOT RECOMMENDED for production)",
        ),
    ] = False,
) -> None:
    """Run paper trading strategy loop.

    Continuously scans for arbitrage signals and executes paper trades.
    All trades are simulated - no real orders are placed.

    Requires strategy approval unless --override-unsafe is used.

    Example:
        pmq paper run --strategy arb --minutes 10 --quantity 5
    """
    from pmq.governance import RiskGate

    settings = get_settings()

    if settings.safety.kill_switch:
        console.print("[red]Kill switch is active. Trading halted.[/red]")
        raise typer.Exit(1)

    dao = DAO()

    # Check strategy approval
    risk_gate = RiskGate(dao=dao)
    try:
        approval_status = risk_gate.enforce_approval(
            strategy_name=strategy,
            allow_override=override_unsafe,
        )
        if approval_status.approved:
            console.print(f"[green]✓ Strategy '{strategy}' is APPROVED[/green]")
        else:
            console.print(
                f"[yellow]⚠ Running with OVERRIDE - strategy '{strategy}' is NOT approved[/yellow]"
            )
    except PermissionError as e:
        console.print(f"[red]{e}[/red]")
        console.print(
            "\n[dim]Use 'pmq approve grant' to approve this strategy, "
            "or --override-unsafe to bypass (not recommended)[/dim]"
        )
        raise typer.Exit(1) from e

    console.print(
        f"\n[bold green]Starting paper trading loop[/bold green]\n"
        f"Strategy: {strategy}\n"
        f"Duration: {minutes} minutes\n"
        f"Quantity: {quantity} per signal\n"
        f"Interval: {interval} seconds\n"
        f"Dry run: {dry_run}\n"
    )

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
                            f"(profit: {signal.profit_potential * 100:.2f}%)[/dim]"
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
                f"{sig['profit_potential'] * 100:.2f}%",
                sig["created_at"][:19],
            )

        console.print(sig_table)

    console.print("\n[dim]Note: All trades are simulated. No real money involved.[/dim]")


# =============================================================================
# Serve Command (Web Console)
# =============================================================================


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to bind to"),
    ] = 8080,
) -> None:
    """Start local operator web console.

    Launches a read-only web dashboard at http://HOST:PORT
    showing signals, trades, positions, and statistics.

    Example:
        pmq serve --port 8080
    """
    import uvicorn

    from pmq.web import create_app

    console.print(f"[bold green]Starting operator console at http://{host}:{port}[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    web_app = create_app()
    uvicorn.run(web_app, host=host, port=port, log_level="info")


# =============================================================================
# Run Command (Operator Loop)
# =============================================================================


@app.command("run")
def operator_run(
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Sync/scan interval in seconds"),
    ] = 60,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Number of markets to fetch per cycle"),
    ] = 200,
    cycles: Annotated[
        int,
        typer.Option("--cycles", "-c", help="Number of cycles (0 = infinite)"),
    ] = 0,
    paper: Annotated[
        bool,
        typer.Option("--paper", help="Execute paper trades on signals"),
    ] = False,
    quantity: Annotated[
        float,
        typer.Option("--quantity", "-q", help="Trade quantity per signal (if --paper)"),
    ] = 10.0,
) -> None:
    """Run continuous operator loop (sync -> scan -> optional paper trade).

    This is the main operator command for unattended operation.
    It loops: fetch data, scan signals, optionally execute paper trades.

    Uses exponential backoff on API errors.
    Respects kill switch.

    Example:
        pmq run --interval 60 --limit 100
        pmq run --interval 30 --paper --quantity 5
    """
    settings = get_settings()

    if settings.safety.kill_switch:
        console.print("[red]Kill switch is active. Operator loop halted.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Starting operator loop[/bold green]\n"
        f"Interval: {interval}s\n"
        f"Limit: {limit} markets\n"
        f"Cycles: {'infinite' if cycles == 0 else cycles}\n"
        f"Paper trading: {paper}\n"
    )

    dao = DAO()
    arb_scanner = ArbitrageScanner()
    ledger = PaperLedger(dao=dao) if paper else None
    client = GammaClient()

    cycle_count = 0
    backoff = 1  # Exponential backoff starting point
    max_backoff = 300  # Max 5 minutes

    try:
        while cycles == 0 or cycle_count < cycles:
            cycle_count += 1
            now = datetime.now(UTC).isoformat()

            # Check kill switch each cycle
            settings = get_settings()
            if settings.safety.kill_switch:
                console.print("[red]Kill switch activated. Stopping.[/red]")
                break

            console.print(f"\n[cyan]Cycle {cycle_count} @ {now[:19]}[/cyan]")

            try:
                # Sync
                console.print(f"[dim]Syncing {limit} markets...[/dim]")
                markets = client.list_markets(limit=limit)
                dao.upsert_markets(markets)
                dao.set_runtime_state("last_sync_at", now)
                console.print(f"[green]✓ Synced {len(markets)} markets[/green]")

                # Scan
                signals = arb_scanner.scan_markets(markets, top_n=10)
                dao.set_runtime_state("last_scan_at", now)
                if signals:
                    console.print(f"[green]Found {len(signals)} signals[/green]")
                    for sig in signals:
                        dao.save_arb_signal(sig)
                else:
                    console.print("[dim]No signals found[/dim]")

                # Paper trade if enabled
                if paper and ledger and signals:
                    for sig in signals[:3]:  # Max 3 per cycle
                        try:
                            ledger.execute_arb_trade(sig, quantity=quantity)
                            console.print(f"[green]✓ Paper trade: {sig.market_id[:16]}...[/green]")
                        except SafetyError as e:
                            console.print(f"[yellow]Safety blocked: {e}[/yellow]")
                    dao.set_runtime_state("last_paper_at", now)

                # Reset backoff on success
                backoff = 1

            except GammaClientError as e:
                dao.set_runtime_state("last_error", f"{now}: {e}")
                console.print(f"[red]API error: {e}[/red]")
                console.print(f"[yellow]Backing off {backoff}s...[/yellow]")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            except Exception as e:
                dao.set_runtime_state("last_error", f"{now}: {e}")
                console.print(f"[red]Error: {e}[/red]")

            # Wait for next cycle
            if cycles == 0 or cycle_count < cycles:
                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        client.close()

    console.print(f"\n[bold]Completed {cycle_count} cycles[/bold]")


# =============================================================================
# Export Command
# =============================================================================


@app.command()
def export(
    data_type: Annotated[
        str,
        typer.Argument(help="Data to export: signals, trades, positions, all"),
    ] = "all",
    output_dir: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory"),
    ] = Path("exports"),
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum rows to export"),
    ] = 1000,
    signal_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Signal type filter (ARBITRAGE, STAT_ARB)"),
    ] = None,
) -> None:
    """Export data to CSV files.

    Exports signals, trades, and/or positions to CSV format.
    Files are timestamped for easy organization.

    Example:
        pmq export signals --out exports/ --limit 500
        pmq export all
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dao = DAO()

    exported_files: list[str] = []

    # Export signals
    if data_type in ("signals", "all"):
        type_filter = None
        if signal_type:
            try:
                type_filter = SignalType(signal_type.upper())
            except ValueError:
                console.print(f"[yellow]Unknown signal type: {signal_type}[/yellow]")

        signals = dao.get_signals_for_export(signal_type=type_filter, limit=limit)
        if signals:
            filename = output_dir / f"signals_{timestamp}.csv"
            _write_csv(filename, signals)
            exported_files.append(str(filename))
            console.print(f"[green]✓ Exported {len(signals)} signals to {filename}[/green]")
        else:
            console.print("[dim]No signals to export[/dim]")

    # Export trades
    if data_type in ("trades", "all"):
        trades = dao.get_trades_for_export(limit=limit)
        if trades:
            filename = output_dir / f"trades_{timestamp}.csv"
            _write_csv(filename, trades)
            exported_files.append(str(filename))
            console.print(f"[green]✓ Exported {len(trades)} trades to {filename}[/green]")
        else:
            console.print("[dim]No trades to export[/dim]")

    # Export positions
    if data_type in ("positions", "all"):
        positions = dao.get_positions_for_export()
        if positions:
            filename = output_dir / f"positions_{timestamp}.csv"
            _write_csv(filename, positions)
            exported_files.append(str(filename))
            console.print(f"[green]✓ Exported {len(positions)} positions to {filename}[/green]")
        else:
            console.print("[dim]No positions to export[/dim]")

    if exported_files:
        console.print(f"\n[bold]Exported {len(exported_files)} file(s) to {output_dir}/[/bold]")
    else:
        console.print("[yellow]No data exported[/yellow]")


def _write_csv(filepath: Path, data: list[dict[str, Any]]) -> None:
    """Write data to CSV file.

    Args:
        filepath: Output file path
        data: List of dicts to write
    """
    if not data:
        return

    # Flatten any nested lists/dicts to strings
    flat_data = []
    for row in data:
        flat_row = {}
        for k, v in row.items():
            if isinstance(v, list | dict):
                flat_row[k] = str(v)
            else:
                flat_row[k] = v
        flat_data.append(flat_row)

    fieldnames = list(flat_data[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)


# =============================================================================
# Backtest Commands
# =============================================================================

backtest_app = typer.Typer(help="Backtesting commands")
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Strategy to backtest: arb, statarb, observer"),
    ] = "arb",
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (YYYY-MM-DD)"),
    ] = "2024-01-01",
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (YYYY-MM-DD)"),
    ] = "2024-12-31",
    balance: Annotated[
        float,
        typer.Option("--balance", "-b", help="Initial balance"),
    ] = 10000.0,
    quantity: Annotated[
        float,
        typer.Option("--quantity", "-q", help="Trade quantity per signal"),
    ] = 10.0,
    pairs_config: Annotated[
        str | None,
        typer.Option("--pairs", help="Pairs config file for statarb"),
    ] = None,
) -> None:
    """Run a backtest on historical snapshot data.

    Requires snapshots to be collected first via `pmq sync --snapshot`.
    Results are deterministic - same inputs produce same outputs.

    Strategies:
        arb      - Arbitrage strategy (buys when YES+NO < threshold)
        statarb  - Statistical arbitrage on configured pairs
        observer - Validation baseline (observes without trading)

    Example:
        pmq backtest run --strategy arb --from 2024-01-01 --to 2024-01-07
        pmq backtest run --strategy statarb --pairs config/pairs.yml
        pmq backtest run --strategy observer --from 2024-12-01 --to 2024-12-31
    """
    from pmq.backtest import BacktestRunner

    dao = DAO()

    # Check for snapshots
    snapshot_count = dao.count_snapshots()
    if snapshot_count == 0:
        console.print(
            "[red]No snapshots found. Run 'pmq sync --snapshot' to collect data first.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Found {snapshot_count} total snapshots[/cyan]")

    runner = BacktestRunner(dao=dao, initial_balance=balance)

    with console.status(f"[bold green]Running {strategy} backtest..."):
        try:
            if strategy == "arb":
                run_id, metrics = runner.run_arb_backtest(
                    start_date=from_date,
                    end_date=to_date,
                    quantity=quantity,
                )
            elif strategy == "statarb":
                run_id, metrics = runner.run_statarb_backtest(
                    start_date=from_date,
                    end_date=to_date,
                    pairs_config=pairs_config,
                    quantity=quantity,
                )
            elif strategy == "observer":
                run_id, metrics = runner.run_observer_backtest(
                    start_date=from_date,
                    end_date=to_date,
                )
            else:
                console.print(
                    f"[red]Unknown strategy: {strategy}. Use: arb, statarb, observer[/red]"
                )
                raise typer.Exit(1)

            # Display results
            console.print("\n[bold green]Backtest Complete[/bold green]")
            console.print(f"Run ID: [cyan]{run_id}[/cyan]")

            table = Table(title="Backtest Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            pnl_color = "green" if metrics.total_pnl >= 0 else "red"
            table.add_row("Strategy", strategy)
            table.add_row("Period", f"{from_date} to {to_date}")
            table.add_row("Initial Balance", f"${balance:,.2f}")
            table.add_row("Final Balance", f"${metrics.final_balance:,.2f}")
            table.add_row("Total PnL", f"[{pnl_color}]${metrics.total_pnl:,.2f}[/{pnl_color}]")
            table.add_row("Max Drawdown", f"{metrics.max_drawdown:.2%}")
            table.add_row("Win Rate", f"{metrics.win_rate:.2%}")
            table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            table.add_row("Total Trades", str(metrics.total_trades))
            table.add_row("Trades/Day", f"{metrics.trades_per_day:.1f}")
            table.add_row("Total Notional", f"${metrics.total_notional:,.2f}")

            console.print(table)

            console.print(f"\n[dim]View details: pmq backtest report --run-id {run_id}[/dim]")

        except Exception as e:
            console.print(f"[red]Backtest failed: {e}[/red]")
            raise typer.Exit(1) from e


@backtest_app.command("report")
def backtest_report(
    run_id: Annotated[
        str,
        typer.Option("--run-id", "-r", help="Backtest run ID"),
    ],
    show_trades: Annotated[
        bool,
        typer.Option("--trades", help="Show individual trades"),
    ] = False,
) -> None:
    """Display detailed report for a backtest run.

    Example:
        pmq backtest report --run-id abc123
        pmq backtest report --run-id abc123 --trades
    """
    from pmq.backtest import BacktestRunner

    dao = DAO()
    runner = BacktestRunner(dao=dao)

    report = runner.get_run_report(run_id)

    if not report:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    run = report["run"]
    metrics = report["metrics"]

    # Run info
    console.print(f"\n[bold]Backtest Report: {run_id}[/bold]\n")

    info_table = Table(title="Run Info")
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value")

    info_table.add_row("Strategy", run["strategy"])
    info_table.add_row("Period", f"{run['start_date'][:10]} to {run['end_date'][:10]}")
    info_table.add_row("Status", run["status"])
    info_table.add_row("Initial Balance", f"${run['initial_balance']:,.2f}")
    if run.get("final_balance"):
        info_table.add_row("Final Balance", f"${run['final_balance']:,.2f}")
    info_table.add_row("Created", run["created_at"][:19])
    if run.get("completed_at"):
        info_table.add_row("Completed", run["completed_at"][:19])

    console.print(info_table)

    # Metrics
    if metrics:
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")

        pnl_color = "green" if metrics["total_pnl"] >= 0 else "red"
        metrics_table.add_row(
            "Total PnL", f"[{pnl_color}]${metrics['total_pnl']:,.2f}[/{pnl_color}]"
        )
        metrics_table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        metrics_table.add_row("Win Rate", f"{metrics['win_rate']:.2%}")
        metrics_table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        metrics_table.add_row("Total Trades", str(metrics["total_trades"]))
        metrics_table.add_row("Trades/Day", f"{metrics['trades_per_day']:.1f}")
        metrics_table.add_row("Capital Utilization", f"{metrics['capital_utilization']:.2%}")

        console.print(metrics_table)

    # Trades
    if show_trades and report["trades"]:
        trades_table = Table(title=f"Trades ({report['trade_count']} total)")
        trades_table.add_column("Time", style="dim")
        trades_table.add_column("Market", max_width=20)
        trades_table.add_column("Side")
        trades_table.add_column("Outcome")
        trades_table.add_column("Price", justify="right")
        trades_table.add_column("Qty", justify="right")
        trades_table.add_column("Notional", justify="right")

        for trade in report["trades"][:50]:  # Limit display
            trades_table.add_row(
                trade["trade_time"][:19],
                trade["market_id"][:20],
                trade["side"],
                trade["outcome"],
                f"{trade['price']:.4f}",
                f"{trade['quantity']:.1f}",
                f"${trade['notional']:.2f}",
            )

        console.print(trades_table)

        if report["trade_count"] > 50:
            console.print(f"[dim]... and {report['trade_count'] - 50} more trades[/dim]")

    console.print("\n[dim]Note: Backtest results are not guarantees of future performance.[/dim]")


@backtest_app.command("list")
def backtest_list(
    strategy: Annotated[
        str | None,
        typer.Option("--strategy", "-s", help="Filter by strategy"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum runs to show"),
    ] = 20,
) -> None:
    """List recent backtest runs.

    Example:
        pmq backtest list
        pmq backtest list --strategy arb
    """
    from pmq.backtest import BacktestRunner

    dao = DAO()
    runner = BacktestRunner(dao=dao)

    runs = runner.list_runs(strategy=strategy, limit=limit)

    if not runs:
        console.print("[dim]No backtest runs found[/dim]")
        return

    table = Table(title="Backtest Runs")
    table.add_column("Run ID", style="cyan", max_width=12)
    table.add_column("Strategy")
    table.add_column("Period")
    table.add_column("Status")
    table.add_column("PnL", justify="right")
    table.add_column("Created")

    for run in runs:
        run_id_short = run["id"][:8] + "..."
        period = f"{run['start_date'][:10]} to {run['end_date'][:10]}"
        pnl = ""
        if run.get("final_balance") and run.get("initial_balance"):
            pnl_val = run["final_balance"] - run["initial_balance"]
            pnl_color = "green" if pnl_val >= 0 else "red"
            pnl = f"[{pnl_color}]${pnl_val:,.2f}[/{pnl_color}]"

        table.add_row(
            run_id_short,
            run["strategy"],
            period,
            run["status"],
            pnl,
            run["created_at"][:19],
        )

    console.print(table)


@backtest_app.command("export")
def backtest_export(
    run_id: Annotated[
        str,
        typer.Option("--run-id", "-r", help="Backtest run ID"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory"),
    ] = Path("exports"),
    format_type: Annotated[
        str,
        typer.Option("--format", "-f", help="Export format: csv, json"),
    ] = "csv",
) -> None:
    """Export backtest results to file.

    Example:
        pmq backtest export --run-id abc123 --format csv --out exports/
    """
    import json as json_module

    from pmq.backtest import BacktestRunner

    dao = DAO()
    runner = BacktestRunner(dao=dao)

    report = runner.get_run_report(run_id)

    if not report:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if format_type == "csv":
        # Export trades
        trades = report["trades"]
        if trades:
            filename = output_dir / f"backtest_trades_{run_id[:8]}_{timestamp}.csv"
            _write_csv(filename, trades)
            console.print(f"[green]✓ Exported {len(trades)} trades to {filename}[/green]")

        # Export summary
        summary = [
            {
                "run_id": run_id,
                "strategy": report["run"]["strategy"],
                "start_date": report["run"]["start_date"],
                "end_date": report["run"]["end_date"],
                "status": report["run"]["status"],
                **(report["metrics"] or {}),
            }
        ]
        summary_file = output_dir / f"backtest_summary_{run_id[:8]}_{timestamp}.csv"
        _write_csv(summary_file, summary)
        console.print(f"[green]✓ Exported summary to {summary_file}[/green]")

    elif format_type == "json":
        filename = output_dir / f"backtest_{run_id[:8]}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json_module.dump(report, f, indent=2, default=str)
        console.print(f"[green]✓ Exported to {filename}[/green]")

    else:
        console.print(f"[red]Unknown format: {format_type}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Snapshots Commands (Phase 2.5)
# =============================================================================

snapshots_app = typer.Typer(help="Snapshot collection and quality commands")
app.add_typer(snapshots_app, name="snapshots")


@snapshots_app.command("run")
def snapshots_run(
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Snapshot interval in seconds"),
    ] = 60,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Number of markets to fetch per cycle"),
    ] = 200,
    duration_minutes: Annotated[
        int,
        typer.Option("--duration-minutes", "-d", help="Total duration in minutes (0 = infinite)"),
    ] = 60,
    with_orderbook: Annotated[
        bool,
        typer.Option(
            "--with-orderbook/--no-orderbook",
            help="Fetch order book data for microstructure (spread, depth)",
        ),
    ] = True,
    orderbook_source: Annotated[
        str,
        typer.Option(
            "--orderbook-source",
            help="Order book data source: rest (default), wss (WebSocket streaming)",
        ),
    ] = "rest",
    wss_staleness_seconds: Annotated[
        float,
        typer.Option(
            "--wss-staleness",
            help="WSS cache staleness threshold in seconds (fallback to REST if exceeded)",
        ),
    ] = 30.0,
) -> None:
    """Run automated snapshot collection loop.

    Collects market snapshots at regular intervals for backtesting.
    Does NOT execute any trades - data capture only.

    Phase 4.9: With --with-orderbook (default), also captures order book
    microstructure (best bid/ask, spread, depth) for each market.

    Phase 5.0: --orderbook-source controls the data source:
    - rest (default): Fetch order books via REST API each cycle
    - wss: Stream order books via WebSocket, fallback to REST if stale/missing

    Example:
        pmq snapshots run --interval 60 --limit 200 --duration-minutes 60
        pmq snapshots run --interval 60 --no-orderbook  # Skip order books
        pmq snapshots run --orderbook-source wss  # Use WebSocket streaming
        pmq snapshots run --orderbook-source wss --wss-staleness 15  # Stricter staleness
    """
    import asyncio
    import os

    settings = get_settings()

    # Validate orderbook-source
    if orderbook_source not in ("rest", "wss"):
        console.print(
            f"[red]Invalid --orderbook-source: {orderbook_source}. Must be 'rest' or 'wss'.[/red]"
        )
        raise typer.Exit(1)

    # Check kill switch
    if settings.safety.kill_switch or os.environ.get("PMQ_SNAPSHOT_KILL", "").lower() == "true":
        console.print("[red]Kill switch is active. Snapshot collection halted.[/red]")
        raise typer.Exit(1)

    orderbook_str = "[green]ON[/green]" if with_orderbook else "[dim]OFF[/dim]"
    source_str = f"[cyan]{orderbook_source.upper()}[/cyan]" if with_orderbook else "[dim]N/A[/dim]"
    console.print(
        f"[bold green]Starting snapshot collection[/bold green]\n"
        f"Interval: {interval}s\n"
        f"Limit: {limit} markets\n"
        f"Duration: {'infinite' if duration_minutes == 0 else f'{duration_minutes} minutes'}\n"
        f"Order Book: {orderbook_str}\n"
        f"Order Book Source: {source_str}\n"
    )

    dao = DAO()
    client = GammaClient()

    # Phase 4.9: OrderBook fetcher for REST microstructure data
    ob_fetcher = None
    if with_orderbook:
        from pmq.markets.orderbook import OrderBookFetcher

        ob_fetcher = OrderBookFetcher()

    # Phase 5.0: WSS client for streaming order books
    wss_client = None
    if with_orderbook and orderbook_source == "wss":
        from pmq.markets.wss_market import MarketWssClient

        wss_client = MarketWssClient(staleness_seconds=wss_staleness_seconds)

    # WSS stats tracking
    wss_hits = 0
    wss_fallbacks = 0

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60) if duration_minutes > 0 else float("inf")
    cycle_count = 0
    backoff = 1
    max_backoff = 300

    async def run_wss_collection() -> None:
        """Async wrapper for WSS-enabled collection loop."""
        nonlocal cycle_count, backoff, wss_hits, wss_fallbacks

        if wss_client:
            await wss_client.connect()
            connected = await wss_client.wait_connected(timeout=10.0)
            if not connected:
                console.print("[yellow]WSS connection timeout, will retry...[/yellow]")

        try:
            while time.time() < end_time:
                cycle_count += 1
                now = datetime.now(UTC).isoformat()

                # Check kill switch each cycle
                if os.environ.get("PMQ_SNAPSHOT_KILL", "").lower() == "true":
                    console.print("[red]Kill switch activated. Stopping.[/red]")
                    break

                console.print(f"\n[cyan]Cycle {cycle_count} @ {now[:19]}[/cyan]")

                try:
                    # Fetch markets
                    console.print(f"[dim]Fetching {limit} markets...[/dim]")
                    markets = client.list_markets(limit=limit)

                    # Upsert market data
                    dao.upsert_markets(markets)

                    # Subscribe to new markets if using WSS
                    if wss_client:
                        token_ids = [m.yes_token_id for m in markets if m.yes_token_id]
                        if token_ids:
                            await wss_client.subscribe(token_ids)

                    # Fetch order books (WSS with REST fallback, or REST only)
                    orderbook_data: dict[str, Any] | None = None
                    ob_success_count = 0
                    cycle_wss_hits = 0
                    cycle_fallbacks = 0

                    if with_orderbook:
                        console.print(
                            f"[dim]Fetching order books ({orderbook_source.upper()})...[/dim]"
                        )
                        orderbook_data = {}

                        for market in markets:
                            token_id = market.yes_token_id
                            if not token_id:
                                continue

                            ob = None

                            # Try WSS cache first if enabled
                            if wss_client:
                                ob = wss_client.get_orderbook(token_id)
                                if ob and ob.has_valid_book:
                                    cycle_wss_hits += 1

                            # Fallback to REST if WSS miss/stale
                            if ob is None and ob_fetcher:
                                try:
                                    ob = ob_fetcher.fetch_order_book(token_id)
                                    if wss_client:  # Track fallback only in WSS mode
                                        cycle_fallbacks += 1
                                except Exception as e:
                                    logger.debug(
                                        f"REST order book fetch failed for {market.id}: {e}"
                                    )

                            if ob and ob.has_valid_book:
                                orderbook_data[market.id] = ob.to_dict()
                                ob_success_count += 1

                        wss_hits += cycle_wss_hits
                        wss_fallbacks += cycle_fallbacks

                    # Save snapshots
                    snapshot_time = datetime.now(UTC).isoformat()
                    snapshot_count = dao.save_snapshots_bulk(markets, snapshot_time, orderbook_data)

                    msg = (
                        f"[green]✓ Saved {snapshot_count} snapshots at {snapshot_time[:19]}[/green]"
                    )
                    if with_orderbook:
                        msg += f" [dim]({ob_success_count} with order books)[/dim]"
                        if wss_client:
                            msg += f" [dim](WSS: {cycle_wss_hits}, REST fallback: {cycle_fallbacks})[/dim]"
                    console.print(msg)

                    # Update runtime state
                    dao.set_runtime_state("last_snapshot_at", snapshot_time)
                    dao.set_runtime_state("snapshot_cycle_count", str(cycle_count))

                    # Reset backoff on success
                    backoff = 1

                except GammaClientError as e:
                    dao.set_runtime_state("last_snapshot_error", f"{now}: {e}")
                    console.print(f"[red]API error: {e}[/red]")
                    console.print(f"[yellow]Backing off {backoff}s...[/yellow]")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    continue

                except Exception as e:
                    dao.set_runtime_state("last_snapshot_error", f"{now}: {e}")
                    console.print(f"[red]Error: {e}[/red]")

                # Wait for next cycle
                if time.time() < end_time:
                    await asyncio.sleep(interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        finally:
            if wss_client:
                await wss_client.close()

    # Run the collection loop
    try:
        if orderbook_source == "wss":
            # Run async WSS-enabled collection
            asyncio.run(run_wss_collection())
        else:
            # Run sync REST-only collection
            try:
                while time.time() < end_time:
                    cycle_count += 1
                    now = datetime.now(UTC).isoformat()

                    # Check kill switch each cycle
                    if os.environ.get("PMQ_SNAPSHOT_KILL", "").lower() == "true":
                        console.print("[red]Kill switch activated. Stopping.[/red]")
                        break

                    console.print(f"\n[cyan]Cycle {cycle_count} @ {now[:19]}[/cyan]")

                    try:
                        # Fetch markets
                        console.print(f"[dim]Fetching {limit} markets...[/dim]")
                        markets = client.list_markets(limit=limit)

                        # Upsert market data
                        dao.upsert_markets(markets)

                        # Fetch order books via REST
                        orderbook_data: dict[str, Any] | None = None
                        ob_success_count = 0
                        if ob_fetcher and with_orderbook:
                            console.print("[dim]Fetching order books (REST)...[/dim]")
                            orderbook_data = {}
                            for market in markets:
                                token_id = market.yes_token_id
                                if token_id:
                                    try:
                                        ob = ob_fetcher.fetch_order_book(token_id)
                                        if ob.has_valid_book:
                                            orderbook_data[market.id] = ob.to_dict()
                                            ob_success_count += 1
                                    except Exception as e:
                                        logger.debug(
                                            f"Order book fetch failed for {market.id}: {e}"
                                        )

                        # Save snapshots
                        snapshot_time = datetime.now(UTC).isoformat()
                        snapshot_count = dao.save_snapshots_bulk(
                            markets, snapshot_time, orderbook_data
                        )

                        msg = f"[green]✓ Saved {snapshot_count} snapshots at {snapshot_time[:19]}[/green]"
                        if with_orderbook:
                            msg += f" [dim]({ob_success_count} with order books)[/dim]"
                        console.print(msg)

                        # Update runtime state
                        dao.set_runtime_state("last_snapshot_at", snapshot_time)
                        dao.set_runtime_state("snapshot_cycle_count", str(cycle_count))

                        # Reset backoff on success
                        backoff = 1

                    except GammaClientError as e:
                        dao.set_runtime_state("last_snapshot_error", f"{now}: {e}")
                        console.print(f"[red]API error: {e}[/red]")
                        console.print(f"[yellow]Backing off {backoff}s...[/yellow]")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue

                    except Exception as e:
                        dao.set_runtime_state("last_snapshot_error", f"{now}: {e}")
                        console.print(f"[red]Error: {e}[/red]")

                    # Wait for next cycle
                    if time.time() < end_time:
                        time.sleep(interval)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user[/yellow]")

    finally:
        client.close()
        if ob_fetcher:
            ob_fetcher.close()

    elapsed = int((time.time() - start_time) / 60)
    summary = f"\n[bold]Completed {cycle_count} cycles in {elapsed} minutes[/bold]"
    if orderbook_source == "wss" and (wss_hits + wss_fallbacks) > 0:
        wss_pct = (
            (wss_hits / (wss_hits + wss_fallbacks)) * 100 if (wss_hits + wss_fallbacks) > 0 else 0
        )
        summary += f"\n[dim]WSS coverage: {wss_pct:.1f}% ({wss_hits} hits, {wss_fallbacks} REST fallbacks)[/dim]"
    console.print(summary)


@snapshots_app.command("quality")
def snapshots_quality(
    from_date: Annotated[
        str | None,
        typer.Option("--from", help="Start date (YYYY-MM-DD or ISO)"),
    ] = None,
    to_date: Annotated[
        str | None,
        typer.Option("--to", help="End date (YYYY-MM-DD or ISO)"),
    ] = None,
    last_minutes: Annotated[
        int | None,
        typer.Option("--last-minutes", help="Rolling window: last N minutes"),
    ] = None,
    last_times: Annotated[
        int | None,
        typer.Option("--last-times", help="Last K distinct snapshot times"),
    ] = None,
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Expected interval in seconds"),
    ] = 60,
    contiguous: Annotated[
        bool | None,
        typer.Option(
            "--contiguous/--no-contiguous",
            help="Stop at gaps (default True for --last-times, ignored for other modes)",
        ),
    ] = None,
) -> None:
    """Analyze snapshot data quality.

    Supports three window modes:
    1. Explicit: --from/--to date range
    2. Rolling: --last-minutes N (last N minutes)
    3. Last times: --last-times K (last K distinct snapshots)

    The --last-times mode is recommended for evaluating recent data quality
    without being penalized by historical gaps.

    Gap-aware (contiguous) mode:
    When --contiguous is enabled (default for --last-times), the analyzer
    stops at gaps in the data to avoid penalizing recent healthy data due
    to old session gaps. Use --no-contiguous to analyze all available times.

    Examples:
        pmq snapshots quality --from 2024-01-01 --to 2024-01-07 --interval 60
        pmq snapshots quality --last-minutes 60 --interval 60
        pmq snapshots quality --last-times 30 --interval 60
        pmq snapshots quality --last-times 200 --interval 60 --contiguous
        pmq snapshots quality --last-times 200 --interval 60 --no-contiguous
    """
    from pmq.quality import QualityChecker, QualityReporter

    # Validate arguments - exactly one window mode must be specified
    explicit_mode = from_date is not None and to_date is not None
    rolling_mode = last_minutes is not None
    times_mode = last_times is not None

    mode_count = sum([explicit_mode, rolling_mode, times_mode])
    if mode_count == 0:
        console.print("[red]Error: Must specify --from/--to, --last-minutes, or --last-times[/red]")
        raise typer.Exit(1)
    if mode_count > 1:
        console.print("[red]Error: Cannot combine window modes. Use one of:[/red]")
        console.print("  --from/--to (explicit range)")
        console.print("  --last-minutes (rolling window)")
        console.print("  --last-times (last K snapshots)")
        raise typer.Exit(1)

    checker = QualityChecker()
    reporter = QualityReporter()

    # Determine contiguous mode: default True for last-times, False otherwise
    use_contiguous = contiguous if contiguous is not None else times_mode

    with console.status("[bold green]Analyzing snapshot quality..."):
        if rolling_mode:
            assert last_minutes is not None
            result = checker.check_last_minutes(last_minutes, interval)
            window_desc = f"Last {last_minutes} minutes"
        elif times_mode:
            assert last_times is not None
            result = checker.check_last_times(last_times, interval, contiguous=use_contiguous)
            window_desc = f"Last {last_times} snapshot times"
            if use_contiguous:
                window_desc += " (contiguous)"
        else:
            # Phase 4.7: Use check_explicit_window for proper expected/observed computation
            assert from_date is not None and to_date is not None
            # Normalize times
            start_time = from_date if "T" in from_date else f"{from_date}T00:00:00+00:00"
            end_time = to_date if "T" in to_date else f"{to_date}T23:59:59+00:00"
            result = checker.check_explicit_window(
                start_time=start_time,
                end_time=end_time,
                expected_interval_seconds=interval,
            )
            window_desc = f"{from_date} to {to_date}"

    # Display results
    status = reporter.get_status_badge(result)
    status_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(status, "dim")

    # Readiness badge
    ready_badge = (
        "[green]✓ READY[/green]" if result.ready_for_scorecard else "[yellow]⚠ NOT READY[/yellow]"
    )

    console.print(
        f"\n[bold]Quality Report: [{status_color}]{status.upper()}[/{status_color}][/bold]  {ready_badge}"
    )

    table = Table(title="Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Window Mode", result.window_mode)
    table.add_row("Window", window_desc)
    if result.window_from and result.window_to:
        table.add_row("Actual Range", f"{result.window_from[:19]} to {result.window_to[:19]}")
    table.add_row("Expected Interval", f"{interval}s")
    table.add_row("Distinct Times", f"{result.distinct_times} / {result.expected_times}")
    table.add_row("Total Snapshots", str(result.snapshots_written))
    table.add_row("Markets Covered", str(result.markets_seen))
    table.add_row("Coverage", f"{result.coverage_pct:.1f}%")
    table.add_row("Maturity Score", f"{result.maturity_score}/100")
    table.add_row("Ready for Scorecard", "Yes" if result.ready_for_scorecard else "No")
    table.add_row("Missing Intervals", str(result.missing_intervals))
    table.add_row("Largest Gap", f"{result.largest_gap_seconds:.0f}s")
    table.add_row("Duplicates", str(result.duplicate_count))

    # Contiguous mode info (Phase 4.5)
    if result.contiguous:
        table.add_row("Contiguous Mode", "[green]Yes[/green]")
        if result.gap_cutoff_time:
            table.add_row("Gap Cutoff", result.gap_cutoff_time[:19])
            total_avail = result.notes.get("total_available", 0)
            table.add_row("Times Used", f"{result.distinct_times} of {total_avail} available")

    console.print(table)

    # Show top gaps if any
    if result.gaps:
        console.print(
            f"\n[yellow]Found {len(result.gaps)} gaps (>50% of expected interval)[/yellow]"
        )
        for gap in result.gaps[:5]:
            console.print(
                f"  Gap: {gap.gap_start[:19]} to {gap.gap_end[:19]} "
                f"({gap.gap_seconds:.0f}s, expected {gap.expected_seconds}s)"
            )


@snapshots_app.command("coverage")
def snapshots_coverage(
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (YYYY-MM-DD or ISO)"),
    ],
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (YYYY-MM-DD or ISO)"),
    ],
) -> None:
    """Show snapshot coverage summary for a time window.

    Displays market-level coverage statistics.

    Example:
        pmq snapshots coverage --from 2024-01-01 --to 2024-01-07
    """
    from pmq.quality import QualityReporter

    reporter = QualityReporter()

    with console.status("[bold green]Calculating coverage..."):
        summary = reporter.get_coverage_summary(from_date, to_date)

    console.print(f"\n[bold]Snapshot Coverage: {from_date} to {to_date}[/bold]")

    table = Table(title="Coverage Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Snapshots", str(summary["total_snapshots"]))
    table.add_row("Distinct Times", str(summary["distinct_times"]))
    table.add_row("Markets Covered", str(summary["markets_covered"]))

    overall = summary.get("overall_stats", {})
    if overall:
        table.add_row("Overall Total", str(overall.get("total_snapshots", 0)))
        table.add_row("Last 24h Count", str(overall.get("last_24h_count", 0)))
        if overall.get("first_snapshot"):
            table.add_row("First Snapshot", overall["first_snapshot"][:19])
        if overall.get("last_snapshot"):
            table.add_row("Last Snapshot", overall["last_snapshot"][:19])

    console.print(table)

    # Top markets
    top_markets = summary.get("top_markets", [])
    if top_markets:
        markets_table = Table(title="Top Markets by Snapshot Count")
        markets_table.add_column("Market ID", style="cyan", max_width=16)
        markets_table.add_column("Snapshots", justify="right")
        markets_table.add_column("First", style="dim")
        markets_table.add_column("Last", style="dim")

        for m in top_markets[:10]:
            markets_table.add_row(
                m["market_id"][:16] + "...",
                str(m["snapshot_count"]),
                m.get("first", "")[:10] if m.get("first") else "",
                m.get("last", "")[:10] if m.get("last") else "",
            )

        console.print(markets_table)


@snapshots_app.command("summary")
def snapshots_summary() -> None:
    """Show overall snapshot statistics.

    Example:
        pmq snapshots summary
    """
    dao = DAO()
    summary = dao.get_snapshot_summary()

    console.print("\n[bold]Snapshot Summary[/bold]")

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Snapshots", str(summary["total_snapshots"]))
    table.add_row("Unique Markets", str(summary["unique_markets"]))
    table.add_row("Last 24h", str(summary["last_24h_count"]))

    if summary.get("first_snapshot"):
        table.add_row("First Snapshot", summary["first_snapshot"][:19])
    if summary.get("last_snapshot"):
        table.add_row("Last Snapshot", summary["last_snapshot"][:19])

    console.print(table)

    # Show latest quality report if exists
    latest_report = dao.get_latest_quality_report()
    if latest_report:
        console.print("\n[bold]Latest Quality Report[/bold]")
        q_table = Table()
        q_table.add_column("Metric", style="cyan")
        q_table.add_column("Value", justify="right")

        q_table.add_row(
            "Window", f"{latest_report['window_from'][:10]} to {latest_report['window_to'][:10]}"
        )
        q_table.add_row("Coverage", f"{latest_report['coverage_pct']:.1f}%")
        q_table.add_row("Missing Intervals", str(latest_report["missing_intervals"]))
        q_table.add_row("Duplicates", str(latest_report["duplicate_count"]))
        q_table.add_row("Created", latest_report["created_at"][:19])

        console.print(q_table)


# =============================================================================
# Strategy Approval Commands (Phase 3)
# =============================================================================

approve_app = typer.Typer(help="Strategy approval and risk governance commands")
app.add_typer(approve_app, name="approve")


@approve_app.command("evaluate")
def approve_evaluate(
    run_id: Annotated[
        str,
        typer.Option("--run-id", "-r", help="Backtest run ID to evaluate"),
    ],
    validation_mode: Annotated[
        bool,
        typer.Option(
            "--validation",
            help="Use relaxed thresholds for baseline validation strategies",
        ),
    ] = False,
) -> None:
    """Evaluate a backtest run for approval.

    Computes a strategy scorecard with pass/fail criteria and
    recommended risk limits.

    Use --validation for observer/baseline strategies that need
    relaxed thresholds (e.g., 0 trades allowed).

    Example:
        pmq approve evaluate --run-id abc123
        pmq approve evaluate --run-id abc123 --validation
    """
    from pmq.backtest import BacktestRunner
    from pmq.governance import compute_scorecard

    dao = DAO()
    runner = BacktestRunner(dao=dao)

    # Get backtest report
    report = runner.get_run_report(run_id)
    if not report:
        console.print(f"[red]Backtest run not found: {run_id}[/red]")
        raise typer.Exit(1)

    run = report["run"]
    metrics = report.get("metrics")

    if not metrics:
        console.print(f"[red]No metrics found for run: {run_id}[/red]")
        raise typer.Exit(1)

    # Auto-enable validation mode for observer strategy
    if run["strategy"] == "observer":
        validation_mode = True

    # Get data quality for the backtest window
    from pmq.quality import QualityReporter

    quality_reporter = QualityReporter()
    quality_report = quality_reporter.generate_report(
        start_time=run["start_date"],
        end_time=run["end_date"],
        save=False,
    )

    # Compute scorecard
    scorecard = compute_scorecard(
        total_pnl=metrics["total_pnl"],
        max_drawdown=metrics["max_drawdown"],
        win_rate=metrics["win_rate"],
        sharpe_ratio=metrics["sharpe_ratio"],
        total_trades=metrics["total_trades"],
        trades_per_day=metrics["trades_per_day"],
        capital_utilization=metrics.get("capital_utilization", 0.5),
        initial_balance=run.get("initial_balance", 10000.0),
        data_quality_pct=quality_report.coverage_pct,
        validation_mode=validation_mode,
        data_quality_status=quality_report.status,
        maturity_score=quality_report.maturity_score,
        ready_for_scorecard=quality_report.ready_for_scorecard,
    )

    # Display results
    mode_suffix = " (validation mode)" if validation_mode else ""
    status = "[green]PASSED[/green]" if scorecard.passed else "[red]FAILED[/red]"
    console.print(f"\n[bold]Strategy Evaluation: {status}{mode_suffix}[/bold]")
    console.print(f"Run ID: [cyan]{run_id}[/cyan]")
    console.print(f"Strategy: [cyan]{run['strategy']}[/cyan]")
    console.print(f"Data Quality Status: [cyan]{quality_report.status}[/cyan]")
    console.print(
        f"Maturity: [cyan]{quality_report.maturity_score}/100[/cyan] {'✓ Ready' if quality_report.ready_for_scorecard else '⚠ Not Ready'}"
    )

    # Score breakdown
    score_table = Table(title="Scorecard")
    score_table.add_column("Metric", style="cyan")
    score_table.add_column("Value", justify="right")
    score_table.add_column("Score", justify="right")

    score_table.add_row("Total PnL", f"${metrics['total_pnl']:,.2f}", "—")
    score_table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}", "—")
    score_table.add_row("Win Rate", f"{metrics['win_rate']:.2%}", "—")
    score_table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", "—")
    score_table.add_row("Trades/Day", f"{metrics['trades_per_day']:.1f}", "—")
    score_table.add_row("Data Quality", f"{quality_report.coverage_pct:.1f}%", "—")
    score_table.add_row("[bold]TOTAL SCORE[/bold]", "", f"[bold]{scorecard.score:.1f}/100[/bold]")

    console.print(score_table)

    # Reasons
    if scorecard.reasons:
        console.print("\n[bold]Pass/Fail Reasons:[/bold]")
        for reason in scorecard.reasons:
            icon = "✓" if "pass" in reason.lower() or "ok" in reason.lower() else "✗"
            color = "green" if icon == "✓" else "red"
            console.print(f"  [{color}]{icon}[/{color}] {reason}")

    if scorecard.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in scorecard.warnings:
            console.print(f"  ⚠ {warning}")

    # Recommended limits
    if scorecard.passed:
        limits = scorecard.recommended_limits
        limits_table = Table(title="Recommended Risk Limits")
        limits_table.add_column("Limit", style="cyan")
        limits_table.add_column("Value", justify="right")

        limits_table.add_row("Max Notional/Market", f"${limits.max_notional_per_market:,.0f}")
        limits_table.add_row("Max Total Notional", f"${limits.max_total_notional:,.0f}")
        limits_table.add_row("Max Positions", str(limits.max_positions))
        limits_table.add_row("Max Trades/Hour", str(limits.max_trades_per_hour))
        limits_table.add_row("Stop Loss", f"{limits.stop_loss_pct:.1%}")

        console.print(limits_table)

        validation_flag = " --validation" if validation_mode else ""
        console.print(
            f"\n[dim]To approve: pmq approve grant --run-id {run_id} "
            f"--name {run['strategy']} --version v1{validation_flag}[/dim]"
        )
    else:
        if validation_mode:
            console.print(
                "\n[dim]Strategy did not pass even in validation mode. Check for critical issues.[/dim]"
            )
        else:
            console.print(
                "\n[dim]Strategy did not pass. Try --validation for baseline strategies, or improve performance.[/dim]"
            )


@approve_app.command("grant")
def approve_grant(
    run_id: Annotated[
        str,
        typer.Option("--run-id", "-r", help="Backtest run ID to approve"),
    ],
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Strategy name"),
    ],
    version: Annotated[
        str,
        typer.Option("--version", help="Strategy version"),
    ],
    approved_by: Annotated[
        str,
        typer.Option("--approved-by", help="Approver identifier"),
    ] = "cli",
    force: Annotated[
        bool,
        typer.Option("--force", help="Force approval even if scorecard fails"),
    ] = False,
    validation_mode: Annotated[
        bool,
        typer.Option(
            "--validation",
            help="Use relaxed thresholds for baseline validation strategies",
        ),
    ] = False,
) -> None:
    """Grant approval for a strategy based on backtest results.

    Creates an APPROVED record that enables paper/live execution.

    Example:
        pmq approve grant --run-id abc123 --name arb --version v1
        pmq approve grant --run-id abc123 --name observer --version v1 --validation
    """
    from pmq.backtest import BacktestRunner
    from pmq.governance import compute_scorecard, limits_to_dict

    dao = DAO()
    runner = BacktestRunner(dao=dao)

    # Get backtest report
    report = runner.get_run_report(run_id)
    if not report:
        console.print(f"[red]Backtest run not found: {run_id}[/red]")
        raise typer.Exit(1)

    run = report["run"]
    metrics = report.get("metrics")

    if not metrics:
        console.print(f"[red]No metrics found for run: {run_id}[/red]")
        raise typer.Exit(1)

    # Auto-enable validation mode for observer strategy
    if run["strategy"] == "observer":
        validation_mode = True

    # Get data quality
    from pmq.quality import QualityReporter

    quality_reporter = QualityReporter()
    quality_report = quality_reporter.generate_report(
        start_time=run["start_date"],
        end_time=run["end_date"],
        save=False,
    )

    # Compute scorecard
    scorecard = compute_scorecard(
        total_pnl=metrics["total_pnl"],
        max_drawdown=metrics["max_drawdown"],
        win_rate=metrics["win_rate"],
        sharpe_ratio=metrics["sharpe_ratio"],
        total_trades=metrics["total_trades"],
        trades_per_day=metrics["trades_per_day"],
        capital_utilization=metrics.get("capital_utilization", 0.5),
        initial_balance=run.get("initial_balance", 10000.0),
        data_quality_pct=quality_report.coverage_pct,
        validation_mode=validation_mode,
        data_quality_status=quality_report.status,
        maturity_score=quality_report.maturity_score,
        ready_for_scorecard=quality_report.ready_for_scorecard,
    )

    if not scorecard.passed and not force:
        console.print("[red]Scorecard failed. Use --force to approve anyway.[/red]")
        console.print("\n[dim]Reasons:[/dim]")
        for reason in scorecard.reasons:
            console.print(f"  • {reason}")
        raise typer.Exit(1)

    # Get manifest for git SHA / config hash
    manifest = dao.get_backtest_manifest(run_id)
    git_sha = manifest.get("git_sha") if manifest else None
    config_hash = manifest.get("config_hash") if manifest else None

    # Save approval
    if validation_mode:
        status = "APPROVED_VALIDATION" if scorecard.passed else "APPROVED_FORCED"
    else:
        status = "APPROVED" if scorecard.passed else "APPROVED_FORCED"
    approval_id = dao.save_approval(
        strategy_name=name,
        strategy_version=version,
        window_from=run["start_date"],
        window_to=run["end_date"],
        run_id=run_id,
        git_sha=git_sha,
        config_hash=config_hash,
        score=scorecard.score,
        status=status,
        reasons=scorecard.reasons,
        limits=limits_to_dict(scorecard.recommended_limits),
        approved_by=approved_by,
    )

    # Log risk event
    dao.save_risk_event(
        severity="INFO",
        event_type="APPROVAL_GRANTED",
        strategy_name=name,
        message=f"Strategy {name} v{version} approved with score {scorecard.score:.1f}",
        details={
            "approval_id": approval_id,
            "run_id": run_id,
            "score": scorecard.score,
            "forced": not scorecard.passed,
        },
    )

    console.print("\n[bold green]✓ Strategy APPROVED[/bold green]")
    console.print(f"Approval ID: [cyan]{approval_id}[/cyan]")
    console.print(f"Strategy: [cyan]{name} v{version}[/cyan]")
    console.print(f"Score: [cyan]{scorecard.score:.1f}/100[/cyan]")

    if not scorecard.passed:
        console.print("[yellow]⚠ Approval was forced despite failing scorecard[/yellow]")

    console.print(
        f"\n[dim]This strategy is now eligible for: pmq paper run --strategy {name}[/dim]"
    )


@approve_app.command("list")
def approve_list(
    status: Annotated[
        str | None,
        typer.Option("--status", "-s", help="Filter by status: APPROVED, REVOKED, PENDING"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum approvals to show"),
    ] = 20,
) -> None:
    """List strategy approvals.

    Example:
        pmq approve list
        pmq approve list --status APPROVED
    """
    dao = DAO()
    approvals = dao.get_approvals(status=status, limit=limit)

    if not approvals:
        console.print("[dim]No approvals found[/dim]")
        return

    table = Table(title="Strategy Approvals")
    table.add_column("ID", style="dim")
    table.add_column("Strategy", style="cyan")
    table.add_column("Version")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    table.add_column("Approved By")
    table.add_column("Created")

    for appr in approvals:
        status_str = appr["status"]
        status_color = {"APPROVED": "green", "REVOKED": "red", "PENDING": "yellow"}.get(
            status_str, "dim"
        )

        table.add_row(
            str(appr["id"]),
            appr["strategy_name"],
            appr["strategy_version"],
            f"{appr['score']:.1f}",
            f"[{status_color}]{status_str}[/{status_color}]",
            appr.get("approved_by") or "—",
            appr["created_at"][:19],
        )

    console.print(table)


@approve_app.command("revoke")
def approve_revoke(
    approval_id: Annotated[
        int,
        typer.Option("--approval-id", "-a", help="Approval ID to revoke"),
    ],
    reason: Annotated[
        str,
        typer.Option("--reason", "-r", help="Reason for revocation"),
    ],
) -> None:
    """Revoke a strategy approval.

    Example:
        pmq approve revoke --approval-id 1 --reason "Degraded performance"
    """
    dao = DAO()

    # Check approval exists
    approval = dao.get_approval(approval_id)
    if not approval:
        console.print(f"[red]Approval not found: {approval_id}[/red]")
        raise typer.Exit(1)

    if approval["status"] == "REVOKED":
        console.print("[yellow]Approval already revoked[/yellow]")
        return

    # Revoke
    dao.revoke_approval(approval_id, reason)

    # Log risk event
    dao.save_risk_event(
        severity="WARN",
        event_type="APPROVAL_REVOKED",
        strategy_name=approval["strategy_name"],
        message=f"Approval {approval_id} revoked: {reason}",
        details={
            "approval_id": approval_id,
            "reason": reason,
        },
    )

    console.print(f"[bold yellow]✓ Approval {approval_id} revoked[/bold yellow]")
    console.print(
        f"Strategy: [cyan]{approval['strategy_name']} v{approval['strategy_version']}[/cyan]"
    )
    console.print(f"Reason: {reason}")


@approve_app.command("risk-events")
def approve_risk_events(
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum events to show"),
    ] = 50,
    severity: Annotated[
        str | None,
        typer.Option("--severity", "-s", help="Filter by severity: INFO, WARN, CRITICAL"),
    ] = None,
) -> None:
    """List risk events.

    Example:
        pmq approve risk-events
        pmq approve risk-events --severity CRITICAL
    """
    dao = DAO()
    events = dao.get_risk_events(severity=severity, limit=limit)

    if not events:
        console.print("[dim]No risk events found[/dim]")
        return

    table = Table(title="Risk Events")
    table.add_column("Time", style="dim")
    table.add_column("Severity")
    table.add_column("Type", style="cyan")
    table.add_column("Strategy")
    table.add_column("Message", max_width=50)

    for event in events:
        sev = event["severity"]
        sev_color = {"INFO": "blue", "WARN": "yellow", "CRITICAL": "red"}.get(sev, "dim")

        table.add_row(
            event["created_at"][:19],
            f"[{sev_color}]{sev}[/{sev_color}]",
            event["event_type"],
            event.get("strategy_name") or "—",
            event["message"][:50],
        )

    console.print(table)


# =============================================================================
# Evaluation Pipeline Commands (Phase 4)
# =============================================================================

eval_app = typer.Typer(help="Evaluation pipeline commands for automated strategy validation")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Strategy to evaluate: arb, statarb, observer"),
    ] = "arb",
    version: Annotated[
        str,
        typer.Option("--version", help="Strategy version string"),
    ] = "v1",
    last_times: Annotated[
        int,
        typer.Option("--last-times", help="Quality window: last K snapshot times"),
    ] = 30,
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Expected snapshot interval in seconds"),
    ] = 60,
    paper_minutes: Annotated[
        int,
        typer.Option("--paper-minutes", help="Optional paper trading smoke test duration (0=skip)"),
    ] = 0,
    quantity: Annotated[
        float,
        typer.Option("--quantity", "-q", help="Trade quantity for backtest"),
    ] = 10.0,
    balance: Annotated[
        float,
        typer.Option("--balance", "-b", help="Initial balance for backtest"),
    ] = 10000.0,
    pairs_config: Annotated[
        str | None,
        typer.Option("--pairs", "-p", help="Pairs config file (required for statarb)"),
    ] = None,
    # Walk-forward options (statarb only)
    walkforward: Annotated[
        bool | None,
        typer.Option(
            "--walkforward/--no-walkforward",
            help="Use walk-forward evaluation for statarb (auto-detect if not set)",
        ),
    ] = None,
    statarb_params: Annotated[
        str | None,
        typer.Option(
            "--statarb-params",
            help="Path to statarb z-score params YAML (default: config/statarb_best.yml)",
        ),
    ] = None,
    train_times: Annotated[
        int,
        typer.Option("--train-times", help="Number of snapshots for TRAIN segment (walk-forward)"),
    ] = 100,
    test_times: Annotated[
        int,
        typer.Option("--test-times", help="Number of snapshots for TEST segment (walk-forward)"),
    ] = 50,
    # Realism parameters (Phase 4.8)
    fee_bps: Annotated[
        float | None,
        typer.Option(
            "--fee-bps",
            help="Fee in basis points (overrides YAML; default 2.0 if not in YAML)",
        ),
    ] = None,
    slippage_bps: Annotated[
        float | None,
        typer.Option(
            "--slippage-bps",
            help="Slippage in basis points (overrides YAML; default 5.0 if not in YAML)",
        ),
    ] = None,
    min_liquidity: Annotated[
        float | None,
        typer.Option(
            "--min-liquidity",
            help="Global min liquidity threshold (overrides per-pair config)",
        ),
    ] = None,
    max_spread: Annotated[
        float | None,
        typer.Option(
            "--max-spread",
            help="Global max spread threshold (overrides per-pair config)",
        ),
    ] = None,
) -> None:
    """Run automated evaluation pipeline.

    Executes end-to-end strategy validation:
    1. Check data quality (must be READY)
    2. Run deterministic backtest (or walk-forward for statarb)
    3. Evaluate for approval
    4. Optional paper trading smoke test
    5. Generate report

    For statarb strategy, --pairs is required. Generate with:
        pmq statarb pairs suggest --out config/pairs.yml

    Walk-forward evaluation (statarb only):
        When --walkforward is enabled or auto-detected (version contains 'zscore',
        or --statarb-params provided, or config/statarb_best.yml exists), the
        evaluation uses walk-forward methodology:
        - TRAIN segment: fit z-score parameters
        - TEST segment: evaluate strategy (used for scorecard)
        This prevents data leakage and overfitting.

    Realism parameters (Phase 4.8, statarb walk-forward only):
        Cost assumptions and constraint filtering ensure evaluation reflects
        realistic trading conditions. Precedence:
        1. CLI flags (--fee-bps, --slippage-bps, --min-liquidity, --max-spread)
        2. Values in statarb params YAML (if present)
        3. Project defaults (fee=2.0, slippage=5.0; constraints unset unless per-pair)

    Example:
        pmq eval run --strategy arb --version v1 --last-times 30
        pmq eval run --strategy statarb --version v1 --pairs config/pairs.yml --last-times 30
        pmq eval run --strategy statarb --version zscore-v1 --pairs config/pairs.yml --walkforward
        pmq eval run --strategy statarb --pairs config/pairs.yml --statarb-params config/statarb_best.yml
        pmq eval run --strategy statarb --pairs config/pairs.yml --walkforward --fee-bps 3.0 --slippage-bps 8.0
        pmq eval run --strategy statarb --pairs config/pairs.yml --walkforward --min-liquidity 500 --max-spread 0.03
        pmq eval run --strategy observer --version v1 --last-times 30 --paper-minutes 10
    """
    from pmq.evaluation import EvaluationPipeline, EvaluationReporter

    # Print startup info
    pairs_info = f"Pairs: {pairs_config}" if pairs_config else "Pairs: N/A"
    walkforward_info = (
        f"Walk-forward: {walkforward}" if walkforward is not None else "Walk-forward: auto-detect"
    )
    # Build realism info line for statarb
    realism_parts = []
    if fee_bps is not None:
        realism_parts.append(f"fee={fee_bps}bps")
    if slippage_bps is not None:
        realism_parts.append(f"slip={slippage_bps}bps")
    if min_liquidity is not None:
        realism_parts.append(f"min_liq={min_liquidity}")
    if max_spread is not None:
        realism_parts.append(f"max_spread={max_spread}")
    realism_info = f"Realism: {', '.join(realism_parts)}" if realism_parts else ""

    console.print(
        f"\n[bold green]Starting Evaluation Pipeline[/bold green]\n"
        f"Strategy: {strategy} v{version}\n"
        f"Quality Window: last {last_times} snapshots\n"
        f"Interval: {interval}s\n"
        f"{pairs_info}\n"
        f"{walkforward_info}\n"
        + (f"{realism_info}\n" if realism_info else "")
        + f"Paper Minutes: {paper_minutes if paper_minutes > 0 else 'skip'}\n"
    )

    dao = DAO()
    pipeline = EvaluationPipeline(dao=dao)
    reporter = EvaluationReporter(dao=dao)

    try:
        with console.status("[bold green]Running evaluation pipeline..."):
            result = pipeline.run(
                strategy_name=strategy,
                strategy_version=version,
                window_mode="last_times",
                window_value=last_times,
                interval_seconds=interval,
                paper_minutes=paper_minutes if paper_minutes > 0 else None,
                quantity=quantity,
                initial_balance=balance,
                pairs_config=pairs_config,
                walk_forward=walkforward,
                statarb_params_path=statarb_params,
                train_times=train_times,
                test_times=test_times,
                # Phase 4.8: Realism parameters
                fee_bps_override=fee_bps,
                slippage_bps_override=slippage_bps,
                min_liquidity_override=min_liquidity,
                max_spread_override=max_spread,
            )

            # Save reports as artifacts
            reporter.save_report_to_db(result.eval_id, result=result)
    except ValueError as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(1) from e

    # Display results
    status_color = "green" if result.final_status == "PASSED" else "red"
    status_emoji = "✅" if result.final_status == "PASSED" else "❌"

    console.print(f"\n[bold]Evaluation Complete {status_emoji}[/bold]")
    console.print(f"Eval ID: [cyan]{result.eval_id}[/cyan]")
    console.print(f"Final Status: [{status_color}]{result.final_status}[/{status_color}]")

    # Summary table
    table = Table(title="Evaluation Results")
    table.add_column("Step", style="cyan")
    table.add_column("Result", justify="right")

    # Quality
    quality_color = "green" if result.ready_for_scorecard else "yellow"
    table.add_row("Data Quality", f"[{quality_color}]{result.quality_status}[/{quality_color}]")
    table.add_row("Maturity Score", f"{result.maturity_score}/100")
    table.add_row("Ready for Scorecard", "Yes" if result.ready_for_scorecard else "No")

    # Backtest / Walk-forward
    if result.backtest_run_id:
        if result.walk_forward:
            table.add_row("Mode", "[magenta]Walk-Forward (TEST only)[/magenta]")
            table.add_row(
                "TRAIN Window",
                f"{result.train_window_from[:19] if result.train_window_from else 'N/A'} → "
                f"{result.train_window_to[:19] if result.train_window_to else 'N/A'}",
            )
            table.add_row(
                "TEST Window",
                f"{result.test_window_from[:19] if result.test_window_from else 'N/A'} → "
                f"{result.test_window_to[:19] if result.test_window_to else 'N/A'}",
            )
            table.add_row("Fitted Pairs", f"{result.fitted_pairs_count}/{result.total_pairs_count}")
        pnl_color = "green" if result.backtest_pnl >= 0 else "red"
        table.add_row(
            "TEST PnL" if result.walk_forward else "Backtest PnL",
            f"[{pnl_color}]${result.backtest_pnl:.2f}[/{pnl_color}]",
        )
        if result.walk_forward:
            table.add_row("TEST Sharpe", f"{result.backtest_sharpe:.3f}")
            table.add_row("TEST Win Rate", f"{result.backtest_win_rate:.1%}")
            table.add_row("TEST Trades", str(result.backtest_total_trades))
        table.add_row("Backtest Score", f"{result.backtest_score:.1f}/100")

    # Approval
    approval_color = "green" if result.approval_status == "PASSED" else "red"
    table.add_row("Approval", f"[{approval_color}]{result.approval_status}[/{approval_color}]")

    # Paper
    if result.paper_run_id:
        table.add_row("Paper Trades", str(result.paper_trades_count))
        table.add_row("Paper Errors", str(result.paper_errors_count))

    console.print(table)

    # Walk-forward note
    if result.walk_forward:
        console.print(
            "\n[dim italic]Note: Scorecard evaluated on TEST only (walk-forward, no data leakage)[/dim italic]"
        )

    # Summary
    console.print(f"\n[bold]Summary:[/bold] {result.summary}")

    # Commands executed
    if result.commands:
        console.print("\n[dim]Commands executed:[/dim]")
        for cmd in result.commands:
            console.print(f"  [dim]$ {cmd}[/dim]")

    console.print(f"\n[dim]View details: pmq eval report --id {result.eval_id}[/dim]")
    console.print(f"[dim]Export report: pmq eval export --id {result.eval_id} --format md[/dim]")


@eval_app.command("list")
def eval_list(
    strategy: Annotated[
        str | None,
        typer.Option("--strategy", "-s", help="Filter by strategy"),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status: PASSED, FAILED, PENDING"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum evaluations to show"),
    ] = 20,
) -> None:
    """List recent evaluation runs.

    Example:
        pmq eval list
        pmq eval list --strategy arb --status PASSED
    """
    from pmq.evaluation import EvaluationReporter

    dao = DAO()
    reporter = EvaluationReporter(dao=dao)

    evals = reporter.list_evaluations(
        limit=limit,
        strategy_name=strategy,
        status=status,
    )

    if not evals:
        console.print("[dim]No evaluations found[/dim]")
        return

    table = Table(title="Evaluation Runs")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Strategy")
    table.add_column("Version")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    table.add_column("Maturity", justify="right")
    table.add_column("Created")

    for ev in evals:
        eval_id_short = ev["id"][:8] + "..."
        status_str = ev.get("final_status", "PENDING")
        status_color = {"PASSED": "green", "FAILED": "red", "PENDING": "yellow"}.get(
            status_str, "dim"
        )

        score = ev.get("backtest_score", 0)
        maturity = ev.get("maturity_score", 0)

        table.add_row(
            eval_id_short,
            ev.get("strategy_name", "—"),
            ev.get("strategy_version", "—"),
            f"{score:.1f}" if score else "—",
            f"[{status_color}]{status_str}[/{status_color}]",
            f"{maturity}/100" if maturity else "—",
            ev["created_at"][:19],
        )

    console.print(table)


@eval_app.command("report")
def eval_report(
    eval_id: Annotated[
        str,
        typer.Option("--id", help="Evaluation ID"),
    ],
    show_artifacts: Annotated[
        bool,
        typer.Option("--artifacts", help="Show artifact details"),
    ] = False,
) -> None:
    """Display detailed report for an evaluation.

    Example:
        pmq eval report --id abc123
        pmq eval report --id abc123 --artifacts
    """
    from pmq.evaluation import EvaluationReporter

    dao = DAO()
    reporter = EvaluationReporter(dao=dao)

    # Get evaluation
    eval_data = reporter.get_evaluation(eval_id)
    if not eval_data:
        console.print(f"[red]Evaluation not found: {eval_id}[/red]")
        raise typer.Exit(1)

    # Display header
    final_status = eval_data.get("final_status", "UNKNOWN")
    status_emoji = "✅" if final_status == "PASSED" else "❌"
    status_color = "green" if final_status == "PASSED" else "red"

    console.print(f"\n[bold]Evaluation Report {status_emoji}[/bold]")
    console.print(f"ID: [cyan]{eval_id}[/cyan]")
    console.print(
        f"Strategy: [cyan]{eval_data.get('strategy_name', 'N/A')} "
        f"v{eval_data.get('strategy_version', 'N/A')}[/cyan]"
    )
    console.print(f"Created: {eval_data.get('created_at', 'N/A')[:19]}")
    console.print(f"Git SHA: [dim]{eval_data.get('git_sha', 'N/A')}[/dim]")
    console.print(f"Final Status: [{status_color}]{final_status}[/{status_color}]")

    # Summary
    console.print(f"\n[bold]Summary:[/bold] {eval_data.get('summary', 'N/A')}")

    # Quality section
    console.print("\n[bold]Step 1: Data Quality[/bold]")
    quality_table = Table()
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Value")

    quality_table.add_row("Window Mode", eval_data.get("window_mode", "N/A"))
    quality_table.add_row("Status", eval_data.get("quality_status", "N/A"))
    quality_table.add_row("Maturity Score", f"{eval_data.get('maturity_score', 0)}/100")
    quality_table.add_row(
        "Ready for Scorecard", "Yes" if eval_data.get("ready_for_scorecard") else "No"
    )
    if eval_data.get("window_from") and eval_data.get("window_to"):
        quality_table.add_row(
            "Window",
            f"{eval_data['window_from'][:19]} to {eval_data['window_to'][:19]}",
        )

    console.print(quality_table)

    # Backtest section
    if eval_data.get("backtest_run_id"):
        console.print("\n[bold]Step 2: Backtest[/bold]")
        backtest_table = Table()
        backtest_table.add_column("Metric", style="cyan")
        backtest_table.add_column("Value", justify="right")

        pnl = eval_data.get("backtest_pnl", 0)
        pnl_color = "green" if pnl >= 0 else "red"
        backtest_table.add_row("Run ID", eval_data["backtest_run_id"][:16] + "...")
        backtest_table.add_row("PnL", f"[{pnl_color}]${pnl:.2f}[/{pnl_color}]")
        backtest_table.add_row("Score", f"{eval_data.get('backtest_score', 0):.1f}/100")

        console.print(backtest_table)

    # Approval section
    console.print("\n[bold]Step 3: Approval[/bold]")
    approval_status = eval_data.get("approval_status", "PENDING")
    approval_color = "green" if approval_status == "PASSED" else "red"
    console.print(f"Status: [{approval_color}]{approval_status}[/{approval_color}]")

    import json as json_module

    reasons_json = eval_data.get("approval_reasons_json")
    if reasons_json:
        try:
            reasons = json_module.loads(reasons_json)
            if reasons:
                console.print("Reasons:")
                for reason in reasons:
                    icon = "✓" if "pass" in reason.lower() or "ok" in reason.lower() else "✗"
                    color = "green" if icon == "✓" else "red"
                    console.print(f"  [{color}]{icon}[/{color}] {reason}")
        except json_module.JSONDecodeError:
            pass

    # Paper section
    if eval_data.get("paper_run_id"):
        console.print("\n[bold]Step 4: Paper Trading[/bold]")
        paper_table = Table()
        paper_table.add_column("Metric", style="cyan")
        paper_table.add_column("Value", justify="right")

        paper_table.add_row("Run ID", eval_data["paper_run_id"])
        paper_table.add_row("Trades", str(eval_data.get("paper_trades_count", 0)))
        paper_table.add_row("Errors", str(eval_data.get("paper_errors_count", 0)))

        console.print(paper_table)

    # Commands
    commands_json = eval_data.get("commands_json")
    if commands_json:
        try:
            commands = json_module.loads(commands_json)
            if commands:
                console.print("\n[bold]Commands Executed:[/bold]")
                for cmd in commands:
                    console.print(f"  [dim]$ {cmd}[/dim]")
        except json_module.JSONDecodeError:
            pass

    # Artifacts
    if show_artifacts:
        artifacts = reporter.get_artifacts(eval_id)
        if artifacts:
            console.print("\n[bold]Artifacts:[/bold]")
            for art in artifacts:
                console.print(
                    f"  • {art['kind']} ({len(art['content'])} bytes) @ {art['created_at'][:19]}"
                )

    console.print(f"\n[dim]Export: pmq eval export --id {eval_id} --format md|json|csv[/dim]")


@eval_app.command("export")
def eval_export(
    eval_id: Annotated[
        str,
        typer.Option("--id", help="Evaluation ID"),
    ],
    format_type: Annotated[
        str,
        typer.Option("--format", "-f", help="Export format: md, json, csv"),
    ] = "md",
    output_dir: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory"),
    ] = Path("exports"),
) -> None:
    """Export evaluation report to file.

    Example:
        pmq eval export --id abc123 --format md
        pmq eval export --id abc123 --format json --out reports/
    """
    from pmq.evaluation import EvaluationReporter

    dao = DAO()
    reporter = EvaluationReporter(dao=dao)

    # Check evaluation exists
    eval_data = reporter.get_evaluation(eval_id)
    if not eval_data:
        console.print(f"[red]Evaluation not found: {eval_id}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    eval_id_short = eval_id[:8]

    if format_type == "md":
        content = reporter.generate_report_md(eval_data=eval_data)
        filename = output_dir / f"eval_{eval_id_short}_{timestamp}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]✓ Exported markdown report to {filename}[/green]")

    elif format_type == "json":
        content = reporter.generate_report_json(eval_data=eval_data)
        filename = output_dir / f"eval_{eval_id_short}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]✓ Exported JSON report to {filename}[/green]")

    elif format_type == "csv":
        content = reporter.generate_report_csv(eval_data=eval_data)
        filename = output_dir / f"eval_{eval_id_short}_{timestamp}.csv"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]✓ Exported CSV report to {filename}[/green]")

    else:
        console.print(f"[red]Unknown format: {format_type}. Use: md, json, csv[/red]")
        raise typer.Exit(1)


# =============================================================================
# StatArb Commands (Phase 4.1)
# =============================================================================

statarb_app = typer.Typer(help="Statistical arbitrage commands")
app.add_typer(statarb_app, name="statarb")


@statarb_app.command("pairs")
def statarb_pairs(
    action: Annotated[
        str,
        typer.Argument(help="Action: validate, suggest"),
    ],
    pairs_config: Annotated[
        Path,
        typer.Option("--pairs", "-p", help="Path to pairs config file"),
    ] = Path("config/pairs.yml"),
    last_times: Annotated[
        int,
        typer.Option("--last-times", help="For suggest: use last K snapshot times"),
    ] = 30,
    _interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="For suggest: expected snapshot interval seconds"),
    ] = 60,  # noqa: ARG001 - reserved for future use
    output: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="For suggest: output file path"),
    ] = None,
    top: Annotated[
        int,
        typer.Option("--top", help="For suggest: maximum pairs to suggest"),
    ] = 50,
) -> None:
    """Validate or generate pairs configuration.

    Actions:
        validate - Validate an existing pairs config file
        suggest  - Generate candidate pairs from captured snapshot data (DB-only)

    Example:
        pmq statarb pairs validate --pairs config/pairs.yml
        pmq statarb pairs suggest --last-times 30 --interval 60 --out config/pairs.yml
    """
    from pmq.statarb import PairsConfigError, load_validated_pairs_config
    from pmq.statarb.pairs_config import PairConfig, generate_pairs_yaml

    if action == "validate":
        try:
            result = load_validated_pairs_config(pairs_config)
            console.print("[green]✓ Valid pairs configuration[/green]")
            console.print(f"  File: {result.config_path}")
            console.print(f"  Hash: {result.config_hash}")
            console.print(f"  Enabled pairs: {len(result.enabled_pairs)}")
            console.print(f"  Disabled pairs: {len(result.disabled_pairs)}")

            if result.warnings:
                for w in result.warnings:
                    console.print(f"  [yellow]⚠ {w}[/yellow]")

            # Show pairs
            if result.enabled_pairs:
                table = Table(title="Enabled Pairs")
                table.add_column("Name", style="cyan")
                table.add_column("Market A", max_width=20)
                table.add_column("Market B", max_width=20)
                table.add_column("Correlation")

                for p in result.enabled_pairs:
                    table.add_row(
                        p.name,
                        f"{p.market_a_id[:16]}...",
                        f"{p.market_b_id[:16]}...",
                        str(p.correlation),
                    )
                console.print(table)

        except PairsConfigError as e:
            console.print("[red]✗ Invalid pairs configuration[/red]")
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    elif action == "suggest":
        # Suggest pairs from snapshot data (DB-only, no live API)
        dao = DAO()

        # Get recent snapshot times
        snapshot_times = dao.get_recent_snapshot_times(limit=last_times)
        if not snapshot_times:
            console.print("[red]No snapshots found. Run 'pmq sync --snapshot' first.[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Analyzing {len(snapshot_times)} snapshot times...[/cyan]")

        # Get all markets from these snapshots
        start_time = snapshot_times[0]
        end_time = snapshot_times[-1]
        snapshots = dao.get_snapshots(start_time, end_time)

        # Group snapshots by market
        market_snapshots: dict[str, list[dict[str, Any]]] = {}
        for snap in snapshots:
            mid = snap["market_id"]
            if mid not in market_snapshots:
                market_snapshots[mid] = []
            market_snapshots[mid].append(snap)

        # Get market metadata
        market_ids = list(market_snapshots.keys())
        markets = {m["id"]: m for m in dao.get_markets_by_ids(market_ids)}

        console.print(f"[cyan]Found {len(markets)} unique markets with snapshot data[/cyan]")

        # Simple heuristic: Group markets by slug prefix (related events)
        # This is a basic approach; more sophisticated methods could use
        # question similarity, event grouping, etc.
        slug_groups: dict[str, list[str]] = {}
        for mid, market in markets.items():
            slug = market.get("slug", "")
            if not slug:
                continue
            # Extract prefix (first 2-3 words or event slug)
            parts = slug.split("-")
            prefix = "-".join(parts[:3]) if len(parts) >= 3 else slug
            if prefix not in slug_groups:
                slug_groups[prefix] = []
            slug_groups[prefix].append(mid)

        # Generate pairs from groups (markets with same prefix)
        suggested_pairs: list[PairConfig] = []
        pair_count = 0

        for prefix, group_ids in sorted(slug_groups.items(), key=lambda x: -len(x[1])):
            if len(group_ids) < 2:
                continue
            if pair_count >= top:
                break

            # Create pairs within the group
            for i, market_a_id in enumerate(group_ids):
                if pair_count >= top:
                    break
                for market_b_id in group_ids[i + 1 :]:
                    if pair_count >= top:
                        break

                    market_a = markets.get(market_a_id, {})
                    market_b = markets.get(market_b_id, {})

                    # Skip if either market has low coverage
                    a_snapshots = len(market_snapshots.get(market_a_id, []))
                    b_snapshots = len(market_snapshots.get(market_b_id, []))
                    if a_snapshots < 5 or b_snapshots < 5:
                        continue

                    q_a = market_a.get("question", "")[:30]
                    q_b = market_b.get("question", "")[:30]
                    name = f"{prefix}: {q_a} vs {q_b}"
                    suggested_pairs.append(
                        PairConfig(
                            market_a_id=market_a_id,
                            market_b_id=market_b_id,
                            name=name[:80],
                            correlation=1.0,
                            enabled=True,
                        )
                    )
                    pair_count += 1

        if not suggested_pairs:
            console.print(
                "[yellow]No pairs could be suggested from snapshot data.\n"
                "This could mean:\n"
                "  - Markets don't have enough shared slug prefixes\n"
                "  - Not enough snapshots collected yet\n"
                "  - Markets are too diverse (no related events)\n\n"
                "Try collecting more snapshots or manually creating pairs.[/yellow]"
            )
            raise typer.Exit(1)

        console.print(f"[green]✓ Suggested {len(suggested_pairs)} pairs[/green]")

        # Generate YAML
        timestamp = datetime.now(UTC).isoformat()[:19]
        yaml_content = generate_pairs_yaml(
            suggested_pairs,
            header_comment=f"Generated from {len(snapshot_times)} snapshots on {timestamp}",
        )

        # Output
        output_path = output or pairs_config
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        console.print(f"[green]✓ Saved pairs config to {output_path}[/green]")
        console.print("[dim]Review and edit the file, then run 'pmq statarb pairs validate'[/dim]")

    else:
        console.print(f"[red]Unknown action: {action}. Use: validate, suggest[/red]")
        raise typer.Exit(1)


@statarb_app.command("explain")
def statarb_explain(
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (ISO format or YYYY-MM-DD)"),
    ],
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (ISO format or YYYY-MM-DD)"),
    ],
    pairs_config: Annotated[
        Path,
        typer.Option("--pairs", "-p", help="Path to pairs config file"),
    ] = Path("config/pairs.yml"),
) -> None:
    """Explain why statarb produced zero or few signals.

    Provides deterministic diagnostics from captured snapshot data:
    - Number of snapshots in window
    - Per-pair: data coverage, spread statistics, signal counts, skip reasons

    This command is fully deterministic and does not call live APIs.

    Example:
        pmq statarb explain --from 2024-12-30 --to 2024-12-30 --pairs config/pairs.yml
    """
    from pmq.statarb import PairsConfigError, load_validated_pairs_config

    # Normalize dates
    if len(from_date) == 10:
        from_date = f"{from_date}T00:00:00"
    if len(to_date) == 10:
        to_date = f"{to_date}T23:59:59"

    # Load pairs config
    try:
        pairs_result = load_validated_pairs_config(pairs_config)
    except PairsConfigError as e:
        console.print(f"[red]Failed to load pairs config: {e}[/red]")
        raise typer.Exit(1) from e

    dao = DAO()
    settings = get_settings()

    # Get snapshots in window
    snapshot_times = dao.get_snapshot_times(from_date, to_date)
    if not snapshot_times:
        console.print(f"[red]No snapshots found in window {from_date} to {to_date}[/red]")
        console.print("[dim]Run 'pmq sync --snapshot' to collect data first.[/dim]")
        raise typer.Exit(1)

    console.print("\n[bold]StatArb Signal Analysis[/bold]")
    console.print(f"Window: {from_date} to {to_date}")
    console.print(f"Pairs config: {pairs_config} (hash: {pairs_result.config_hash})")
    console.print(f"Entry threshold: {settings.statarb.entry_threshold}")

    # Summary stats
    console.print("\n[bold]Data Summary[/bold]")
    console.print(f"  Snapshot times: {len(snapshot_times)}")
    console.print(f"  First snapshot: {snapshot_times[0]}")
    console.print(f"  Last snapshot: {snapshot_times[-1]}")
    console.print(f"  Enabled pairs: {len(pairs_result.enabled_pairs)}")

    # Analyze each pair
    pair_stats: list[dict[str, Any]] = []

    for pair in pairs_result.enabled_pairs:
        stats: dict[str, Any] = {
            "name": pair.name,
            "market_a_id": pair.market_a_id,
            "market_b_id": pair.market_b_id,
            "correlation": pair.correlation,
            "a_snapshot_count": 0,
            "b_snapshot_count": 0,
            "both_count": 0,
            "spreads": [],
            "signal_count": 0,
            "skip_reasons": [],
        }

        # Get snapshots for both markets
        a_snapshots = dao.get_snapshots(from_date, to_date, [pair.market_a_id])
        b_snapshots = dao.get_snapshots(from_date, to_date, [pair.market_b_id])

        stats["a_snapshot_count"] = len(a_snapshots)
        stats["b_snapshot_count"] = len(b_snapshots)

        if not a_snapshots:
            stats["skip_reasons"].append("Market A has no snapshots in window")
        if not b_snapshots:
            stats["skip_reasons"].append("Market B has no snapshots in window")

        if a_snapshots and b_snapshots:
            # Index by time
            a_by_time = {s["snapshot_time"]: s for s in a_snapshots}
            b_by_time = {s["snapshot_time"]: s for s in b_snapshots}

            # Find overlapping times
            common_times = set(a_by_time.keys()) & set(b_by_time.keys())
            stats["both_count"] = len(common_times)

            if not common_times:
                stats["skip_reasons"].append("No overlapping snapshot times")
            else:
                # Calculate spreads
                for t in sorted(common_times):
                    price_a = a_by_time[t]["yes_price"]
                    price_b = b_by_time[t]["yes_price"]

                    if price_a <= 0 or price_b <= 0:
                        continue

                    if pair.correlation < 0:
                        spread = price_a - (1.0 - price_b)
                    else:
                        spread = price_a - price_b

                    stats["spreads"].append(spread)

                    if abs(spread) > settings.statarb.entry_threshold:
                        stats["signal_count"] += 1

                if not stats["spreads"]:
                    stats["skip_reasons"].append("All prices were zero or invalid")

        pair_stats.append(stats)

    # Display results
    console.print("\n[bold]Per-Pair Analysis[/bold]")

    total_signals = 0
    table = Table()
    table.add_column("Pair", style="cyan", max_width=30)
    table.add_column("A Snaps", justify="right")
    table.add_column("B Snaps", justify="right")
    table.add_column("Both", justify="right")
    table.add_column("Avg Spread", justify="right")
    table.add_column("Max |Spread|", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Skip Reasons", max_width=30)

    for s in pair_stats:
        avg_spread = sum(s["spreads"]) / len(s["spreads"]) if s["spreads"] else 0
        max_abs_spread = max(abs(x) for x in s["spreads"]) if s["spreads"] else 0
        total_signals += s["signal_count"]

        signal_color = "green" if s["signal_count"] > 0 else "dim"
        skip_text = "; ".join(s["skip_reasons"][:2]) if s["skip_reasons"] else "—"

        table.add_row(
            s["name"][:30],
            str(s["a_snapshot_count"]),
            str(s["b_snapshot_count"]),
            str(s["both_count"]),
            f"{avg_spread:.4f}",
            f"{max_abs_spread:.4f}",
            f"[{signal_color}]{s['signal_count']}[/{signal_color}]",
            skip_text,
        )

    console.print(table)

    # Summary
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Total potential signals: {total_signals}")
    if total_signals == 0:
        console.print(
            f"\n[yellow]No signals detected. Possible causes:[/yellow]\n"
            f"  • Entry threshold ({settings.statarb.entry_threshold}) may be too high\n"
            f"  • Pairs may not have enough spread divergence\n"
            f"  • Markets may not have overlapping snapshot coverage\n"
            f"  • Market IDs may not exist in snapshot data\n\n"
            f"[dim]Try adjusting PMQ_STATARB_ENTRY_THRESHOLD or updating pairs config.[/dim]"
        )
    else:
        console.print(
            "\n[green]Signals were detected. If backtest still shows 0 trades:[/green]\n"
            "  • Ensure backtest date range matches this analysis\n"
            "  • Check that --pairs flag is passed to backtest command"
        )


@statarb_app.command("discover")
def statarb_discover(
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (ISO format or YYYY-MM-DD)"),
    ],
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (ISO format or YYYY-MM-DD)"),
    ],
    top: Annotated[
        int,
        typer.Option("--top", help="Maximum pairs to discover"),
    ] = 50,
    min_overlap: Annotated[
        int,
        typer.Option("--min-overlap", help="Minimum overlapping snapshot times"),
    ] = 10,
    min_correlation: Annotated[
        float,
        typer.Option("--min-corr", help="Minimum absolute correlation"),
    ] = 0.3,
    output: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output file path (default: stdout as YAML)"),
    ] = None,
    format_type: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: yml, json"),
    ] = "yml",
) -> None:
    """Discover pairs from snapshot data using correlation analysis.

    Computes correlation of YES prices across overlapping snapshot times.
    Results are deterministic: same inputs produce same output ordering.

    This command does NOT call any live APIs - it only uses stored snapshot data.

    Example:
        pmq statarb discover --from 2024-12-01 --to 2024-12-30 --top 20
        pmq statarb discover --from 2024-12-01 --to 2024-12-30 --out config/pairs.yml
    """
    from pmq.statarb import discover_pairs, generate_pairs_yaml

    # Normalize dates
    if len(from_date) == 10:
        from_date = f"{from_date}T00:00:00"
    if len(to_date) == 10:
        to_date = f"{to_date}T23:59:59"

    dao = DAO()

    console.print(f"[cyan]Discovering pairs from {from_date[:10]} to {to_date[:10]}...[/cyan]")

    # Get all snapshots in range
    snapshots = dao.get_snapshots(from_date, to_date)
    if not snapshots:
        console.print(f"[red]No snapshots found in window {from_date} to {to_date}[/red]")
        console.print("[dim]Run 'pmq sync --snapshot' to collect data first.[/dim]")
        raise typer.Exit(1)

    console.print(f"[cyan]Loaded {len(snapshots)} snapshots[/cyan]")

    # Get market metadata for all markets in snapshots
    market_ids = list({s["market_id"] for s in snapshots})
    markets = {m["id"]: m for m in dao.get_markets_by_ids(market_ids)}
    console.print(f"[cyan]Found {len(markets)} unique markets[/cyan]")

    # Discover pairs
    candidates = discover_pairs(
        snapshots=snapshots,
        markets=markets,
        min_overlap=min_overlap,
        top=top,
        min_correlation=min_correlation,
    )

    if not candidates:
        console.print(
            f"[yellow]No pairs found with correlation >= {min_correlation} "
            f"and overlap >= {min_overlap}.[/yellow]\n"
            "Try:\n"
            "  • Lowering --min-corr (e.g., 0.1)\n"
            "  • Lowering --min-overlap (e.g., 5)\n"
            "  • Collecting more snapshots\n"
            "  • Widening the date range"
        )
        raise typer.Exit(1)

    console.print(f"[green]✓ Discovered {len(candidates)} pairs[/green]")

    # Display table
    table = Table(title="Discovered Pairs")
    table.add_column("#", style="dim")
    table.add_column("Market A", max_width=25)
    table.add_column("Market B", max_width=25)
    table.add_column("Corr", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Max |Spread|", justify="right")

    for i, c in enumerate(candidates[:20], 1):
        q_a = (
            c.market_a_question[:22] + "..."
            if len(c.market_a_question) > 25
            else c.market_a_question
        )
        q_b = (
            c.market_b_question[:22] + "..."
            if len(c.market_b_question) > 25
            else c.market_b_question
        )
        corr_color = "green" if c.correlation > 0 else "red"
        table.add_row(
            str(i),
            q_a or c.market_a_id[:16],
            q_b or c.market_b_id[:16],
            f"[{corr_color}]{c.correlation:+.3f}[/{corr_color}]",
            str(c.overlap_count),
            f"{c.max_abs_spread:.4f}",
        )

    console.print(table)
    if len(candidates) > 20:
        console.print(f"[dim]... and {len(candidates) - 20} more pairs[/dim]")

    # Generate output
    pair_configs = [c.to_pair_config() for c in candidates]
    timestamp = datetime.now(UTC).isoformat()[:19]

    if output:
        if format_type == "json":
            import json

            content = json.dumps(
                {"pairs": [p.__dict__ for p in pair_configs]},
                indent=2,
            )
        else:
            content = generate_pairs_yaml(
                pair_configs,
                header_comment=f"Discovered from snapshots {from_date[:10]} to {to_date[:10]} on {timestamp}",
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]✓ Saved to {output}[/green]")
    else:
        # Print YAML to stdout
        content = generate_pairs_yaml(
            pair_configs,
            header_comment=f"Discovered from snapshots {from_date[:10]} to {to_date[:10]} on {timestamp}",
        )
        console.print("\n[bold]Generated pairs config:[/bold]")
        console.print(content)


@statarb_app.command("validate")
def statarb_validate(
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (ISO format or YYYY-MM-DD)"),
    ],
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (ISO format or YYYY-MM-DD)"),
    ],
    pairs_config: Annotated[
        Path,
        typer.Option("--pairs", "-p", help="Path to pairs config file"),
    ] = Path("config/pairs.yml"),
    min_overlap: Annotated[
        int,
        typer.Option("--min-overlap", help="Minimum required overlapping snapshots"),
    ] = 10,
) -> None:
    """Validate pairs config against snapshot data.

    Checks that each pair has sufficient overlapping snapshot coverage
    in the specified date range.

    Example:
        pmq statarb validate --from 2024-12-01 --to 2024-12-30 --pairs config/pairs.yml
    """
    from pmq.statarb import PairsConfigError, load_validated_pairs_config, validate_pair_overlap

    # Normalize dates
    if len(from_date) == 10:
        from_date = f"{from_date}T00:00:00"
    if len(to_date) == 10:
        to_date = f"{to_date}T23:59:59"

    # Load pairs config
    try:
        pairs_result = load_validated_pairs_config(pairs_config)
    except PairsConfigError as e:
        console.print(f"[red]Failed to load pairs config: {e}[/red]")
        raise typer.Exit(1) from e

    dao = DAO()

    console.print(f"[cyan]Validating {len(pairs_result.enabled_pairs)} pairs...[/cyan]")

    # Get snapshots for all market IDs in pairs
    all_market_ids = set()
    for p in pairs_result.enabled_pairs:
        all_market_ids.add(p.market_a_id)
        all_market_ids.add(p.market_b_id)

    snapshots = dao.get_snapshots(from_date, to_date, list(all_market_ids))

    if not snapshots:
        console.print("[red]No snapshots found for any pair markets in window[/red]")
        console.print("[dim]Run 'pmq sync --snapshot' to collect data first.[/dim]")
        raise typer.Exit(1)

    # Validate each pair
    valid_count = 0
    invalid_count = 0
    results: list[tuple[str, dict[str, Any]]] = []

    for pair in pairs_result.enabled_pairs:
        result = validate_pair_overlap(pair, snapshots, min_overlap)
        results.append((pair.name, result))
        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1

    # Display results
    table = Table(title="Pair Validation Results")
    table.add_column("Pair", style="cyan", max_width=30)
    table.add_column("A Snaps", justify="right")
    table.add_column("B Snaps", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Status")
    table.add_column("Reason", max_width=30)

    for name, result in results:
        status = "[green]✓ Valid[/green]" if result["valid"] else "[red]✗ Invalid[/red]"
        overlap_color = "green" if result["overlap_count"] >= min_overlap else "red"
        table.add_row(
            name[:30],
            str(result["a_count"]),
            str(result["b_count"]),
            f"[{overlap_color}]{result['overlap_count']}[/{overlap_color}]",
            status,
            result["reason"] or "—",
        )

    console.print(table)

    # Summary
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Valid pairs: [green]{valid_count}[/green]")
    console.print(f"  Invalid pairs: [red]{invalid_count}[/red]")
    console.print(f"  Min overlap required: {min_overlap}")

    if invalid_count > 0:
        console.print(
            "\n[yellow]Some pairs have insufficient data coverage.[/yellow]\n"
            "Try:\n"
            "  • Collecting more snapshots (pmq sync --snapshot)\n"
            "  • Using 'pmq statarb discover' to find pairs with better coverage\n"
            "  • Disabling invalid pairs in config (enabled: false)"
        )
        raise typer.Exit(1)

    console.print("\n[green]✓ All pairs have sufficient overlap for the date range.[/green]")


@statarb_app.command("tune")
def statarb_tune(
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (ISO format or YYYY-MM-DD)"),
    ],
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (ISO format or YYYY-MM-DD)"),
    ],
    pairs_config: Annotated[
        Path,
        typer.Option("--pairs", "-p", help="Path to pairs config file"),
    ] = Path("config/pairs.yml"),
    grid_config: Annotated[
        Path | None,
        typer.Option("--grid", "-g", help="Path to grid search config YAML"),
    ] = None,
    train_times: Annotated[
        int,
        typer.Option("--train-times", help="Number of snapshots for TRAIN"),
    ] = 120,
    test_times: Annotated[
        int,
        typer.Option("--test-times", help="Number of snapshots for TEST"),
    ] = 60,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Number of top results to show/save"),
    ] = 10,
    output: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output CSV file for leaderboard"),
    ] = None,
    export_best: Annotated[
        Path | None,
        typer.Option("--export-best", help="Export best config to YAML file"),
    ] = None,
    fee_bps: Annotated[
        float | None,
        typer.Option("--fee-bps", help="Override fee in basis points (default: 2.0)"),
    ] = None,
    slippage_bps: Annotated[
        float | None,
        typer.Option("--slippage-bps", help="Override slippage in basis points (default: 5.0)"),
    ] = None,
) -> None:
    """Run grid search to find optimal statarb parameters.

    Performs walk-forward evaluation for each parameter combination:
    - Split data into TRAIN (fit params) and TEST (evaluate) segments
    - Rank results by Sharpe ratio on TEST data
    - Output leaderboard and optionally export best config

    This command is deterministic and does NOT bypass governance gates.

    Example:
        pmq statarb tune --from 2024-12-01 --to 2024-12-30 --pairs config/pairs.yml
        pmq statarb tune --from 2024-12-01 --to 2024-12-30 --grid config/statarb_grid.yml --out results/tuning.csv
    """
    from pmq.statarb import PairsConfigError, load_validated_pairs_config
    from pmq.statarb.tuning import (
        GridConfig,
        export_best_config,
        load_grid_config,
        run_grid_search,
        save_leaderboard_csv,
    )

    # Normalize dates
    if len(from_date) == 10:
        from_date = f"{from_date}T00:00:00"
    if len(to_date) == 10:
        to_date = f"{to_date}T23:59:59"

    # Load pairs config
    try:
        pairs_result = load_validated_pairs_config(pairs_config)
    except PairsConfigError as e:
        console.print(f"[red]Failed to load pairs config: {e}[/red]")
        raise typer.Exit(1) from e

    if not pairs_result.enabled_pairs:
        console.print("[red]No enabled pairs in config[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Loaded {len(pairs_result.enabled_pairs)} pairs[/cyan]")

    # Load grid config
    if grid_config:
        grid = load_grid_config(grid_config)
        console.print(f"[cyan]Loaded grid config from {grid_config}[/cyan]")
    else:
        grid = GridConfig()
        console.print("[cyan]Using default grid config[/cyan]")

    # Apply CLI overrides for costs (Phase 4.6)
    if fee_bps is not None:
        grid.fee_bps = [fee_bps]
        console.print(f"[cyan]Using fee_bps override: {fee_bps}[/cyan]")
    if slippage_bps is not None:
        grid.slippage_bps = [slippage_bps]
        console.print(f"[cyan]Using slippage_bps override: {slippage_bps}[/cyan]")

    console.print(f"[cyan]Grid has {grid.total_combinations} parameter combinations[/cyan]")
    console.print(f"[cyan]Costs: fee_bps={grid.fee_bps}, slippage_bps={grid.slippage_bps}[/cyan]")

    # Get market IDs from pairs
    pair_market_ids = set()
    for pair in pairs_result.enabled_pairs:
        pair_market_ids.add(pair.market_a_id)
        pair_market_ids.add(pair.market_b_id)

    # Load snapshots
    dao = DAO()
    snapshots = dao.get_snapshots(from_date, to_date, list(pair_market_ids))

    if not snapshots:
        console.print(f"[red]No snapshots found for {from_date[:10]} to {to_date[:10]}[/red]")
        console.print("[dim]Run 'pmq sync --snapshot' to collect data first.[/dim]")
        raise typer.Exit(1)

    console.print(f"[cyan]Loaded {len(snapshots)} snapshots[/cyan]")

    # Run grid search
    console.print("\n[bold]Starting grid search...[/bold]")
    with console.status("[bold green]Running parameter combinations..."):
        leaderboard = run_grid_search(
            snapshots=snapshots,
            pairs=pairs_result.enabled_pairs,
            train_count=train_times,
            test_count=test_times,
            grid=grid,
            top_k=top_k,
        )

    # Display results
    console.print("\n[bold]Tuning Results[/bold]")
    console.print(f"  Combinations tested: {leaderboard.total_combinations}")
    console.print(f"  TRAIN snapshots: {train_times}")
    console.print(f"  TEST snapshots: {test_times}")

    if not leaderboard.results:
        console.print("[red]No valid results from grid search[/red]")
        raise typer.Exit(1)

    # Show leaderboard table
    table = Table(title="Top Parameter Combinations (ranked by Sharpe on TEST)")
    table.add_column("Rank", style="dim")
    table.add_column("Sharpe", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("WR", justify="right")
    table.add_column("DD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("lookback", justify="right")
    table.add_column("entry_z", justify="right")
    table.add_column("exit_z", justify="right")
    table.add_column("max_hold", justify="right")

    for r in leaderboard.results:
        sharpe_color = "green" if r.sharpe > 0 else "red"
        pnl_color = "green" if r.pnl > 0 else "red"
        table.add_row(
            str(r.rank),
            f"[{sharpe_color}]{r.sharpe:.3f}[/{sharpe_color}]",
            f"[{pnl_color}]${r.pnl:.2f}[/{pnl_color}]",
            f"{r.win_rate:.1%}",
            f"{r.max_drawdown:.1%}",
            str(r.total_trades),
            str(r.params["lookback"]),
            str(r.params["entry_z"]),
            str(r.params["exit_z"]),
            str(r.params["max_hold_bars"]),
        )

    console.print(table)

    # Best result summary
    best = leaderboard.results[0]
    console.print("\n[bold]Best Configuration[/bold]")
    console.print(f"  lookback: {best.params['lookback']}")
    console.print(f"  entry_z: {best.params['entry_z']}")
    console.print(f"  exit_z: {best.params['exit_z']}")
    console.print(f"  max_hold_bars: {best.params['max_hold_bars']}")
    console.print(f"  cooldown_bars: {best.params['cooldown_bars']}")
    console.print(f"  fee_bps: {best.params['fee_bps']}")
    console.print(f"  slippage_bps: {best.params['slippage_bps']}")
    console.print(f"  TEST Sharpe: {best.sharpe:.3f}")
    console.print(f"  TEST PnL: ${best.pnl:.2f}")

    # Save outputs
    if output:
        save_leaderboard_csv(leaderboard, output)
        console.print(f"\n[green]✓ Saved leaderboard to {output}[/green]")

    if export_best:
        export_best_config(leaderboard, export_best)
        console.print(f"[green]✓ Exported best config to {export_best}[/green]")

    # Guidance
    console.print(
        "\n[dim]To use best config, set environment variables or update config file:[/dim]"
    )
    console.print(f"[dim]  PMQ_STATARB_LOOKBACK={best.params['lookback']}[/dim]")
    console.print(f"[dim]  PMQ_STATARB_ENTRY_Z={best.params['entry_z']}[/dim]")
    console.print(f"[dim]  PMQ_STATARB_EXIT_Z={best.params['exit_z']}[/dim]")
    console.print(f"[dim]  PMQ_STATARB_MAX_HOLD_BARS={best.params['max_hold_bars']}[/dim]")


@statarb_app.command("walkforward")
def statarb_walkforward(
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (ISO format or YYYY-MM-DD)"),
    ],
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (ISO format or YYYY-MM-DD)"),
    ],
    pairs_config: Annotated[
        Path,
        typer.Option("--pairs", "-p", help="Path to pairs config file"),
    ] = Path("config/pairs.yml"),
    train_times: Annotated[
        int,
        typer.Option("--train-times", help="Number of snapshots for TRAIN"),
    ] = 120,
    test_times: Annotated[
        int,
        typer.Option("--test-times", help="Number of snapshots for TEST"),
    ] = 60,
    lookback: Annotated[
        int,
        typer.Option("--lookback", help="Z-score lookback window"),
    ] = 30,
    entry_z: Annotated[
        float,
        typer.Option("--entry-z", help="Entry z-score threshold"),
    ] = 2.0,
    exit_z: Annotated[
        float,
        typer.Option("--exit-z", help="Exit z-score threshold"),
    ] = 0.5,
    max_hold_bars: Annotated[
        int,
        typer.Option("--max-hold", help="Max bars before forced exit"),
    ] = 60,
    fee_bps: Annotated[
        float,
        typer.Option("--fee-bps", help="Fee in basis points (default: 2.0)"),
    ] = 2.0,
    slippage_bps: Annotated[
        float,
        typer.Option("--slippage-bps", help="Slippage in basis points (default: 5.0)"),
    ] = 5.0,
) -> None:
    """Run single walk-forward evaluation with z-score signals.

    Splits data into TRAIN and TEST segments:
    - Fit OLS beta and spread params on TRAIN
    - Generate signals and compute metrics on TEST
    - Display TRAIN summary and TEST scorecard metrics

    Example:
        pmq statarb walkforward --from 2024-12-01 --to 2024-12-30
        pmq statarb walkforward --from 2024-12-01 --to 2024-12-30 --entry-z 1.5 --exit-z 0.3
    """
    from pmq.backtest.runner import BacktestRunner

    # Run walk-forward backtest
    runner = BacktestRunner()
    run_id, metrics, wf_data = runner.run_walkforward_statarb(
        start_date=from_date,
        end_date=to_date,
        pairs_config=str(pairs_config),
        train_count=train_times,
        test_count=test_times,
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        max_hold_bars=max_hold_bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )

    # Display results
    console.print("\n[bold]Walk-Forward Evaluation Results[/bold]")
    console.print(f"  Run ID: {run_id}")

    if "error" in wf_data:
        console.print(f"[red]Error: {wf_data['error']}[/red]")
        raise typer.Exit(1)

    # TRAIN summary
    split = wf_data.get("split", {})
    console.print("\n[bold]TRAIN Summary[/bold]")
    console.print(f"  Snapshots: {split.get('train_count', 0)}")
    console.print(
        f"  Period: {split.get('first_train', '?')[:10]} to {split.get('last_train', '?')[:10]}"
    )

    fitted = wf_data.get("fitted_params", {})
    valid_pairs = sum(1 for p in fitted.values() if p.get("is_valid", False))
    console.print(f"  Pairs fitted: {valid_pairs}/{len(fitted)}")

    # TEST summary
    test_m = wf_data.get("test_metrics", {})
    console.print("\n[bold]TEST Summary (Scorecard Metrics)[/bold]")
    console.print(f"  Snapshots: {split.get('test_count', 0)}")
    console.print(
        f"  Period: {split.get('first_test', '?')[:10]} to {split.get('last_test', '?')[:10]}"
    )

    pnl_color = "green" if metrics.total_pnl > 0 else "red"
    sharpe_color = (
        "green" if metrics.sharpe_ratio > 1.0 else "yellow" if metrics.sharpe_ratio > 0 else "red"
    )
    wr_color = "green" if metrics.win_rate > 0.5 else "yellow" if metrics.win_rate > 0.4 else "red"

    console.print(f"  Total PnL: [{pnl_color}]${metrics.total_pnl:.2f}[/{pnl_color}]")
    console.print(f"  Sharpe Ratio: [{sharpe_color}]{metrics.sharpe_ratio:.3f}[/{sharpe_color}]")
    console.print(f"  Win Rate: [{wr_color}]{metrics.win_rate:.1%}[/{wr_color}]")
    console.print(f"  Max Drawdown: {metrics.max_drawdown:.1%}")
    console.print(f"  Total Trades: {metrics.total_trades}")
    console.print(f"  Net PnL: ${test_m.get('net_pnl', 0):.2f}")
    console.print(f"  Total Fees: ${test_m.get('total_fees', 0):.2f}")

    # Pair summaries
    pair_summaries = wf_data.get("pair_summaries", [])
    if pair_summaries:
        console.print("\n[bold]Per-Pair Results[/bold]")
        table = Table()
        table.add_column("Pair", style="cyan", max_width=30)
        table.add_column("Status")
        table.add_column("Beta", justify="right")
        table.add_column("TEST Points", justify="right")
        table.add_column("Signals", justify="right")

        for ps in pair_summaries:
            status_color = "green" if ps.get("status") == "evaluated" else "red"
            beta = ps.get("beta", 0)
            table.add_row(
                ps.get("pair_name", "?")[:30],
                f"[{status_color}]{ps.get('status', '?')}[/{status_color}]",
                f"{beta:.3f}" if beta else "—",
                str(ps.get("test_points", 0)),
                str(ps.get("signals", 0)),
            )

        console.print(table)

    # Config used
    config = wf_data.get("config_used", {})
    console.print("\n[bold]Config Used[/bold]")
    console.print(f"  lookback: {config.get('lookback', '?')}")
    console.print(f"  entry_z: {config.get('entry_z', '?')}")
    console.print(f"  exit_z: {config.get('exit_z', '?')}")
    console.print(f"  max_hold_bars: {config.get('max_hold_bars', '?')}")
    console.print(f"  fee_bps: {config.get('fee_bps', '?')}")
    console.print(f"  slippage_bps: {config.get('slippage_bps', '?')}")


# =============================================================================
# Operations Commands (Phase 5.1, 5.2, 5.3)
# =============================================================================

ops_app = typer.Typer(help="Operations commands for production data capture")
app.add_typer(ops_app, name="ops")


@ops_app.command("daemon")
def ops_daemon(
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Snapshot interval in seconds"),
    ] = 60,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Number of markets to fetch per cycle"),
    ] = 200,
    orderbook_source: Annotated[
        str,
        typer.Option(
            "--orderbook-source",
            help="Order book data source: rest, wss (default)",
        ),
    ] = "wss",
    wss_staleness_seconds: Annotated[
        float | None,
        typer.Option(
            "--wss-staleness",
            help="Deprecated: use --wss-health-timeout instead (kept for compat)",
        ),
    ] = None,
    # Phase 5.4: Health-gated fallback flags
    wss_health_timeout: Annotated[
        float,
        typer.Option(
            "--wss-health-timeout",
            help="WSS connection health timeout in seconds (default: 60)",
        ),
    ] = 60.0,
    max_book_age: Annotated[
        float,
        typer.Option(
            "--max-book-age",
            help="Max cache age safety cap in seconds (default: 1800 = 30 min)",
        ),
    ] = 1800.0,
    reconcile_sample: Annotated[
        int,
        typer.Option(
            "--reconcile-sample",
            help="Max markets to reconcile per tick (default: 10)",
        ),
    ] = 10,
    reconcile_min_age: Annotated[
        float,
        typer.Option(
            "--reconcile-min-age",
            help="Min cache age to trigger reconciliation (default: 300s)",
        ),
    ] = 300.0,
    reconcile_timeout: Annotated[
        float,
        typer.Option(
            "--reconcile-timeout",
            help="Timeout for reconciliation REST batch (default: 5s)",
        ),
    ] = 5.0,
    max_hours: Annotated[
        float | None,
        typer.Option(
            "--max-hours",
            help="Maximum hours to run (default: infinite)",
        ),
    ] = None,
    export_dir: Annotated[
        Path,
        typer.Option(
            "--export-dir",
            help="Directory for daily export artifacts",
        ),
    ] = Path("exports"),
    with_orderbook: Annotated[
        bool,
        typer.Option(
            "--with-orderbook/--no-orderbook",
            help="Fetch order book data",
        ),
    ] = True,
    # Phase 5.2: Snapshot export and retention flags
    snapshot_export: Annotated[
        bool,
        typer.Option(
            "--snapshot-export/--no-snapshot-export",
            help="Export snapshots to gzip CSV on day rollover",
        ),
    ] = True,
    snapshot_export_format: Annotated[
        str,
        typer.Option(
            "--snapshot-export-format",
            help="Snapshot export format: csv_gz (default)",
        ),
    ] = "csv_gz",
    retention_days: Annotated[
        int | None,
        typer.Option(
            "--retention-days",
            help="Delete snapshots older than N days after export (off by default)",
        ),
    ] = None,
) -> None:
    """Run continuous snapshot capture daemon.

    Production-grade data capture with:
    - Resilient WSS connection with REST fallback
    - Application-level keepalive ("PING") for stable long-lived WSS (Phase 5.3)
    - Health-gated fallback: REST only when WSS unhealthy OR cache missing (Phase 5.4)
    - Quiet markets use cached data without REST fallback (event-driven WSS model)
    - REST reconciliation sampler for drift detection (Phase 5.4)
    - Coverage tracking per tick (wss_hits, rest_fallbacks, stale, missing)
    - Daily export artifacts (CSV, JSON, markdown)
    - Daily snapshot exports to gzip CSV (Phase 5.2)
    - Optional retention cleanup for old snapshots (Phase 5.2)
    - Clean shutdown on SIGINT/SIGTERM

    Artifacts exported daily (UTC rollover):
    - exports/ticks_YYYY-MM-DD.csv - Per-tick history + reconciliation stats
    - exports/coverage_YYYY-MM-DD.json - Coverage + drift stats
    - exports/daemon_summary_YYYY-MM-DD.md - Human-readable summary
    - exports/snapshots_YYYY-MM-DD.csv.gz - Snapshot data (Phase 5.2)

    Example:
        pmq ops daemon --interval 60 --limit 200
        pmq ops daemon --orderbook-source rest --max-hours 24
        pmq ops daemon --export-dir ./data/exports
        pmq ops daemon --retention-days 30  # Keep 30 days of snapshots
        pmq ops daemon --wss-health-timeout 60 --max-book-age 1800
        pmq ops daemon --reconcile-sample 10 --reconcile-min-age 300
    """
    import asyncio

    from pmq.ops.daemon import DaemonConfig, DaemonRunner, setup_signal_handlers

    # Validate orderbook-source
    if orderbook_source not in ("rest", "wss"):
        console.print(
            f"[red]Invalid --orderbook-source: {orderbook_source}. Must be 'rest' or 'wss'.[/red]"
        )
        raise typer.Exit(1)

    # Validate snapshot-export-format
    if snapshot_export_format not in ("csv_gz",):
        console.print(
            f"[red]Invalid --snapshot-export-format: {snapshot_export_format}. Must be 'csv_gz'.[/red]"
        )
        raise typer.Exit(1)

    # Create config
    config = DaemonConfig(
        interval_seconds=interval,
        limit=limit,
        orderbook_source=orderbook_source,
        wss_staleness_seconds=wss_staleness_seconds,
        max_hours=max_hours,
        export_dir=export_dir,
        with_orderbook=with_orderbook,
        snapshot_export=snapshot_export,
        snapshot_export_format=snapshot_export_format,
        retention_days=retention_days,
        # Phase 5.4: Health-gated fallback settings
        wss_health_timeout=wss_health_timeout,
        max_book_age=max_book_age,
        reconcile_sample=reconcile_sample,
        reconcile_min_age=reconcile_min_age,
        reconcile_timeout=reconcile_timeout,
    )

    # Initialize dependencies
    dao = DAO()
    gamma_client = GammaClient()

    # OrderBook fetcher for REST fallback
    ob_fetcher = None
    if with_orderbook:
        from pmq.markets.orderbook import OrderBookFetcher

        ob_fetcher = OrderBookFetcher()

    # WSS client for streaming
    wss_client = None
    if with_orderbook and orderbook_source == "wss":
        from pmq.markets.wss_market import MarketWssClient

        # Phase 5.4: Configure health timeout
        wss_client = MarketWssClient(
            health_timeout_seconds=wss_health_timeout,
        )

    # Create runner
    runner = DaemonRunner(
        config=config,
        dao=dao,
        gamma_client=gamma_client,
        wss_client=wss_client,
        ob_fetcher=ob_fetcher,
    )

    # Setup signal handlers
    setup_signal_handlers(runner)

    # Print startup info
    retention_str = f"{retention_days} days" if retention_days else "disabled"
    console.print(
        f"[bold green]Starting ops daemon (Phase 5.4)[/bold green]\n"
        f"Interval: {interval}s\n"
        f"Limit: {limit} markets\n"
        f"Order Book Source: [cyan]{orderbook_source.upper()}[/cyan]\n"
        f"WSS Health Timeout: {wss_health_timeout:.0f}s\n"
        f"Max Book Age: {max_book_age:.0f}s\n"
        f"Reconcile: {reconcile_sample} samples, min age {reconcile_min_age:.0f}s\n"
        f"Max Hours: {max_hours or 'infinite'}\n"
        f"Export Dir: {export_dir}\n"
        f"Snapshot Export: {'enabled' if snapshot_export else 'disabled'}\n"
        f"Retention: {retention_str}\n"
    )

    # Count active markets
    try:
        markets = gamma_client.list_markets(limit=limit)
        active_count = sum(1 for m in markets if m.active and not m.closed)
        console.print(f"[cyan]Markets available: {len(markets)} ({active_count} active)[/cyan]\n")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch initial market count: {e}[/yellow]\n")

    console.print("[dim]Press Ctrl+C to stop gracefully[/dim]\n")

    # Run async daemon
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")

    console.print("[green]Daemon stopped[/green]")


@ops_app.command("status")
def ops_status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Show operator status: DB stats, latest snapshots, daemon state.

    Displays:
    - Total snapshot count and latest snapshot time
    - Latest daemon tick timestamp (from runtime_state)
    - Coverage statistics from most recent export

    Example:
        pmq ops status
        pmq ops status --json
    """
    import json as json_mod
    from pathlib import Path

    dao = DAO()

    # Get snapshot stats
    total_snapshots = dao.count_snapshots()
    latest_snapshot_time = dao.get_latest_snapshot_time()

    # Get daemon state from runtime_state
    daemon_last_tick = dao.get_runtime_state("daemon_last_tick")
    daemon_total_ticks = dao.get_runtime_state("daemon_total_ticks")
    daemon_last_error = dao.get_runtime_state("daemon_last_error")

    # Try to find latest coverage file
    export_dir = Path("exports")
    latest_coverage: dict[str, Any] | None = None
    if export_dir.exists():
        coverage_files = sorted(export_dir.glob("coverage_*.json"), reverse=True)
        if coverage_files:
            try:
                with open(coverage_files[0]) as f:
                    latest_coverage = json_mod.load(f)
            except Exception:
                pass

    status = {
        "snapshots": {
            "total_count": total_snapshots,
            "latest_time": latest_snapshot_time,
        },
        "daemon": {
            "last_tick": daemon_last_tick,
            "total_ticks": int(daemon_total_ticks) if daemon_total_ticks else None,
            "last_error": daemon_last_error,
        },
        "latest_coverage": latest_coverage,
    }

    if json_output:
        console.print(json_mod.dumps(status, indent=2))
    else:
        console.print("[bold]Polymarket Ops Status[/bold]\n")

        # Snapshots section
        console.print("[cyan]Snapshots:[/cyan]")
        console.print(f"  Total: {total_snapshots:,}")
        console.print(f"  Latest: {latest_snapshot_time or 'none'}")

        # Daemon section
        console.print("\n[cyan]Daemon:[/cyan]")
        console.print(f"  Last tick: {daemon_last_tick or 'never'}")
        console.print(f"  Total ticks: {daemon_total_ticks or 'n/a'}")
        if daemon_last_error:
            console.print(f"  [yellow]Last error: {daemon_last_error}[/yellow]")

        # Coverage section
        if latest_coverage:
            console.print(
                f"\n[cyan]Latest Coverage ({latest_coverage.get('date', 'unknown')}):[/cyan]"
            )
            console.print(f"  Ticks: {latest_coverage.get('total_ticks', 0):,}")
            console.print(f"  Snapshots: {latest_coverage.get('total_snapshots', 0):,}")
            console.print(f"  WSS hits: {latest_coverage.get('wss_hits', 0):,}")
            console.print(f"  REST fallbacks: {latest_coverage.get('rest_fallbacks', 0):,}")
            wss_pct = latest_coverage.get("wss_coverage_pct", 0.0)
            console.print(f"  WSS coverage: {wss_pct:.1f}%")
        else:
            console.print("\n[dim]No coverage data found in exports/[/dim]")


if __name__ == "__main__":
    app()
