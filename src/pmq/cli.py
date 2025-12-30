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
from pmq.logging import setup_logging
from pmq.models import SignalType
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
                console.print(f"[red]Unknown strategy: {strategy}. Use: arb, statarb, observer[/red]")
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
) -> None:
    """Run automated snapshot collection loop.

    Collects market snapshots at regular intervals for backtesting.
    Does NOT execute any trades - data capture only.

    Example:
        pmq snapshots run --interval 60 --limit 200 --duration-minutes 60
    """
    import os

    settings = get_settings()

    # Check kill switch
    if settings.safety.kill_switch or os.environ.get("PMQ_SNAPSHOT_KILL", "").lower() == "true":
        console.print("[red]Kill switch is active. Snapshot collection halted.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Starting snapshot collection[/bold green]\n"
        f"Interval: {interval}s\n"
        f"Limit: {limit} markets\n"
        f"Duration: {'infinite' if duration_minutes == 0 else f'{duration_minutes} minutes'}\n"
    )

    dao = DAO()
    client = GammaClient()

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60) if duration_minutes > 0 else float("inf")
    cycle_count = 0
    backoff = 1
    max_backoff = 300

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

                # Save snapshots
                snapshot_time = datetime.now(UTC).isoformat()
                snapshot_count = dao.save_snapshots_bulk(markets, snapshot_time)

                console.print(
                    f"[green]✓ Saved {snapshot_count} snapshots at {snapshot_time[:19]}[/green]"
                )

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

    elapsed = int((time.time() - start_time) / 60)
    console.print(f"\n[bold]Completed {cycle_count} cycles in {elapsed} minutes[/bold]")


@snapshots_app.command("quality")
def snapshots_quality(
    from_date: Annotated[
        str,
        typer.Option("--from", help="Start date (YYYY-MM-DD or ISO)"),
    ],
    to_date: Annotated[
        str,
        typer.Option("--to", help="End date (YYYY-MM-DD or ISO)"),
    ],
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Expected interval in seconds"),
    ] = 60,
) -> None:
    """Analyze snapshot data quality for a time window.

    Checks for gaps, duplicates, and coverage issues.
    Saves a quality report to the database.

    Example:
        pmq snapshots quality --from 2024-01-01 --to 2024-01-07 --interval 60
    """
    from pmq.quality import QualityReporter

    reporter = QualityReporter()

    with console.status("[bold green]Analyzing snapshot quality..."):
        result = reporter.generate_report(
            start_time=from_date,
            end_time=to_date,
            expected_interval_seconds=interval,
            save=True,
        )

    # Display results
    status = reporter.get_status_badge(result)
    status_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(status, "dim")

    console.print(
        f"\n[bold]Quality Report: [{status_color}]{status.upper()}[/{status_color}][/bold]"
    )

    table = Table(title="Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Window", f"{from_date} to {to_date}")
    table.add_row("Expected Interval", f"{interval}s")
    table.add_row("Total Snapshots", str(result.snapshots_written))
    table.add_row("Markets Covered", str(result.markets_seen))
    table.add_row("Coverage", f"{result.coverage_pct:.1f}%")
    table.add_row("Missing Intervals", str(result.missing_intervals))
    table.add_row("Largest Gap", f"{result.largest_gap_seconds:.0f}s")
    table.add_row("Duplicates", str(result.duplicate_count))

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
    )

    # Display results
    mode_suffix = " (validation mode)" if validation_mode else ""
    status = "[green]PASSED[/green]" if scorecard.passed else "[red]FAILED[/red]"
    console.print(f"\n[bold]Strategy Evaluation: {status}{mode_suffix}[/bold]")
    console.print(f"Run ID: [cyan]{run_id}[/cyan]")
    console.print(f"Strategy: [cyan]{run['strategy']}[/cyan]")
    console.print(f"Data Quality Status: [cyan]{quality_report.status}[/cyan]")

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


if __name__ == "__main__":
    app()
