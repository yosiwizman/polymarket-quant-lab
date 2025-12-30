"""Backtest runner orchestrates strategy execution.

The runner loads historical snapshots, replays them chronologically,
and executes strategies via the BacktestEngine. It ensures determinism
by processing data in a fixed order.
"""

import json
import uuid
from typing import Any

from pmq.backtest.engine import BacktestEngine
from pmq.backtest.metrics import BacktestMetrics, MetricsCalculator
from pmq.config import ArbitrageConfig, get_settings
from pmq.logging import get_logger
from pmq.models import Outcome, Side
from pmq.storage.dao import DAO

logger = get_logger("backtest.runner")


class BacktestRunner:
    """Orchestrates backtest execution.

    Loads snapshots, replays chronologically, and runs strategies.
    Ensures deterministic results by processing in fixed order.
    """

    def __init__(
        self,
        dao: DAO | None = None,
        initial_balance: float = 10000.0,
    ) -> None:
        """Initialize runner.

        Args:
            dao: Data access object
            initial_balance: Starting balance for backtest
        """
        self._dao = dao or DAO()
        self._initial_balance = initial_balance
        self._engine: BacktestEngine | None = None
        self._metrics_calc = MetricsCalculator()

    def run_arb_backtest(
        self,
        start_date: str,
        end_date: str,
        quantity: float = 10.0,
        threshold: float | None = None,
        min_liquidity: float | None = None,
    ) -> tuple[str, BacktestMetrics]:
        """Run arbitrage strategy backtest.

        Args:
            start_date: Start date (ISO format or YYYY-MM-DD)
            end_date: End date (ISO format or YYYY-MM-DD)
            quantity: Trade quantity per signal
            threshold: Arbitrage threshold (default from config)
            min_liquidity: Minimum liquidity (default from config)

        Returns:
            Tuple of (run_id, metrics)
        """
        # Normalize dates
        start_date = self._normalize_date(start_date, start=True)
        end_date = self._normalize_date(end_date, start=False)

        # Get config
        settings = get_settings()
        arb_config = ArbitrageConfig(
            threshold=threshold or settings.arbitrage.threshold,
            min_liquidity=min_liquidity or settings.arbitrage.min_liquidity,
        )

        # Generate run ID
        run_id = str(uuid.uuid4())

        # Store config for reproducibility
        config_json = json.dumps(
            {
                "strategy": "arb",
                "quantity": quantity,
                "threshold": arb_config.threshold,
                "min_liquidity": arb_config.min_liquidity,
                "initial_balance": self._initial_balance,
            }
        )

        # Create run record
        self._dao.create_backtest_run(
            run_id=run_id,
            strategy="arb",
            start_date=start_date,
            end_date=end_date,
            initial_balance=self._initial_balance,
            config_json=config_json,
        )

        logger.info(f"Starting backtest {run_id}: arb from {start_date} to {end_date}")

        # Initialize engine
        self._engine = BacktestEngine(initial_balance=self._initial_balance)

        try:
            # Load snapshots
            snapshot_times = self._dao.get_snapshot_times(start_date, end_date)

            if not snapshot_times:
                logger.warning(f"No snapshots found for {start_date} to {end_date}")
                self._dao.complete_backtest_run(run_id, self._initial_balance, status="failed")
                # Return empty metrics
                metrics = BacktestMetrics(
                    total_pnl=0,
                    max_drawdown=0,
                    win_rate=0,
                    sharpe_ratio=0,
                    total_trades=0,
                    trades_per_day=0,
                    capital_utilization=0,
                    total_notional=0,
                    final_balance=self._initial_balance,
                    peak_equity=self._initial_balance,
                    lowest_equity=self._initial_balance,
                )
                return run_id, metrics

            logger.info(f"Found {len(snapshot_times)} snapshot times")

            # Replay each snapshot time
            for snapshot_time in snapshot_times:
                self._engine.set_time(snapshot_time)

                # Get all snapshots at this time
                snapshots = self._dao.get_snapshots(snapshot_time, snapshot_time)

                # Build market prices dict for equity calculation
                market_prices: dict[str, dict[str, float]] = {}
                for snap in snapshots:
                    market_prices[snap["market_id"]] = {
                        "yes_price": snap["yes_price"],
                        "no_price": snap["no_price"],
                    }

                # Scan for arbitrage signals
                for snap in snapshots:
                    # Check arb condition
                    yes_price = snap["yes_price"]
                    no_price = snap["no_price"]
                    combined = yes_price + no_price
                    liquidity = snap.get("liquidity", 0)

                    if combined < arb_config.threshold and liquidity >= arb_config.min_liquidity:
                        # Execute arb trade
                        yes_trade, no_trade = self._engine.execute_arb_trade(
                            market_id=snap["market_id"],
                            yes_price=yes_price,
                            no_price=no_price,
                            quantity=quantity,
                        )

                        # Save trades to DB
                        if yes_trade:
                            self._dao.save_backtest_trade(
                                run_id=run_id,
                                market_id=yes_trade.market_id,
                                side=yes_trade.side.value,
                                outcome=yes_trade.outcome.value,
                                price=yes_trade.price,
                                quantity=yes_trade.quantity,
                                trade_time=yes_trade.trade_time,
                            )
                        if no_trade:
                            self._dao.save_backtest_trade(
                                run_id=run_id,
                                market_id=no_trade.market_id,
                                side=no_trade.side.value,
                                outcome=no_trade.outcome.value,
                                price=no_trade.price,
                                quantity=no_trade.quantity,
                                trade_time=no_trade.trade_time,
                            )

                # Record equity at this time
                self._engine.record_equity(market_prices)

            # Calculate metrics
            metrics = self._metrics_calc.calculate(self._engine, start_date, end_date)

            # Save metrics
            self._dao.save_backtest_metrics(
                run_id=run_id,
                total_pnl=metrics.total_pnl,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                sharpe_ratio=metrics.sharpe_ratio,
                total_trades=metrics.total_trades,
                trades_per_day=metrics.trades_per_day,
                capital_utilization=metrics.capital_utilization,
                extra_metrics={
                    "total_notional": metrics.total_notional,
                    "final_balance": metrics.final_balance,
                    "peak_equity": metrics.peak_equity,
                    "lowest_equity": metrics.lowest_equity,
                },
            )

            # Complete run
            self._dao.complete_backtest_run(run_id, metrics.final_balance, status="completed")

            logger.info(
                f"Backtest {run_id} completed: "
                f"PnL=${metrics.total_pnl:.2f}, trades={metrics.total_trades}"
            )

            return run_id, metrics

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            self._dao.complete_backtest_run(
                run_id,
                self._engine.balance if self._engine else self._initial_balance,
                status="failed",
            )
            raise

    def run_statarb_backtest(
        self,
        start_date: str,
        end_date: str,
        pairs_config: str | None = None,
        quantity: float = 10.0,
    ) -> tuple[str, BacktestMetrics]:
        """Run stat-arb strategy backtest.

        Args:
            start_date: Start date (ISO format or YYYY-MM-DD)
            end_date: End date (ISO format or YYYY-MM-DD)
            pairs_config: Path to pairs config file
            quantity: Trade quantity per signal

        Returns:
            Tuple of (run_id, metrics)
        """
        # Normalize dates
        start_date = self._normalize_date(start_date, start=True)
        end_date = self._normalize_date(end_date, start=False)

        # Generate run ID
        run_id = str(uuid.uuid4())

        # Store config
        config_json = json.dumps(
            {
                "strategy": "statarb",
                "quantity": quantity,
                "pairs_config": pairs_config,
                "initial_balance": self._initial_balance,
            }
        )

        # Create run record
        self._dao.create_backtest_run(
            run_id=run_id,
            strategy="statarb",
            start_date=start_date,
            end_date=end_date,
            initial_balance=self._initial_balance,
            config_json=config_json,
        )

        logger.info(f"Starting backtest {run_id}: statarb from {start_date} to {end_date}")

        # Initialize engine
        self._engine = BacktestEngine(initial_balance=self._initial_balance)

        try:
            # Load pairs from config
            from pmq.config import StatArbConfig
            from pmq.strategies.statarb import StatArbScanner

            # Create scanner - uses default config if pairs_config is None
            if pairs_config:
                statarb_config = StatArbConfig(pairs_file=pairs_config)
                scanner = StatArbScanner(config=statarb_config)
            else:
                scanner = StatArbScanner()

            if not scanner.pairs:
                logger.warning("No stat-arb pairs configured")
                self._dao.complete_backtest_run(run_id, self._initial_balance, status="failed")
                metrics = BacktestMetrics(
                    total_pnl=0,
                    max_drawdown=0,
                    win_rate=0,
                    sharpe_ratio=0,
                    total_trades=0,
                    trades_per_day=0,
                    capital_utilization=0,
                    total_notional=0,
                    final_balance=self._initial_balance,
                    peak_equity=self._initial_balance,
                    lowest_equity=self._initial_balance,
                )
                return run_id, metrics

            # Get market IDs from pairs
            pair_market_ids = set()
            for pair in scanner.pairs:
                pair_market_ids.add(pair.market_a_id)
                pair_market_ids.add(pair.market_b_id)

            # Load snapshots
            snapshot_times = self._dao.get_snapshot_times(start_date, end_date)

            if not snapshot_times:
                logger.warning(f"No snapshots found for {start_date} to {end_date}")
                self._dao.complete_backtest_run(run_id, self._initial_balance, status="failed")
                metrics = BacktestMetrics(
                    total_pnl=0,
                    max_drawdown=0,
                    win_rate=0,
                    sharpe_ratio=0,
                    total_trades=0,
                    trades_per_day=0,
                    capital_utilization=0,
                    total_notional=0,
                    final_balance=self._initial_balance,
                    peak_equity=self._initial_balance,
                    lowest_equity=self._initial_balance,
                )
                return run_id, metrics

            logger.info(f"Found {len(snapshot_times)} snapshot times")

            # Replay each snapshot time
            for snapshot_time in snapshot_times:
                self._engine.set_time(snapshot_time)

                # Get snapshots for pair markets
                snapshots = self._dao.get_snapshots(
                    snapshot_time, snapshot_time, list(pair_market_ids)
                )

                # Build market prices dict
                market_prices: dict[str, dict[str, float]] = {}
                for snap in snapshots:
                    market_prices[snap["market_id"]] = {
                        "yes_price": snap["yes_price"],
                        "no_price": snap["no_price"],
                    }

                # Check each pair for signals
                for pair in scanner.pairs:
                    if pair.market_a_id not in market_prices:
                        continue
                    if pair.market_b_id not in market_prices:
                        continue

                    price_a = market_prices[pair.market_a_id]["yes_price"]
                    price_b = market_prices[pair.market_b_id]["yes_price"]

                    # Calculate spread
                    spread = price_a - price_b

                    # Check entry threshold
                    settings = get_settings()
                    if abs(spread) > settings.statarb.entry_threshold:
                        # Execute stat-arb trade
                        if spread > 0:
                            # Long B, Short A (buy B YES, sell A YES if we have it)
                            self._engine.execute_trade(
                                market_id=pair.market_b_id,
                                side=Side.BUY,
                                outcome=Outcome.YES,
                                price=price_b,
                                quantity=quantity,
                            )
                        else:
                            # Long A, Short B
                            self._engine.execute_trade(
                                market_id=pair.market_a_id,
                                side=Side.BUY,
                                outcome=Outcome.YES,
                                price=price_a,
                                quantity=quantity,
                            )

                # Record equity
                self._engine.record_equity(market_prices)

            # Calculate metrics
            metrics = self._metrics_calc.calculate(self._engine, start_date, end_date)

            # Save metrics
            self._dao.save_backtest_metrics(
                run_id=run_id,
                total_pnl=metrics.total_pnl,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                sharpe_ratio=metrics.sharpe_ratio,
                total_trades=metrics.total_trades,
                trades_per_day=metrics.trades_per_day,
                capital_utilization=metrics.capital_utilization,
            )

            # Complete run
            self._dao.complete_backtest_run(run_id, metrics.final_balance, status="completed")

            logger.info(
                f"Backtest {run_id} completed: "
                f"PnL=${metrics.total_pnl:.2f}, trades={metrics.total_trades}"
            )

            return run_id, metrics

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            self._dao.complete_backtest_run(
                run_id,
                self._engine.balance if self._engine else self._initial_balance,
                status="failed",
            )
            raise

    def get_run_report(self, run_id: str) -> dict[str, Any] | None:
        """Get full report for a backtest run.

        Args:
            run_id: Backtest run ID

        Returns:
            Dict with run info, metrics, and trades
        """
        run = self._dao.get_backtest_run(run_id)
        if not run:
            return None

        metrics = self._dao.get_backtest_metrics(run_id)
        trades = self._dao.get_backtest_trades(run_id, limit=100)

        return {
            "run": run,
            "metrics": metrics,
            "trades": trades,
            "trade_count": len(trades),
        }

    def list_runs(
        self,
        strategy: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List backtest runs.

        Args:
            strategy: Optional strategy filter
            limit: Maximum results

        Returns:
            List of run summaries
        """
        return self._dao.get_backtest_runs(strategy=strategy, limit=limit)

    def _normalize_date(self, date_str: str, start: bool = True) -> str:
        """Normalize date string to ISO format.

        Args:
            date_str: Date string (YYYY-MM-DD or ISO)
            start: If True, use start of day; else end of day

        Returns:
            ISO formatted datetime string
        """
        # If already has time component, return as-is
        if "T" in date_str:
            return date_str

        # Add time component
        if start:
            return f"{date_str}T00:00:00"
        else:
            return f"{date_str}T23:59:59"
