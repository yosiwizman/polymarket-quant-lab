"""Backtest metrics calculation.

Computes performance metrics from backtest results:
- Total PnL
- Max drawdown
- Win rate
- Sharpe-like ratio (simplified)
- Trades per day
- Capital utilization
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pmq.backtest.engine import BacktestEngine, BacktestTrade
from pmq.logging import get_logger
from pmq.models import Side

logger = get_logger("backtest.metrics")


@dataclass
class BacktestMetrics:
    """Computed backtest metrics."""

    total_pnl: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    total_trades: int
    trades_per_day: float
    capital_utilization: float
    total_notional: float
    final_balance: float
    peak_equity: float
    lowest_equity: float


class MetricsCalculator:
    """Calculate backtest performance metrics."""

    def __init__(self) -> None:
        """Initialize calculator."""
        pass

    def calculate(
        self,
        engine: BacktestEngine,
        start_date: str,
        end_date: str,
    ) -> BacktestMetrics:
        """Calculate all metrics from backtest results.

        Args:
            engine: Completed backtest engine
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestMetrics with all computed metrics
        """
        # Basic stats
        total_trades = len(engine.trades)
        total_notional = sum(t.notional for t in engine.trades)
        final_balance = engine.balance

        # Calculate PnL
        total_pnl = final_balance - engine.initial_balance

        # Calculate win rate
        win_rate = self._calculate_win_rate(engine.trades)

        # Calculate max drawdown from equity curve
        max_drawdown, peak_equity, lowest_equity = self._calculate_drawdown(engine.equity_curve)

        # Calculate sharpe-like ratio
        sharpe_ratio = self._calculate_sharpe(engine.equity_curve, engine.initial_balance)

        # Calculate trades per day
        trades_per_day = self._calculate_trades_per_day(start_date, end_date, total_trades)

        # Calculate capital utilization
        capital_utilization = self._calculate_capital_utilization(
            engine.equity_curve, engine.initial_balance
        )

        metrics = BacktestMetrics(
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            trades_per_day=trades_per_day,
            capital_utilization=capital_utilization,
            total_notional=total_notional,
            final_balance=final_balance,
            peak_equity=peak_equity,
            lowest_equity=lowest_equity,
        )

        logger.info(
            f"Metrics: PnL=${total_pnl:.2f}, DD={max_drawdown:.2%}, "
            f"WR={win_rate:.2%}, Sharpe={sharpe_ratio:.2f}"
        )

        return metrics

    def _calculate_win_rate(self, trades: list[BacktestTrade]) -> float:
        """Calculate win rate from trades.

        For arbitrage, we consider a 'round trip' profitable if YES+NO < 1.
        Simplified: count profitable buys based on realized gains.

        Args:
            trades: List of backtest trades

        Returns:
            Win rate (0-1)
        """
        if not trades:
            return 0.0

        # Group trades by market
        market_trades: dict[str, list[BacktestTrade]] = {}
        for trade in trades:
            if trade.market_id not in market_trades:
                market_trades[trade.market_id] = []
            market_trades[trade.market_id].append(trade)

        # For arbitrage: count markets where we bought YES+NO for < 1.0
        wins = 0
        total = 0

        for market_id, market_trades_list in market_trades.items():
            # Get all buys
            buys = [t for t in market_trades_list if t.side == Side.BUY]
            if not buys:
                continue

            # Calculate average buy price for arb
            total_yes_cost = sum(t.notional for t in buys if t.outcome.value == "YES")
            total_no_cost = sum(t.notional for t in buys if t.outcome.value == "NO")
            total_yes_qty = sum(t.quantity for t in buys if t.outcome.value == "YES")
            total_no_qty = sum(t.quantity for t in buys if t.outcome.value == "NO")

            if total_yes_qty > 0 and total_no_qty > 0:
                avg_yes = total_yes_cost / total_yes_qty
                avg_no = total_no_cost / total_no_qty
                combined = avg_yes + avg_no

                total += 1
                if combined < 1.0:
                    wins += 1

        return wins / total if total > 0 else 0.0

    def _calculate_drawdown(
        self,
        equity_curve: list[tuple[str, float]],
    ) -> tuple[float, float, float]:
        """Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: List of (time, equity) tuples

        Returns:
            Tuple of (max_drawdown_pct, peak_equity, lowest_equity)
        """
        if not equity_curve:
            return 0.0, 0.0, 0.0

        equities = [e[1] for e in equity_curve]
        peak = equities[0]
        max_drawdown = 0.0
        lowest = equities[0]

        for equity in equities:
            if equity > peak:
                peak = equity
            if equity < lowest:
                lowest = equity

            drawdown = (peak - equity) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown, peak, lowest

    def _calculate_sharpe(
        self,
        equity_curve: list[tuple[str, float]],
        initial_balance: float,
    ) -> float:
        """Calculate simplified Sharpe-like ratio.

        This is a simplified version: (final_return / volatility).
        Not a true Sharpe ratio as we don't have a risk-free rate.

        Args:
            equity_curve: List of (time, equity) tuples
            initial_balance: Starting balance

        Returns:
            Sharpe-like ratio (higher is better)
        """
        if len(equity_curve) < 2:
            return 0.0

        equities = [e[1] for e in equity_curve]

        # Calculate returns
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                ret = (equities[i] - equities[i - 1]) / equities[i - 1]
                returns.append(ret)

        if not returns:
            return 0.0

        # Calculate mean and std of returns
        mean_return = sum(returns) / len(returns)

        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = variance ** 0.5

        if std_return == 0:
            return 0.0

        # Simplified Sharpe (no risk-free rate adjustment)
        sharpe = mean_return / std_return

        return sharpe

    def _calculate_trades_per_day(
        self,
        start_date: str,
        end_date: str,
        total_trades: int,
    ) -> float:
        """Calculate average trades per day.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            total_trades: Total number of trades

        Returns:
            Average trades per day
        """
        try:
            start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            days = (end - start).days
            if days <= 0:
                days = 1
            return total_trades / days
        except (ValueError, TypeError):
            return float(total_trades)

    def _calculate_capital_utilization(
        self,
        equity_curve: list[tuple[str, float]],
        initial_balance: float,
    ) -> float:
        """Calculate average capital utilization.

        This measures how much of the capital was typically deployed.
        Higher values mean more capital at risk.

        Args:
            equity_curve: List of (time, equity) tuples
            initial_balance: Starting balance

        Returns:
            Average capital utilization (0-1)
        """
        if not equity_curve or initial_balance <= 0:
            return 0.0

        # Calculate average equity to initial balance ratio
        equities = [e[1] for e in equity_curve]
        avg_equity = sum(equities) / len(equities)

        # Utilization is how much capital was deployed on average
        # If avg_equity < initial, capital was being used
        # Simplified: use 1 - (avg_cash / initial) approximation
        utilization = 1.0 - (avg_equity / initial_balance)

        # Clamp to 0-1
        return max(0.0, min(1.0, utilization)) if utilization > 0 else 0.0

    def to_dict(self, metrics: BacktestMetrics) -> dict[str, Any]:
        """Convert metrics to dict for storage.

        Args:
            metrics: BacktestMetrics instance

        Returns:
            Dict representation
        """
        return {
            "total_pnl": metrics.total_pnl,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "sharpe_ratio": metrics.sharpe_ratio,
            "total_trades": metrics.total_trades,
            "trades_per_day": metrics.trades_per_day,
            "capital_utilization": metrics.capital_utilization,
            "total_notional": metrics.total_notional,
            "final_balance": metrics.final_balance,
            "peak_equity": metrics.peak_equity,
            "lowest_equity": metrics.lowest_equity,
        }
