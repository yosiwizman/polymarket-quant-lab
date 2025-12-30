"""Deterministic backtest engine for replaying historical market data.

The engine replays market snapshots chronologically and simulates trades
without calling any live APIs. Results are deterministic - the same input
always produces the same output.
"""

from dataclasses import dataclass, field
from typing import Any

from pmq.logging import get_logger
from pmq.models import Outcome, Side

logger = get_logger("backtest.engine")


@dataclass
class BacktestPosition:
    """Position state during backtest."""

    market_id: str
    yes_quantity: float = 0.0
    no_quantity: float = 0.0
    avg_price_yes: float = 0.0
    avg_price_no: float = 0.0
    realized_pnl: float = 0.0

    def update_buy(self, outcome: Outcome, price: float, quantity: float) -> None:
        """Update position for a buy trade.

        Args:
            outcome: YES or NO
            price: Trade price
            quantity: Trade quantity
        """
        if outcome == Outcome.YES:
            new_qty = self.yes_quantity + quantity
            if new_qty > 0:
                self.avg_price_yes = (
                    self.avg_price_yes * self.yes_quantity + price * quantity
                ) / new_qty
            self.yes_quantity = new_qty
        else:
            new_qty = self.no_quantity + quantity
            if new_qty > 0:
                self.avg_price_no = (
                    self.avg_price_no * self.no_quantity + price * quantity
                ) / new_qty
            self.no_quantity = new_qty

    def update_sell(self, outcome: Outcome, price: float, quantity: float) -> None:
        """Update position for a sell trade.

        Args:
            outcome: YES or NO
            price: Trade price
            quantity: Trade quantity
        """
        if outcome == Outcome.YES:
            if self.yes_quantity > 0:
                sell_qty = min(quantity, self.yes_quantity)
                pnl = (price - self.avg_price_yes) * sell_qty
                self.realized_pnl += pnl
            self.yes_quantity = max(0, self.yes_quantity - quantity)
        else:
            if self.no_quantity > 0:
                sell_qty = min(quantity, self.no_quantity)
                pnl = (price - self.avg_price_no) * sell_qty
                self.realized_pnl += pnl
            self.no_quantity = max(0, self.no_quantity - quantity)

    def unrealized_pnl(self, yes_price: float, no_price: float) -> float:
        """Calculate unrealized PnL at given prices."""
        yes_pnl = (yes_price - self.avg_price_yes) * self.yes_quantity
        no_pnl = (no_price - self.avg_price_no) * self.no_quantity
        return yes_pnl + no_pnl

    def total_value(self, yes_price: float, no_price: float) -> float:
        """Calculate total position value at given prices."""
        return self.yes_quantity * yes_price + self.no_quantity * no_price


@dataclass
class BacktestTrade:
    """A trade executed during backtest."""

    market_id: str
    side: Side
    outcome: Outcome
    price: float
    quantity: float
    notional: float
    trade_time: str


@dataclass
class BacktestEngine:
    """Deterministic backtest engine.

    Replays market snapshots and simulates trades without calling live APIs.
    All operations are deterministic - same input produces same output.

    Attributes:
        initial_balance: Starting cash balance
        balance: Current cash balance
        positions: Dict of market_id -> BacktestPosition
        trades: List of executed trades
        equity_curve: List of (time, equity) tuples
    """

    initial_balance: float = 10000.0
    balance: float = field(init=False)
    positions: dict[str, BacktestPosition] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    _current_time: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Initialize balance from initial_balance."""
        self.balance = self.initial_balance

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._current_time = ""

    def set_time(self, timestamp: str) -> None:
        """Set current simulation time.

        Args:
            timestamp: Current timestamp (ISO format)
        """
        self._current_time = timestamp

    def execute_trade(
        self,
        market_id: str,
        side: Side,
        outcome: Outcome,
        price: float,
        quantity: float,
    ) -> BacktestTrade | None:
        """Execute a simulated trade.

        Args:
            market_id: Market identifier
            side: BUY or SELL
            outcome: YES or NO
            price: Trade price
            quantity: Trade quantity

        Returns:
            BacktestTrade if executed, None if insufficient funds
        """
        notional = price * quantity

        # Check funds for buy
        if side == Side.BUY and notional > self.balance:
            logger.debug(f"Insufficient funds: need ${notional:.2f}, have ${self.balance:.2f}")
            return None

        # Get or create position
        if market_id not in self.positions:
            self.positions[market_id] = BacktestPosition(market_id=market_id)
        position = self.positions[market_id]

        # Update balance and position
        if side == Side.BUY:
            self.balance -= notional
            position.update_buy(outcome, price, quantity)
        else:
            self.balance += notional
            position.update_sell(outcome, price, quantity)

        # Record trade
        trade = BacktestTrade(
            market_id=market_id,
            side=side,
            outcome=outcome,
            price=price,
            quantity=quantity,
            notional=notional,
            trade_time=self._current_time,
        )
        self.trades.append(trade)

        logger.debug(
            f"Backtest trade: {side.value} {quantity} {outcome.value} @ {price:.4f} "
            f"on {market_id[:8]}..."
        )

        return trade

    def execute_arb_trade(
        self,
        market_id: str,
        yes_price: float,
        no_price: float,
        quantity: float,
    ) -> tuple[BacktestTrade | None, BacktestTrade | None]:
        """Execute arbitrage trade (buy both YES and NO).

        Args:
            market_id: Market identifier
            yes_price: YES token price
            no_price: NO token price
            quantity: Quantity for each side

        Returns:
            Tuple of (YES trade, NO trade), either may be None
        """
        total_notional = (yes_price + no_price) * quantity
        if total_notional > self.balance:
            return None, None

        yes_trade = self.execute_trade(
            market_id=market_id,
            side=Side.BUY,
            outcome=Outcome.YES,
            price=yes_price,
            quantity=quantity,
        )

        no_trade = self.execute_trade(
            market_id=market_id,
            side=Side.BUY,
            outcome=Outcome.NO,
            price=no_price,
            quantity=quantity,
        )

        return yes_trade, no_trade

    def record_equity(self, market_prices: dict[str, dict[str, float]]) -> None:
        """Record current equity point.

        Args:
            market_prices: Dict of market_id -> {yes_price, no_price}
        """
        equity = self.calculate_equity(market_prices)
        self.equity_curve.append((self._current_time, equity))

    def calculate_equity(self, market_prices: dict[str, dict[str, float]]) -> float:
        """Calculate total equity at current prices.

        Args:
            market_prices: Dict of market_id -> {yes_price, no_price}

        Returns:
            Total equity (cash + position values)
        """
        equity = self.balance

        for market_id, position in self.positions.items():
            if market_id in market_prices:
                prices = market_prices[market_id]
                equity += position.total_value(
                    prices.get("yes_price", 0),
                    prices.get("no_price", 0),
                )

        return equity

    def get_state(self) -> dict[str, Any]:
        """Get current engine state.

        Returns:
            Dict with balance, positions, trade count, equity
        """
        return {
            "balance": self.balance,
            "position_count": len(
                [p for p in self.positions.values() if p.yes_quantity > 0 or p.no_quantity > 0]
            ),
            "trade_count": len(self.trades),
            "total_notional": sum(t.notional for t in self.trades),
        }
