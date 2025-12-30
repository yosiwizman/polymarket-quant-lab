"""Paper trading ledger with safety guardrails.

Simulates trades and tracks virtual positions without real money.
Includes safety mechanisms: position limits, notional caps, rate limits, kill switch.
"""

from datetime import datetime, timezone
from typing import Any

from pmq.config import SafetyConfig, get_settings
from pmq.logging import get_logger, log_trade_event
from pmq.models import ArbitrageSignal, Outcome, PaperPosition, PaperTrade, Side
from pmq.storage.dao import DAO
from pmq.storage.db import get_database

logger = get_logger("strategies.paper")


class SafetyError(Exception):
    """Raised when a safety check fails."""


class SafetyGuard:
    """Safety guardrails for paper trading.

    Enforces limits on positions, notional values, and trade rates.
    Provides a kill switch to halt all trading.
    """

    def __init__(
        self,
        config: SafetyConfig | None = None,
        dao: DAO | None = None,
    ) -> None:
        """Initialize safety guard.

        Args:
            config: Safety configuration
            dao: Data access object
        """
        self._config = config or get_settings().safety
        self._dao = dao or DAO()
        logger.debug(
            f"SafetyGuard initialized: max_positions={self._config.max_positions}, "
            f"max_notional={self._config.max_notional_per_market}, "
            f"max_trades_per_hour={self._config.max_trades_per_hour}"
        )

    @property
    def kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._config.kill_switch

    def check_kill_switch(self) -> None:
        """Check if kill switch is active.

        Raises:
            SafetyError: If kill switch is active
        """
        if self._config.kill_switch:
            self._dao.log_audit("KILL_SWITCH_BLOCKED")
            raise SafetyError("Kill switch is active - all trading halted")

    def check_position_limit(self) -> None:
        """Check if position limit is exceeded.

        Raises:
            SafetyError: If max positions reached
        """
        current = self._dao.count_positions()
        if current >= self._config.max_positions:
            self._dao.log_audit(
                "POSITION_LIMIT_BLOCKED",
                details={"current": current, "max": self._config.max_positions},
            )
            raise SafetyError(
                f"Position limit reached: {current}/{self._config.max_positions}"
            )

    def check_notional_limit(self, market_id: str, additional_notional: float) -> None:
        """Check if notional limit for market would be exceeded.

        Args:
            market_id: Market identifier
            additional_notional: Notional value to add

        Raises:
            SafetyError: If notional limit would be exceeded
        """
        # Calculate current notional in market
        position = self._dao.get_position(market_id)
        current_notional = 0.0
        if position:
            current_notional = (
                position.yes_quantity * position.avg_price_yes
                + position.no_quantity * position.avg_price_no
            )

        new_total = current_notional + additional_notional
        if new_total > self._config.max_notional_per_market:
            self._dao.log_audit(
                "NOTIONAL_LIMIT_BLOCKED",
                market_id=market_id,
                details={
                    "current": current_notional,
                    "additional": additional_notional,
                    "max": self._config.max_notional_per_market,
                },
            )
            raise SafetyError(
                f"Notional limit would be exceeded: "
                f"${new_total:.2f} > ${self._config.max_notional_per_market:.2f}"
            )

    def check_rate_limit(self) -> None:
        """Check if trade rate limit is exceeded.

        Raises:
            SafetyError: If rate limit exceeded
        """
        recent_trades = self._dao.count_trades_in_window(hours=1)
        if recent_trades >= self._config.max_trades_per_hour:
            self._dao.log_audit(
                "RATE_LIMIT_BLOCKED",
                details={"recent": recent_trades, "max": self._config.max_trades_per_hour},
            )
            raise SafetyError(
                f"Rate limit exceeded: {recent_trades}/{self._config.max_trades_per_hour} trades/hour"
            )

    def validate_trade(self, market_id: str, notional: float) -> None:
        """Run all safety checks for a trade.

        Args:
            market_id: Target market
            notional: Trade notional value

        Raises:
            SafetyError: If any check fails
        """
        self.check_kill_switch()
        self.check_rate_limit()
        self.check_notional_limit(market_id, notional)


class PaperLedger:
    """Paper trading ledger for simulating trades.

    Tracks virtual positions and computes theoretical PnL.
    All trades are simulated - no real orders are placed.
    """

    def __init__(
        self,
        dao: DAO | None = None,
        safety: SafetyGuard | None = None,
    ) -> None:
        """Initialize paper ledger.

        Args:
            dao: Data access object
            safety: Safety guard instance
        """
        self._dao = dao or DAO()
        self._safety = safety or SafetyGuard(dao=self._dao)
        logger.info("PaperLedger initialized")

    @property
    def safety(self) -> SafetyGuard:
        """Get safety guard."""
        return self._safety

    def execute_trade(
        self,
        strategy: str,
        market_id: str,
        market_question: str,
        side: Side,
        outcome: Outcome,
        price: float,
        quantity: float,
    ) -> PaperTrade:
        """Execute a paper trade.

        Args:
            strategy: Strategy name (arb, statarb, manual)
            market_id: Market identifier
            market_question: Market question text
            side: BUY or SELL
            outcome: YES or NO
            price: Execution price
            quantity: Number of shares

        Returns:
            Executed paper trade

        Raises:
            SafetyError: If safety checks fail
        """
        notional = price * quantity

        # Run safety checks
        self._safety.validate_trade(market_id, notional)

        # Create trade
        trade = PaperTrade(
            strategy=strategy,
            market_id=market_id,
            market_question=market_question,
            side=side,
            outcome=outcome,
            price=price,
            quantity=quantity,
            notional=notional,
            created_at=datetime.now(timezone.utc),
        )

        # Save trade
        trade_id = self._dao.save_paper_trade(trade)

        # Update position
        self._update_position(trade)

        # Audit log
        self._dao.log_audit(
            "PAPER_TRADE",
            market_id=market_id,
            details={
                "trade_id": trade_id,
                "strategy": strategy,
                "side": side.value,
                "outcome": outcome.value,
                "price": price,
                "quantity": quantity,
                "notional": notional,
            },
        )

        log_trade_event(
            "PAPER_TRADE_EXECUTED",
            market_id,
            trade_id=trade_id,
            side=side.value,
            outcome=outcome.value,
            price=price,
            quantity=quantity,
        )

        return trade

    def _update_position(self, trade: PaperTrade) -> None:
        """Update position after a trade.

        Args:
            trade: Executed trade
        """
        position = self._dao.get_position(trade.market_id)

        if position is None:
            position = PaperPosition(
                market_id=trade.market_id,
                market_question=trade.market_question,
            )

        # Update quantities and average prices
        if trade.outcome == Outcome.YES:
            if trade.side == Side.BUY:
                # Buying YES
                new_qty = position.yes_quantity + trade.quantity
                if new_qty > 0:
                    position.avg_price_yes = (
                        position.avg_price_yes * position.yes_quantity
                        + trade.price * trade.quantity
                    ) / new_qty
                position.yes_quantity = new_qty
            else:
                # Selling YES
                if position.yes_quantity > 0:
                    # Realize PnL
                    pnl = (trade.price - position.avg_price_yes) * min(
                        trade.quantity, position.yes_quantity
                    )
                    position.realized_pnl += pnl
                position.yes_quantity = max(0, position.yes_quantity - trade.quantity)
        else:
            if trade.side == Side.BUY:
                # Buying NO
                new_qty = position.no_quantity + trade.quantity
                if new_qty > 0:
                    position.avg_price_no = (
                        position.avg_price_no * position.no_quantity
                        + trade.price * trade.quantity
                    ) / new_qty
                position.no_quantity = new_qty
            else:
                # Selling NO
                if position.no_quantity > 0:
                    pnl = (trade.price - position.avg_price_no) * min(
                        trade.quantity, position.no_quantity
                    )
                    position.realized_pnl += pnl
                position.no_quantity = max(0, position.no_quantity - trade.quantity)

        position.updated_at = datetime.now(timezone.utc)
        self._dao.upsert_position(position)

    def execute_arb_trade(
        self,
        signal: ArbitrageSignal,
        quantity: float = 10.0,
    ) -> tuple[PaperTrade, PaperTrade]:
        """Execute arbitrage paper trade (buy both YES and NO).

        Args:
            signal: Arbitrage signal
            quantity: Quantity to trade

        Returns:
            Tuple of (YES trade, NO trade)
        """
        yes_trade = self.execute_trade(
            strategy="arb",
            market_id=signal.market_id,
            market_question=signal.market_question,
            side=Side.BUY,
            outcome=Outcome.YES,
            price=signal.yes_price,
            quantity=quantity,
        )

        no_trade = self.execute_trade(
            strategy="arb",
            market_id=signal.market_id,
            market_question=signal.market_question,
            side=Side.BUY,
            outcome=Outcome.NO,
            price=signal.no_price,
            quantity=quantity,
        )

        return yes_trade, no_trade

    def get_position(self, market_id: str) -> PaperPosition | None:
        """Get current position for a market.

        Args:
            market_id: Market identifier

        Returns:
            Position or None
        """
        return self._dao.get_position(market_id)

    def get_all_positions(self) -> list[PaperPosition]:
        """Get all open positions.

        Returns:
            List of positions with holdings
        """
        return self._dao.get_all_positions()

    def get_trades(
        self,
        strategy: str | None = None,
        limit: int = 100,
    ) -> list[PaperTrade]:
        """Get paper trades.

        Args:
            strategy: Filter by strategy
            limit: Maximum results

        Returns:
            List of trades
        """
        return self._dao.get_paper_trades(strategy=strategy, limit=limit)

    def calculate_pnl(
        self,
        markets_data: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Calculate total PnL across all positions.

        Args:
            markets_data: Optional dict of market_id -> market data for current prices

        Returns:
            PnL summary
        """
        positions = self.get_all_positions()

        total_realized = 0.0
        total_unrealized = 0.0

        position_details = []

        for pos in positions:
            total_realized += pos.realized_pnl

            # Get current prices
            current_yes = pos.avg_price_yes
            current_no = pos.avg_price_no

            if markets_data and pos.market_id in markets_data:
                market = markets_data[pos.market_id]
                current_yes = market.get("last_price_yes", current_yes)
                current_no = market.get("last_price_no", current_no)

            unrealized = pos.unrealized_pnl(current_yes, current_no)
            total_unrealized += unrealized

            position_details.append(
                {
                    "market_id": pos.market_id,
                    "market_question": pos.market_question[:50] + "..."
                    if len(pos.market_question) > 50
                    else pos.market_question,
                    "yes_qty": pos.yes_quantity,
                    "no_qty": pos.no_quantity,
                    "realized_pnl": pos.realized_pnl,
                    "unrealized_pnl": unrealized,
                    "total_pnl": pos.realized_pnl + unrealized,
                }
            )

        return {
            "total_realized_pnl": total_realized,
            "total_unrealized_pnl": total_unrealized,
            "total_pnl": total_realized + total_unrealized,
            "position_count": len(positions),
            "positions": position_details,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get trading statistics.

        Returns:
            Statistics dict
        """
        return self._dao.get_trading_stats()
