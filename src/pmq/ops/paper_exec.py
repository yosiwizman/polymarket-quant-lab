"""Paper execution module for live paper trading loop.

Phase 6.1: Live paper execution loop that:
- Generates signals from real-time snapshots using existing scanners
- Routes through governance/risk gate for approval
- Executes in PaperLedger only (no real orders)
- Tracks per-tick and daily paper trading metrics

HARD RULE: No real order placement or private key logic. PaperLedger only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pmq.logging import get_logger
from pmq.strategies import ArbitrageScanner, PaperLedger, StatArbScanner
from pmq.strategies.paper import SafetyError

if TYPE_CHECKING:
    from pmq.governance import RiskGate
    from pmq.storage import DAO

logger = get_logger("ops.paper_exec")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PaperExecConfig:
    """Configuration for paper execution loop.

    SAFETY: Paper execution is disabled by default and requires explicit
    opt-in. All trades go through PaperLedger only - no real orders.
    """

    enabled: bool = False  # Must be explicitly enabled
    max_trades_per_tick: int = 3  # Limit trades per tick to control velocity
    max_markets_scanned: int = 200  # Limit markets to scan per tick
    min_signal_edge_bps: float = 50.0  # Minimum edge in basis points (0.5%)
    require_approval: bool = True  # Use governance risk gate if present
    trade_quantity: float = 10.0  # Default quantity per trade
    # Scanner config overrides (None = use defaults)
    arb_threshold: float | None = None  # ArbitrageScanner threshold
    min_liquidity: float | None = None  # Minimum liquidity requirement


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class PaperExecResult:
    """Result of paper execution for a single tick.

    Contains counts of signals found, executed, blocked, and errors.
    Also includes PnL snapshot from PaperLedger.
    """

    # Signal/trade counts
    signals_found: int = 0
    signals_evaluated: int = 0
    trades_executed: int = 0
    blocked_by_risk: int = 0
    blocked_by_safety: int = 0
    errors: int = 0

    # PnL snapshot from PaperLedger
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    position_count: int = 0

    # Details for logging
    executed_signals: list[dict[str, Any]] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)


# =============================================================================
# Paper Executor
# =============================================================================


class PaperExecutor:
    """Executes paper trades based on signals from live market snapshots.

    This class:
    1. Runs ArbitrageScanner and optionally StatArbScanner on current snapshots
    2. Filters signals by minimum edge requirement
    3. Applies governance risk gate checks (if enabled)
    4. Executes qualifying trades via PaperLedger ONLY
    5. Returns detailed metrics for monitoring

    HARD RULE: No real order placement. All trades go to PaperLedger.
    """

    def __init__(
        self,
        config: PaperExecConfig,
        dao: DAO,
        paper_ledger: PaperLedger | None = None,
        risk_gate: RiskGate | None = None,
        arb_scanner: ArbitrageScanner | None = None,
        statarb_scanner: StatArbScanner | None = None,
    ) -> None:
        """Initialize paper executor.

        Args:
            config: Paper execution configuration
            dao: Data access object
            paper_ledger: Paper trading ledger (creates new if None)
            risk_gate: Governance risk gate (optional)
            arb_scanner: Arbitrage scanner (creates new if None)
            statarb_scanner: Stat-arb scanner (creates new if None)
        """
        self.config = config
        self._dao = dao
        self._paper_ledger = paper_ledger or PaperLedger(dao=dao)
        self._risk_gate = risk_gate
        self._arb_scanner = arb_scanner or ArbitrageScanner()
        self._statarb_scanner = statarb_scanner or StatArbScanner()

        # Apply config overrides to scanners
        if config.arb_threshold is not None:
            self._arb_scanner._config.threshold = config.arb_threshold
        if config.min_liquidity is not None:
            self._arb_scanner._config.min_liquidity = config.min_liquidity

        logger.info(
            f"PaperExecutor initialized: enabled={config.enabled}, "
            f"max_trades_per_tick={config.max_trades_per_tick}, "
            f"min_edge_bps={config.min_signal_edge_bps}, "
            f"require_approval={config.require_approval}"
        )

    @property
    def paper_ledger(self) -> PaperLedger:
        """Get the paper ledger instance."""
        return self._paper_ledger

    def execute_tick(
        self,
        markets_data: list[dict[str, Any]],
        strategy_name: str = "paper_exec",
        current_prices: dict[str, dict[str, float]] | None = None,
    ) -> PaperExecResult:
        """Execute paper trading for a single tick.

        Args:
            markets_data: List of market dicts from DAO (same format as scan_from_db)
            strategy_name: Strategy name for governance check
            current_prices: Optional dict of market_id -> {yes_price, no_price} for PnL

        Returns:
            PaperExecResult with execution metrics
        """
        result = PaperExecResult()

        if not self.config.enabled:
            # Paper execution disabled - just return PnL snapshot
            pnl = self._paper_ledger.calculate_pnl(markets_data=current_prices)
            result.realized_pnl = pnl["total_realized_pnl"]
            result.unrealized_pnl = pnl["total_unrealized_pnl"]
            result.total_pnl = pnl["total_pnl"]
            result.position_count = pnl["position_count"]
            return result

        # Check governance approval if required
        if self.config.require_approval and self._risk_gate:
            try:
                approval = self._risk_gate.check_approval(strategy_name)
                if not approval.approved:
                    logger.debug(f"Paper execution blocked: {approval.reason}")
                    result.blocked_reasons.append(f"Not approved: {approval.reason}")
                    result.blocked_by_risk += 1
                    # Still return PnL snapshot
                    pnl = self._paper_ledger.calculate_pnl(markets_data=current_prices)
                    result.realized_pnl = pnl["total_realized_pnl"]
                    result.unrealized_pnl = pnl["total_unrealized_pnl"]
                    result.total_pnl = pnl["total_pnl"]
                    result.position_count = pnl["position_count"]
                    return result
            except Exception as e:
                logger.warning(f"Risk gate check failed: {e}")
                result.error_messages.append(f"Risk gate error: {e}")
                result.errors += 1

        # Limit markets scanned
        markets_to_scan = markets_data[: self.config.max_markets_scanned]

        # Run arbitrage scanner
        try:
            arb_signals = self._arb_scanner.scan_from_db(
                markets_to_scan,
                top_n=self.config.max_trades_per_tick * 2,  # Get more than we need
            )
            result.signals_found = len(arb_signals)
        except Exception as e:
            logger.warning(f"Arbitrage scanner failed: {e}")
            result.error_messages.append(f"Scanner error: {e}")
            result.errors += 1
            arb_signals = []

        # Filter by minimum edge
        min_edge = self.config.min_signal_edge_bps / 10000  # Convert bps to decimal
        filtered_signals = [sig for sig in arb_signals if sig.profit_potential >= min_edge]
        result.signals_evaluated = len(filtered_signals)

        # Execute trades up to limit
        trades_this_tick = 0
        for signal in filtered_signals:
            if trades_this_tick >= self.config.max_trades_per_tick:
                break

            # Check risk gate trade limit if available
            if self._risk_gate:
                positions = self._paper_ledger.get_all_positions()
                total_notional = sum(
                    (p.yes_quantity * p.avg_price_yes + p.no_quantity * p.avg_price_no)
                    for p in positions
                )
                trade_notional = self.config.trade_quantity * (signal.yes_price + signal.no_price)

                allowed, reason = self._risk_gate.check_trade_limit(
                    market_id=signal.market_id,
                    notional=trade_notional,
                    current_positions=len(positions),
                    current_total_notional=total_notional,
                )

                if not allowed:
                    logger.debug(f"Trade blocked by risk gate: {reason}")
                    result.blocked_reasons.append(reason)
                    result.blocked_by_risk += 1
                    continue

            # Execute paper trade
            try:
                yes_trade, no_trade = self._paper_ledger.execute_arb_trade(
                    signal, quantity=self.config.trade_quantity
                )

                trades_this_tick += 1
                result.trades_executed += 2  # Both YES and NO trades
                result.executed_signals.append(
                    {
                        "market_id": signal.market_id,
                        "yes_price": signal.yes_price,
                        "no_price": signal.no_price,
                        "profit_potential_bps": signal.profit_potential * 10000,
                        "quantity": self.config.trade_quantity,
                    }
                )

                # Record trade in risk gate if available
                if self._risk_gate:
                    self._risk_gate.record_trade()
                    self._risk_gate.record_trade()  # Two trades (YES + NO)

                logger.info(
                    f"Paper trade executed: {signal.market_id[:16]}... "
                    f"edge={signal.profit_potential * 10000:.1f}bps"
                )

            except SafetyError as e:
                logger.debug(f"Trade blocked by safety: {e}")
                result.blocked_reasons.append(f"Safety: {e}")
                result.blocked_by_safety += 1

            except Exception as e:
                logger.warning(f"Paper trade failed: {e}")
                result.error_messages.append(f"Trade error: {e}")
                result.errors += 1

        # Get final PnL snapshot
        pnl = self._paper_ledger.calculate_pnl(markets_data=current_prices)
        result.realized_pnl = pnl["total_realized_pnl"]
        result.unrealized_pnl = pnl["total_unrealized_pnl"]
        result.total_pnl = pnl["total_pnl"]
        result.position_count = pnl["position_count"]

        if result.trades_executed > 0:
            logger.info(
                f"Paper tick: {result.signals_found} signals, "
                f"{result.trades_executed} trades, "
                f"PnL=${result.total_pnl:.2f}"
            )

        return result

    def get_pnl_snapshot(
        self,
        current_prices: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Get current PnL snapshot without executing trades.

        Args:
            current_prices: Optional dict of market_id -> prices for unrealized PnL

        Returns:
            PnL summary dict
        """
        return self._paper_ledger.calculate_pnl(markets_data=current_prices)

    def get_positions(self) -> list[Any]:
        """Get all open paper positions."""
        return self._paper_ledger.get_all_positions()

    def get_recent_trades(self, limit: int = 100) -> list[Any]:
        """Get recent paper trades.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of PaperTrade objects
        """
        return self._paper_ledger.get_trades(limit=limit)

    def get_stats(self) -> dict[str, Any]:
        """Get paper trading statistics."""
        return self._paper_ledger.get_stats()
