"""Paper execution module for live paper trading loop.

Phase 6.1: Live paper execution loop that:
- Generates signals from real-time snapshots using existing scanners
- Routes through governance/risk gate for approval
- Executes in PaperLedger only (no real orders)
- Tracks per-tick and daily paper trading metrics

Phase 6.2: Paper-Exec Explain Mode + Rejection Taxonomy + Calibration Exports:
- Compute ALL candidate opportunities (even rejected ones)
- Classify rejections with explicit taxonomy
- Export top N candidates per tick to JSONL for calibration
- Add diagnostics to daemon_summary for tuning min-edge

HARD RULE: No real order placement or private key logic. PaperLedger only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pmq.logging import get_logger
from pmq.ops.edge_calc import compute_edge_from_prices
from pmq.strategies import ArbitrageScanner, PaperLedger, StatArbScanner
from pmq.strategies.paper import SafetyError

if TYPE_CHECKING:
    from pmq.governance import RiskGate
    from pmq.storage import DAO

logger = get_logger("ops.paper_exec")


# =============================================================================
# Phase 6.2: Rejection Taxonomy
# =============================================================================


class RejectionReason(Enum):
    """Rejection reason taxonomy for paper execution candidates.

    Phase 6.2: Explicit rejection reasons for calibration and debugging.
    Each reason maps to a specific threshold or condition that blocked execution.
    """

    # Passed all checks - executed or would execute
    NONE = "none"
    EXECUTED = "executed"  # Successfully executed

    # Signal quality rejections
    NO_SIGNAL = "no_signal"  # No arbitrage signal detected for market
    EDGE_BELOW_MIN = "edge_below_min"  # Profit potential < min_signal_edge_bps
    LIQUIDITY_BELOW_MIN = "liquidity_below_min"  # Market liquidity too low
    SPREAD_ABOVE_MAX = "spread_above_max"  # Bid-ask spread too wide

    # Market state rejections
    MARKET_INACTIVE = "market_inactive"  # Market not active
    MARKET_CLOSED = "market_closed"  # Market is closed

    # Risk/governance rejections
    RISK_NOT_APPROVED = "risk_not_approved"  # Strategy not approved by governance
    RISK_POSITION_LIMIT = "risk_position_limit"  # Would exceed position limit
    RISK_NOTIONAL_LIMIT = "risk_notional_limit"  # Would exceed notional limit
    RISK_TRADE_RATE_LIMIT = "risk_trade_rate_limit"  # Too many trades in window
    RISK_BLOCKED = "risk_blocked"  # Generic risk gate block

    # Safety rejections
    SAFETY_ERROR = "safety_error"  # SafetyGuard blocked the trade

    # Execution limit
    MAX_TRADES_PER_TICK = "max_trades_per_tick"  # Hit per-tick trade limit


@dataclass
class ExplainCandidate:
    """Candidate opportunity with explain data for calibration.

    Phase 6.2: Captures all relevant data for a potential trade,
    including why it was rejected (if applicable).

    Phase 7: Added raw_edge_bps which is computed BEFORE risk gating.
    This allows calibration even when risk rejects the trade.

    Phase 8: Added orderbook-based arb edge fields:
    - yes_token_id, no_token_id: CLOB token identifiers
    - arb_side: BUY_BOTH, SELL_BOTH, or NONE
    - ask_yes, ask_no, bid_yes, bid_no: Best prices from orderbooks
    """

    # Market identification
    market_id: str
    token_id: str | None = None
    market_question: str = ""

    # Phase 8: Token IDs for both sides of binary market
    yes_token_id: str | None = None
    no_token_id: str | None = None

    # Signal data
    side: str = "arb"  # "arb" for arbitrage, "long", "short" for directional
    arb_side: str = "NONE"  # Phase 8: BUY_BOTH, SELL_BOTH, or NONE
    yes_price: float = 0.0
    no_price: float = 0.0
    mid_price: float = 0.0
    spread_bps: float = 0.0  # Bid-ask spread in basis points (if available)
    edge_bps: float = 0.0  # Computed edge/profit potential in basis points
    raw_edge_bps: float = 0.0  # Edge computed BEFORE risk gating (Phase 7)
    liquidity: float = 0.0  # Market liquidity estimate
    size_estimate: float = 0.0  # Estimated executable size

    # Phase 8: Orderbook prices for arb edge computation
    ask_yes: float | None = None  # Best ask on YES token
    ask_no: float | None = None  # Best ask on NO token
    bid_yes: float | None = None  # Best bid on YES token
    bid_no: float | None = None  # Best bid on NO token

    # Rejection tracking
    rejection_reason: RejectionReason = RejectionReason.NONE
    rejection_detail: str = ""  # Additional context (e.g., "edge=42bps < min=50bps")

    # Execution result (if executed)
    executed: bool = False
    trade_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "market_id": self.market_id,
            "token_id": self.token_id,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "market_question": self.market_question[:100] if self.market_question else "",
            "side": self.side,
            "arb_side": self.arb_side,
            "yes_price": round(self.yes_price, 6),
            "no_price": round(self.no_price, 6),
            "mid_price": round(self.mid_price, 6),
            "spread_bps": round(self.spread_bps, 2),
            "edge_bps": round(self.edge_bps, 2),
            "raw_edge_bps": round(self.raw_edge_bps, 2),
            "ask_yes": round(self.ask_yes, 6) if self.ask_yes is not None else None,
            "ask_no": round(self.ask_no, 6) if self.ask_no is not None else None,
            "bid_yes": round(self.bid_yes, 6) if self.bid_yes is not None else None,
            "bid_no": round(self.bid_no, 6) if self.bid_no is not None else None,
            "liquidity": round(self.liquidity, 2),
            "size_estimate": round(self.size_estimate, 2),
            "rejection_reason": self.rejection_reason.value,
            "rejection_detail": self.rejection_detail,
            "executed": self.executed,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PaperExecConfig:
    """Configuration for paper execution loop.

    SAFETY: Paper execution is disabled by default and requires explicit
    opt-in. All trades go through PaperLedger only - no real orders.

    Phase 6.2: Explain mode captures ALL candidates (even rejected) for
    calibration and debugging. Export path defaults to exports/paper_exec_<date>.jsonl.
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
    # Phase 6.2: Explain mode settings
    explain_enabled: bool = False  # Enable explain mode (capture all candidates)
    explain_top_n: int = 10  # Number of top candidates to track per tick
    explain_export_path: Path | None = None  # JSONL export path (None = auto)


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class PaperExecResult:
    """Result of paper execution for a single tick.

    Contains counts of signals found, executed, blocked, and errors.
    Also includes PnL snapshot from PaperLedger.

    Phase 6.2: Added explain_candidates and rejection_counts for calibration.
    Phase 6.2.1: Added markets_scanned for diagnostics.
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

    # Phase 6.2: Explain mode data
    explain_candidates: list[ExplainCandidate] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)  # reason -> count

    # Phase 6.2.1: Diagnostics
    markets_scanned: int = 0  # Number of markets scanned this tick


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

        Phase 6.2: When explain_enabled, also computes ALL candidates with
        rejection reasons and tracks top N by edge for calibration.
        """
        result = PaperExecResult()
        all_candidates: list[ExplainCandidate] = []  # Phase 6.2: Track all candidates
        rejection_counts: dict[str, int] = {}  # Phase 6.2: Count by reason

        def _count_rejection(reason: RejectionReason) -> None:
            """Increment rejection counter."""
            key = reason.value
            rejection_counts[key] = rejection_counts.get(key, 0) + 1

        if not self.config.enabled:
            # Paper execution disabled - just return PnL snapshot
            pnl = self._paper_ledger.calculate_pnl(markets_data=current_prices)
            result.realized_pnl = pnl["total_realized_pnl"]
            result.unrealized_pnl = pnl["total_unrealized_pnl"]
            result.total_pnl = pnl["total_pnl"]
            result.position_count = pnl["position_count"]
            return result

        # Check governance approval if required
        # Phase 6.2.1: Don't increment blocked_by_risk here - only count when
        # there are actual executable candidates blocked by the gate
        strategy_approved = True
        approval_rejection_reason = ""
        if self.config.require_approval and self._risk_gate:
            try:
                approval = self._risk_gate.check_approval(strategy_name)
                if not approval.approved:
                    strategy_approved = False
                    approval_rejection_reason = approval.reason
                    logger.debug(f"Paper execution blocked: {approval.reason}")
                    result.blocked_reasons.append(f"Not approved: {approval.reason}")
                    # Phase 6.2.1: DON'T increment blocked_by_risk here
                    # We only count as "blocked" when there were actual executable signals
                    # Still return PnL snapshot
                    pnl = self._paper_ledger.calculate_pnl(markets_data=current_prices)
                    result.realized_pnl = pnl["total_realized_pnl"]
                    result.unrealized_pnl = pnl["total_unrealized_pnl"]
                    result.total_pnl = pnl["total_pnl"]
                    result.position_count = pnl["position_count"]
                    # Phase 6.2: Even if not approved, build candidates in explain mode
                    if not self.config.explain_enabled:
                        return result
            except Exception as e:
                logger.warning(f"Risk gate check failed: {e}")
                result.error_messages.append(f"Risk gate error: {e}")
                result.errors += 1

        # Limit markets scanned
        markets_to_scan = markets_data[: self.config.max_markets_scanned]
        result.markets_scanned = len(markets_to_scan)  # Phase 6.2.1: Track for diagnostics

        # Phase 6.2: In explain mode, get ALL signals (not limited)
        scan_top_n = None if self.config.explain_enabled else self.config.max_trades_per_tick * 2

        # Run arbitrage scanner
        try:
            arb_signals = self._arb_scanner.scan_from_db(
                markets_to_scan,
                top_n=scan_top_n,
            )
            result.signals_found = len(arb_signals)
        except Exception as e:
            logger.warning(f"Arbitrage scanner failed: {e}")
            result.error_messages.append(f"Scanner error: {e}")
            result.errors += 1
            arb_signals = []

        # Build lookup for market data
        markets_lookup = {m.get("id", ""): m for m in markets_to_scan}

        # Phase 6.2: Process ALL signals and track candidates with rejection reasons
        min_edge = self.config.min_signal_edge_bps / 10000  # Convert bps to decimal
        min_liquidity = self._arb_scanner._config.min_liquidity

        # Build candidates from signals
        # Phase 8: Compute edge from orderbook data when available
        for signal in arb_signals:
            market_data = markets_lookup.get(signal.market_id, {})
            yes_token_id = market_data.get("yes_token_id")
            no_token_id = market_data.get("no_token_id")

            # Phase 8: Try to use orderbook data for edge computation
            orderbook = market_data.get("orderbook")
            ask_yes: float | None = None
            ask_no: float | None = None
            bid_yes: float | None = None
            bid_no: float | None = None
            arb_side = "NONE"
            mid_price = 0.0
            spread_bps = 0.0

            if orderbook and orderbook.get("best_bid") and orderbook.get("best_ask"):
                # Phase 8: Use orderbook prices for edge computation
                # We have YES orderbook - derive NO prices using binary relation
                bid_yes = orderbook["best_bid"]
                ask_yes = orderbook["best_ask"]
                bid_no = max(0.001, 1.0 - ask_yes)  # Derive from YES ask
                ask_no = min(0.999, 1.0 - bid_yes)  # Derive from YES bid

                # Compute edge using edge_calc
                edge_result = compute_edge_from_prices(
                    ask_yes=ask_yes,
                    ask_no=ask_no,
                    bid_yes=bid_yes,
                    bid_no=bid_no,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                )

                raw_edge_bps = edge_result.raw_edge_bps
                edge_bps = raw_edge_bps
                arb_side = edge_result.arb_side.value
                mid_price = edge_result.mid_price or 0.0
                spread_bps = edge_result.spread_bps or 0.0
            else:
                # Fallback: Use signal's profit_potential from Gamma prices
                edge_bps = signal.profit_potential * 10000
                raw_edge_bps = edge_bps
                mid_price = (
                    (signal.yes_price + signal.no_price) / 2
                    if signal.yes_price + signal.no_price > 0
                    else 0
                )

            # Phase 8: Build candidate with all fields
            candidate = ExplainCandidate(
                market_id=signal.market_id,
                market_question=signal.market_question,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                side="arb",
                arb_side=arb_side,
                yes_price=signal.yes_price,
                no_price=signal.no_price,
                mid_price=mid_price,
                spread_bps=spread_bps,
                edge_bps=edge_bps,
                raw_edge_bps=raw_edge_bps,  # Preserve true edge before any gating
                liquidity=signal.liquidity,
                size_estimate=self.config.trade_quantity,
                ask_yes=ask_yes,
                ask_no=ask_no,
                bid_yes=bid_yes,
                bid_no=bid_no,
            )

            # Determine rejection reason (if any)
            if not strategy_approved:
                candidate.rejection_reason = RejectionReason.RISK_NOT_APPROVED
                candidate.rejection_detail = approval_rejection_reason
                _count_rejection(RejectionReason.RISK_NOT_APPROVED)
            elif not market_data.get("active", True):
                candidate.rejection_reason = RejectionReason.MARKET_INACTIVE
                candidate.rejection_detail = "Market is not active"
                _count_rejection(RejectionReason.MARKET_INACTIVE)
            elif market_data.get("closed", False):
                candidate.rejection_reason = RejectionReason.MARKET_CLOSED
                candidate.rejection_detail = "Market is closed"
                _count_rejection(RejectionReason.MARKET_CLOSED)
            elif signal.liquidity < min_liquidity:
                candidate.rejection_reason = RejectionReason.LIQUIDITY_BELOW_MIN
                candidate.rejection_detail = (
                    f"liquidity={signal.liquidity:.0f} < min={min_liquidity:.0f}"
                )
                _count_rejection(RejectionReason.LIQUIDITY_BELOW_MIN)
            elif raw_edge_bps < self.config.min_signal_edge_bps:
                # Phase 8: Use raw_edge_bps for threshold check
                candidate.rejection_reason = RejectionReason.EDGE_BELOW_MIN
                candidate.rejection_detail = (
                    f"edge={raw_edge_bps:.1f}bps < min={self.config.min_signal_edge_bps:.1f}bps"
                )
                _count_rejection(RejectionReason.EDGE_BELOW_MIN)
            # Passed basic filters - will be evaluated for execution

            all_candidates.append(candidate)

        # Filter signals that pass basic checks for execution
        filtered_signals = [
            sig
            for sig in arb_signals
            if sig.profit_potential >= min_edge and sig.liquidity >= min_liquidity
        ]
        result.signals_evaluated = len(filtered_signals)

        # Execute trades up to limit (only if approved)
        trades_this_tick = 0
        executed_market_ids: set[str] = set()

        if strategy_approved:
            for signal in filtered_signals:
                if trades_this_tick >= self.config.max_trades_per_tick:
                    # Mark remaining candidates as blocked by max trades
                    for cand in all_candidates:
                        if (
                            cand.market_id == signal.market_id
                            and cand.rejection_reason == RejectionReason.NONE
                        ):
                            cand.rejection_reason = RejectionReason.MAX_TRADES_PER_TICK
                            cand.rejection_detail = f"trades_this_tick={trades_this_tick} >= max={self.config.max_trades_per_tick}"
                            _count_rejection(RejectionReason.MAX_TRADES_PER_TICK)
                    continue

                # Check risk gate trade limit if available
                if self._risk_gate:
                    positions = self._paper_ledger.get_all_positions()
                    total_notional = sum(
                        (p.yes_quantity * p.avg_price_yes + p.no_quantity * p.avg_price_no)
                        for p in positions
                    )
                    trade_notional = self.config.trade_quantity * (
                        signal.yes_price + signal.no_price
                    )

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
                        # Update candidate with risk rejection
                        for cand in all_candidates:
                            if (
                                cand.market_id == signal.market_id
                                and cand.rejection_reason == RejectionReason.NONE
                            ):
                                if "position" in reason.lower():
                                    cand.rejection_reason = RejectionReason.RISK_POSITION_LIMIT
                                elif "notional" in reason.lower():
                                    cand.rejection_reason = RejectionReason.RISK_NOTIONAL_LIMIT
                                else:
                                    cand.rejection_reason = RejectionReason.RISK_BLOCKED
                                cand.rejection_detail = reason
                                _count_rejection(cand.rejection_reason)
                        continue

                # Execute paper trade
                try:
                    yes_trade, no_trade = self._paper_ledger.execute_arb_trade(
                        signal, quantity=self.config.trade_quantity
                    )

                    trades_this_tick += 1
                    result.trades_executed += 2  # Both YES and NO trades
                    executed_market_ids.add(signal.market_id)
                    result.executed_signals.append(
                        {
                            "market_id": signal.market_id,
                            "yes_price": signal.yes_price,
                            "no_price": signal.no_price,
                            "profit_potential_bps": signal.profit_potential * 10000,
                            "quantity": self.config.trade_quantity,
                        }
                    )

                    # Mark candidate as executed
                    for cand in all_candidates:
                        if (
                            cand.market_id == signal.market_id
                            and cand.rejection_reason == RejectionReason.NONE
                        ):
                            cand.executed = True
                            cand.trade_ids = [
                                getattr(yes_trade, "id", 0),
                                getattr(no_trade, "id", 0),
                            ]
                            _count_rejection(RejectionReason.NONE)  # Count successful executions

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
                    # Update candidate with safety rejection
                    for cand in all_candidates:
                        if (
                            cand.market_id == signal.market_id
                            and cand.rejection_reason == RejectionReason.NONE
                        ):
                            cand.rejection_reason = RejectionReason.SAFETY_ERROR
                            cand.rejection_detail = str(e)
                            _count_rejection(RejectionReason.SAFETY_ERROR)

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

        # Phase 6.2: Store explain data
        result.rejection_counts = rejection_counts

        # Phase 6.2.1: Generate near-miss candidates from market data when no arb signals
        # This ensures explain mode always has something to report for calibration
        if self.config.explain_enabled and len(all_candidates) == 0:
            all_candidates = self._generate_near_miss_candidates(
                markets_to_scan, markets_lookup, strategy_approved, approval_rejection_reason
            )
            # Update rejection counts from near-miss candidates
            for cand in all_candidates:
                _count_rejection(cand.rejection_reason)

        # Sort candidates by edge (descending) and take top N
        sorted_candidates = sorted(all_candidates, key=lambda c: c.edge_bps, reverse=True)
        result.explain_candidates = sorted_candidates[: self.config.explain_top_n]

        # Phase 6.2.1: Only count blocked_by_risk if there were actual executable signals blocked
        # This is computed after all processing to get accurate count
        if not strategy_approved and result.signals_evaluated > 0:
            # Strategy not approved but we had signals that would have executed
            result.blocked_by_risk += result.signals_evaluated

        if result.trades_executed > 0:
            logger.info(
                f"Paper tick: {result.signals_found} signals, "
                f"{result.trades_executed} trades, "
                f"PnL=${result.total_pnl:.2f}"
            )

        # Phase 6.2: Log explain summary
        if self.config.explain_enabled:
            total_rejected = sum(v for k, v in rejection_counts.items() if k != "none")
            total_executed = rejection_counts.get("none", 0)
            top_edge = sorted_candidates[0].edge_bps if sorted_candidates else 0
            logger.debug(
                f"Explain: {len(all_candidates)} candidates, "
                f"{total_executed} executed, {total_rejected} rejected, "
                f"top_edge={top_edge:.1f}bps, markets_scanned={result.markets_scanned}"
            )

        return result

    def _generate_near_miss_candidates(
        self,
        markets_to_scan: list[dict[str, Any]],
        _markets_lookup: dict[str, dict[str, Any]],  # Reserved for future use
        strategy_approved: bool,
        approval_rejection_reason: str,
    ) -> list[ExplainCandidate]:
        """Generate near-miss candidates from market data when no arb signals found.

        Phase 6.2.1: This ensures explain mode always has something to report,
        even when the ArbitrageScanner finds no signals. We compute edge from
        YES+NO prices and classify why each market didn't qualify.

        Phase 8: Uses orderbook data (if available) to compute true arb edge
        via edge_calc. For YES orderbook, derives NO prices using binary
        market relationship: bid_no ≈ 1 - ask_yes, ask_no ≈ 1 - bid_yes.

        Args:
            markets_to_scan: List of market dicts (may include 'orderbook' key)
            markets_lookup: Dict of market_id -> market dict
            strategy_approved: Whether strategy is approved
            approval_rejection_reason: Reason if not approved

        Returns:
            List of ExplainCandidate sorted by edge_bps (descending), limited to top N
        """
        candidates: list[ExplainCandidate] = []
        min_liquidity = self._arb_scanner._config.min_liquidity
        arb_threshold = self._arb_scanner._config.threshold

        # Cap at 2x explain_top_n to bound runtime
        max_to_process = min(len(markets_to_scan), self.config.explain_top_n * 2)

        for market in markets_to_scan[:max_to_process]:
            market_id = market.get("id", "")
            yes_price = market.get("last_price_yes", 0.0) or 0.0
            no_price = market.get("last_price_no", 0.0) or 0.0
            liquidity = market.get("liquidity", 0.0) or 0.0
            question = market.get("question", "")
            active = market.get("active", True)
            closed = market.get("closed", False)
            yes_token_id = market.get("yes_token_id")
            no_token_id = market.get("no_token_id")

            # Skip if no price data
            if yes_price <= 0 and no_price <= 0:
                continue

            # Phase 8: Try to use orderbook data for edge computation
            orderbook = market.get("orderbook")
            ask_yes: float | None = None
            ask_no: float | None = None
            bid_yes: float | None = None
            bid_no: float | None = None
            arb_side = "NONE"
            mid_price = 0.0
            spread_bps = 0.0

            if orderbook and orderbook.get("best_bid") and orderbook.get("best_ask"):
                # Phase 8: Use orderbook prices for edge computation
                # We have YES orderbook - derive NO prices using binary relation:
                # bid_no ≈ 1 - ask_yes (to sell NO, someone buys YES at ask)
                # ask_no ≈ 1 - bid_yes (to buy NO, someone sells YES at bid)
                bid_yes = orderbook["best_bid"]
                ask_yes = orderbook["best_ask"]
                bid_no = max(0.001, 1.0 - ask_yes)  # Derive from YES ask
                ask_no = min(0.999, 1.0 - bid_yes)  # Derive from YES bid

                # Compute edge using edge_calc
                edge_result = compute_edge_from_prices(
                    ask_yes=ask_yes,
                    ask_no=ask_no,
                    bid_yes=bid_yes,
                    bid_no=bid_no,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                )

                raw_edge_bps = edge_result.raw_edge_bps
                edge_bps = raw_edge_bps  # Same until fees applied
                arb_side = edge_result.arb_side.value
                mid_price = edge_result.mid_price or 0.0
                spread_bps = edge_result.spread_bps or 0.0
            else:
                # Fallback: Compute edge from Gamma prices
                combined_price = yes_price + no_price
                profit_potential = 1.0 - combined_price  # Can be negative
                raw_edge_bps = profit_potential * 10000
                edge_bps = raw_edge_bps
                mid_price = combined_price / 2 if combined_price > 0 else 0

            # Phase 8: Build candidate with all new fields
            candidate = ExplainCandidate(
                market_id=market_id,
                market_question=question,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                side="arb",
                arb_side=arb_side,
                yes_price=yes_price,
                no_price=no_price,
                mid_price=mid_price,
                spread_bps=spread_bps,
                edge_bps=edge_bps,
                raw_edge_bps=raw_edge_bps,  # Preserve true edge before any gating
                liquidity=liquidity,
                size_estimate=self.config.trade_quantity,
                ask_yes=ask_yes,
                ask_no=ask_no,
                bid_yes=bid_yes,
                bid_no=bid_no,
            )

            # Determine rejection reason
            if not strategy_approved:
                candidate.rejection_reason = RejectionReason.RISK_NOT_APPROVED
                candidate.rejection_detail = approval_rejection_reason
            elif not active:
                candidate.rejection_reason = RejectionReason.MARKET_INACTIVE
                candidate.rejection_detail = "Market is not active"
            elif closed:
                candidate.rejection_reason = RejectionReason.MARKET_CLOSED
                candidate.rejection_detail = "Market is closed"
            elif liquidity < min_liquidity:
                candidate.rejection_reason = RejectionReason.LIQUIDITY_BELOW_MIN
                candidate.rejection_detail = f"liquidity={liquidity:.0f} < min={min_liquidity:.0f}"
            elif raw_edge_bps < self.config.min_signal_edge_bps:
                # Phase 8: Use raw_edge_bps for threshold check
                candidate.rejection_reason = RejectionReason.EDGE_BELOW_MIN
                candidate.rejection_detail = (
                    f"edge={raw_edge_bps:.1f}bps < min={self.config.min_signal_edge_bps:.1f}bps"
                )
            elif (yes_price + no_price) >= arb_threshold:
                # No arb opportunity based on Gamma prices
                candidate.rejection_reason = RejectionReason.NO_SIGNAL
                candidate.rejection_detail = (
                    f"YES+NO={yes_price + no_price:.4f} >= threshold={arb_threshold:.4f}"
                )
            else:
                # Should have been picked up by scanner - mark as no signal
                candidate.rejection_reason = RejectionReason.NO_SIGNAL
                candidate.rejection_detail = "Not detected by ArbitrageScanner"

            candidates.append(candidate)

        # Sort by raw_edge_bps (Phase 8: use real edge for sorting)
        candidates.sort(key=lambda c: c.raw_edge_bps, reverse=True)
        return candidates[: self.config.explain_top_n]

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


# =============================================================================
# Phase 6.2: JSONL Export for Explain Mode
# =============================================================================


def write_explain_tick(
    export_path: Path,
    tick_timestamp: str,
    result: PaperExecResult,
) -> None:
    """Write explain data for a single tick to JSONL file.

    Phase 6.2: Appends one JSON line per tick with candidates and rejection counts.
    Phase 6.2.1: Always writes, even if candidates list is empty. This ensures
    the file exists and proves the loop executed.
    Safe to call multiple times per day - appends to existing file.

    Args:
        export_path: Path to JSONL file (will be created if doesn't exist)
        tick_timestamp: ISO timestamp for this tick
        result: PaperExecResult with explain_candidates and rejection_counts
    """
    tick_record = {
        "timestamp": tick_timestamp,
        "markets_scanned": result.markets_scanned,  # Phase 6.2.1
        "signals_found": result.signals_found,
        "signals_evaluated": result.signals_evaluated,
        "trades_executed": result.trades_executed,
        "blocked_by_risk": result.blocked_by_risk,
        "blocked_by_safety": result.blocked_by_safety,
        "total_pnl": round(result.total_pnl, 2),
        "rejection_counts": result.rejection_counts,
        "candidates": [c.to_dict() for c in result.explain_candidates],
    }

    # Ensure directory exists
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to JSONL file
    with open(export_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(tick_record) + "\n")


def get_default_export_path(export_dir: Path, date_str: str | None = None) -> Path:
    """Get default export path for explain JSONL.

    Args:
        export_dir: Directory for exports
        date_str: Date string (YYYY-MM-DD), defaults to today

    Returns:
        Path like exports/paper_exec_2026-01-03.jsonl
    """
    if date_str is None:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    return export_dir / f"paper_exec_{date_str}.jsonl"


@dataclass
class ExplainSummary:
    """Summary statistics from explain mode data.

    Phase 6.2: Computed from accumulated explain tick data for daemon_summary.
    Phase 7: Added raw_edge statistics for calibration when risk gates reject.
    """

    total_ticks: int = 0
    total_candidates: int = 0
    total_executed: int = 0
    total_rejected: int = 0

    # Rejection breakdown
    rejection_counts: dict[str, int] = field(default_factory=dict)

    # Edge statistics (from top candidates)
    avg_top_edge_bps: float = 0.0
    median_top_edge_bps: float = 0.0
    max_top_edge_bps: float = 0.0

    # Phase 7: Raw edge statistics (before risk gating)
    avg_top_raw_edge_bps: float = 0.0
    max_raw_edge_bps: float = 0.0
    ticks_with_raw_edge_above_threshold: int = 0  # Ticks with raw_edge >= threshold

    # Ticks with opportunities
    ticks_with_candidates: int = 0  # Ticks with at least 1 candidate
    ticks_with_candidates_above_min_edge: int = 0  # Ticks with top candidate >= min_edge

    @classmethod
    def from_results(
        cls,
        results: list[PaperExecResult],
        min_edge_bps: float = 50.0,
    ) -> ExplainSummary:
        """Build summary from list of tick results.

        Args:
            results: List of PaperExecResult from each tick
            min_edge_bps: Minimum edge threshold for counting "above min" ticks

        Returns:
            ExplainSummary with aggregated statistics
        """
        summary = cls()
        summary.total_ticks = len(results)

        top_edges: list[float] = []
        top_raw_edges: list[float] = []  # Phase 7
        combined_rejection_counts: dict[str, int] = {}

        for result in results:
            candidates = result.explain_candidates
            rejection_counts = result.rejection_counts

            if candidates:
                summary.ticks_with_candidates += 1
                top_edge = candidates[0].edge_bps if candidates else 0
                top_edges.append(top_edge)

                # Phase 7: Track raw edge from top candidate
                top_raw_edge = candidates[0].raw_edge_bps if candidates else 0
                top_raw_edges.append(top_raw_edge)

                if top_edge >= min_edge_bps:
                    summary.ticks_with_candidates_above_min_edge += 1

                # Phase 7: Count ticks with raw_edge above threshold
                if top_raw_edge >= min_edge_bps:
                    summary.ticks_with_raw_edge_above_threshold += 1

            summary.total_candidates += len(candidates)

            # Aggregate rejection counts
            for reason, count in rejection_counts.items():
                combined_rejection_counts[reason] = combined_rejection_counts.get(reason, 0) + count

        # Compute totals
        summary.total_executed = combined_rejection_counts.get("executed", 0)
        summary.total_rejected = sum(
            v for k, v in combined_rejection_counts.items() if k != "executed"
        )
        summary.rejection_counts = combined_rejection_counts

        # Compute edge statistics
        if top_edges:
            summary.avg_top_edge_bps = sum(top_edges) / len(top_edges)
            sorted_edges = sorted(top_edges)
            mid_idx = len(sorted_edges) // 2
            if len(sorted_edges) % 2 == 0:
                summary.median_top_edge_bps = (
                    sorted_edges[mid_idx - 1] + sorted_edges[mid_idx]
                ) / 2
            else:
                summary.median_top_edge_bps = sorted_edges[mid_idx]
            summary.max_top_edge_bps = max(top_edges)

        # Phase 7: Compute raw edge statistics
        if top_raw_edges:
            summary.avg_top_raw_edge_bps = sum(top_raw_edges) / len(top_raw_edges)
            summary.max_raw_edge_bps = max(top_raw_edges)

        return summary
