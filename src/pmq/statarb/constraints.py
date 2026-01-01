"""Market constraints for statistical arbitrage pair filtering.

This module provides:
- Constraint checking for liquidity and spread thresholds
- Filtering pairs based on market conditions
- Structured reporting of constraint violations

All functions are deterministic: same input produces same output.
"""

from dataclasses import dataclass, field
from typing import Any

from pmq.logging import get_logger
from pmq.statarb.pairs_config import PairConfig

logger = get_logger("statarb.constraints")


@dataclass
class FilterReason:
    """Reason why a pair was filtered out."""

    pair_name: str
    reason: str
    threshold: float | None = None
    actual_value: float | None = None


@dataclass
class ConstraintResult:
    """Result of applying market constraints to pairs."""

    eligible_pairs: list[PairConfig] = field(default_factory=list)
    filtered_pairs: list[PairConfig] = field(default_factory=list)
    filter_reasons: list[FilterReason] = field(default_factory=list)
    constraints_applied: bool = False

    # Summary counts
    total_pairs: int = 0
    eligible_count: int = 0
    filtered_low_liquidity: int = 0
    filtered_high_spread: int = 0
    filtered_other: int = 0

    # Phase 4.9: Microstructure data source tracking
    used_microstructure_spread: int = 0  # Pairs that used spread_bps from order book
    used_microstructure_depth: int = 0  # Pairs that used top_depth_usd from order book
    missing_microstructure: int = 0  # Pairs where microstructure was not available

    @property
    def has_filtered_pairs(self) -> bool:
        """Check if any pairs were filtered out."""
        return len(self.filtered_pairs) > 0


def compute_pair_liquidity(
    snapshots: list[dict[str, Any]],
    market_a_id: str,
    market_b_id: str,
    times: list[str],
) -> tuple[float | None, bool]:
    """Compute average liquidity for a pair over given times.

    Phase 4.9: Prefers top_depth_usd from order book microstructure.
    Falls back to yes_bid_amount + yes_ask_amount if not available.

    Args:
        snapshots: All snapshots
        market_a_id: First market ID
        market_b_id: Second market ID
        times: Times to consider

    Returns:
        Tuple of (average liquidity or None, used_microstructure bool)
    """
    # Index snapshots by (market_id, time)
    snapshot_index: dict[tuple[str, str], dict[str, Any]] = {}
    for snap in snapshots:
        key = (snap["market_id"], snap["snapshot_time"])
        snapshot_index[key] = snap

    liquidity_values: list[float] = []
    used_microstructure = False
    times_set = set(times)

    for t in times_set:
        for market_id in [market_a_id, market_b_id]:
            key = (market_id, t)
            if key in snapshot_index:
                snap = snapshot_index[key]
                # Phase 4.9: Prefer top_depth_usd from order book
                top_depth = snap.get("top_depth_usd")
                if top_depth is not None and top_depth > 0:
                    liquidity_values.append(top_depth)
                    used_microstructure = True
                else:
                    # Fallback to legacy bid/ask amounts
                    bid_amt = snap.get("yes_bid_amount", 0.0) or 0.0
                    ask_amt = snap.get("yes_ask_amount", 0.0) or 0.0
                    if bid_amt > 0 or ask_amt > 0:
                        liquidity_values.append(bid_amt + ask_amt)

    if not liquidity_values:
        return None, False

    return sum(liquidity_values) / len(liquidity_values), used_microstructure


def compute_pair_spread(
    snapshots: list[dict[str, Any]],
    market_a_id: str,
    market_b_id: str,
    times: list[str],
) -> tuple[float | None, bool]:
    """Compute average bid-ask spread for a pair over given times.

    Phase 4.9: Prefers spread_bps from order book microstructure.
    Falls back to computing from best_bid/best_ask if not available.

    Args:
        snapshots: All snapshots
        market_a_id: First market ID
        market_b_id: Second market ID
        times: Times to consider

    Returns:
        Tuple of (average spread as decimal, used_microstructure bool)
        Note: spread_bps is stored in basis points but returned as decimal
              (e.g., 200 bps = 0.02 returned)
    """
    # Index snapshots by (market_id, time)
    snapshot_index: dict[tuple[str, str], dict[str, Any]] = {}
    for snap in snapshots:
        key = (snap["market_id"], snap["snapshot_time"])
        snapshot_index[key] = snap

    spread_values: list[float] = []
    used_microstructure = False
    times_set = set(times)

    for t in times_set:
        for market_id in [market_a_id, market_b_id]:
            key = (market_id, t)
            if key in snapshot_index:
                snap = snapshot_index[key]
                # Phase 4.9: Prefer spread_bps from order book
                spread_bps = snap.get("spread_bps")
                if spread_bps is not None and spread_bps >= 0:
                    # Convert bps to decimal (e.g., 200 bps -> 0.02)
                    spread_values.append(spread_bps / 10_000)
                    used_microstructure = True
                else:
                    # Fallback: compute from best_bid/best_ask
                    bid = snap.get("best_bid") or snap.get("yes_bid", 0.0) or 0.0
                    ask = snap.get("best_ask") or snap.get("yes_ask", 0.0) or 0.0
                    if bid > 0 and ask > 0 and ask > bid:
                        mid = (bid + ask) / 2
                        spread = (ask - bid) / mid
                        spread_values.append(spread)

    if not spread_values:
        return None, False

    return sum(spread_values) / len(spread_values), used_microstructure


def apply_market_constraints(
    pairs: list[PairConfig],
    snapshots: list[dict[str, Any]],
    times: list[str],
    enforce_constraints: bool = True,
) -> ConstraintResult:
    """Apply market constraints to filter eligible pairs.

    Checks each pair against its configured constraints:
    - min_liquidity: Minimum average liquidity required (uses top_depth_usd)
    - max_spread: Maximum average spread allowed (uses spread_bps)

    Phase 4.9: Prefers real microstructure data (spread_bps, top_depth_usd)
    when available, with graceful fallback to legacy fields.

    If a pair has no constraints configured, it passes by default.
    If constraint data is not available in snapshots, the pair passes
    (graceful degradation).

    Args:
        pairs: List of pair configurations
        snapshots: Snapshot data for computing liquidity/spread
        times: Time window to consider
        enforce_constraints: If False, skip filtering (all pairs pass)

    Returns:
        ConstraintResult with eligible and filtered pairs
    """
    result = ConstraintResult(
        total_pairs=len(pairs),
        constraints_applied=enforce_constraints,
    )

    if not enforce_constraints:
        result.eligible_pairs = list(pairs)
        result.eligible_count = len(pairs)
        return result

    for pair in pairs:
        passed = True
        reason: FilterReason | None = None
        pair_used_depth = False
        pair_used_spread = False

        # Check min_liquidity constraint
        if pair.min_liquidity is not None:
            liquidity, used_micro = compute_pair_liquidity(
                snapshots, pair.market_a_id, pair.market_b_id, times
            )
            if used_micro:
                pair_used_depth = True
            if liquidity is not None and liquidity < pair.min_liquidity:
                passed = False
                reason = FilterReason(
                    pair_name=pair.name,
                    reason="low_liquidity",
                    threshold=pair.min_liquidity,
                    actual_value=liquidity,
                )
                result.filtered_low_liquidity += 1
                logger.debug(
                    f"Pair {pair.name} filtered: liquidity {liquidity:.2f} < {pair.min_liquidity}"
                )

        # Check max_spread constraint (only if not already filtered)
        if passed and pair.max_spread is not None:
            spread, used_micro = compute_pair_spread(
                snapshots, pair.market_a_id, pair.market_b_id, times
            )
            if used_micro:
                pair_used_spread = True
            if spread is not None and spread > pair.max_spread:
                passed = False
                reason = FilterReason(
                    pair_name=pair.name,
                    reason="high_spread",
                    threshold=pair.max_spread,
                    actual_value=spread,
                )
                result.filtered_high_spread += 1
                logger.debug(f"Pair {pair.name} filtered: spread {spread:.4f} > {pair.max_spread}")

        # Track microstructure usage
        if pair_used_depth:
            result.used_microstructure_depth += 1
        if pair_used_spread:
            result.used_microstructure_spread += 1
        if not pair_used_depth and not pair_used_spread:
            result.missing_microstructure += 1

        if passed:
            result.eligible_pairs.append(pair)
        else:
            result.filtered_pairs.append(pair)
            if reason:
                result.filter_reasons.append(reason)

    result.eligible_count = len(result.eligible_pairs)

    if result.has_filtered_pairs:
        logger.info(
            f"Constraint filtering: {result.eligible_count}/{result.total_pairs} pairs eligible "
            f"(filtered: {result.filtered_low_liquidity} low liquidity, "
            f"{result.filtered_high_spread} high spread)"
        )

    # Log microstructure usage
    if result.used_microstructure_spread > 0 or result.used_microstructure_depth > 0:
        logger.debug(
            f"Microstructure usage: {result.used_microstructure_spread} pairs used spread_bps, "
            f"{result.used_microstructure_depth} pairs used top_depth_usd, "
            f"{result.missing_microstructure} pairs missing microstructure"
        )

    return result


def constraint_result_to_dict(result: ConstraintResult) -> dict[str, Any]:
    """Convert ConstraintResult to dict for serialization."""
    return {
        "constraints_applied": result.constraints_applied,
        "total_pairs": result.total_pairs,
        "eligible_count": result.eligible_count,
        "filtered_low_liquidity": result.filtered_low_liquidity,
        "filtered_high_spread": result.filtered_high_spread,
        "filtered_other": result.filtered_other,
        # Phase 4.9: Microstructure usage tracking
        "used_microstructure_spread": result.used_microstructure_spread,
        "used_microstructure_depth": result.used_microstructure_depth,
        "missing_microstructure": result.missing_microstructure,
        "filter_reasons": [
            {
                "pair_name": r.pair_name,
                "reason": r.reason,
                "threshold": r.threshold,
                "actual_value": r.actual_value,
            }
            for r in result.filter_reasons
        ],
    }
