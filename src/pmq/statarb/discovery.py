"""Pair discovery from historical snapshot data.

This module provides deterministic pair discovery using correlation of price
changes across overlapping snapshot times. No live API calls - purely
historical data analysis.
"""

from dataclasses import dataclass
from typing import Any

from pmq.logging import get_logger
from pmq.statarb.pairs_config import PairConfig

logger = get_logger("statarb.discovery")


@dataclass(frozen=True)
class CandidatePair:
    """A candidate pair with computed statistics."""

    market_a_id: str
    market_b_id: str
    market_a_question: str
    market_b_question: str
    correlation: float
    overlap_count: int
    avg_spread: float
    max_abs_spread: float

    def to_pair_config(self, name: str = "") -> PairConfig:
        """Convert to PairConfig for YAML output."""
        if not name:
            q_a = self.market_a_question[:25] if self.market_a_question else "Market A"
            q_b = self.market_b_question[:25] if self.market_b_question else "Market B"
            name = f"{q_a} vs {q_b}"
        return PairConfig(
            market_a_id=self.market_a_id,
            market_b_id=self.market_b_id,
            name=name[:80],
            correlation=round(self.correlation, 4),
            enabled=True,
        )


def compute_correlation(prices_a: list[float], prices_b: list[float]) -> float:
    """Compute Pearson correlation between two price series.

    Args:
        prices_a: List of prices for market A
        prices_b: List of prices for market B (same length)

    Returns:
        Correlation coefficient in [-1, 1], or 0.0 if insufficient data
    """
    n = len(prices_a)
    if n < 3 or n != len(prices_b):
        return 0.0

    # Compute means
    mean_a = sum(prices_a) / n
    mean_b = sum(prices_b) / n

    # Compute covariance and standard deviations
    cov = 0.0
    var_a = 0.0
    var_b = 0.0
    for a, b in zip(prices_a, prices_b, strict=True):
        da = a - mean_a
        db = b - mean_b
        cov += da * db
        var_a += da * da
        var_b += db * db

    if var_a < 1e-10 or var_b < 1e-10:
        # Near-zero variance means constant prices, no meaningful correlation
        return 0.0

    return cov / ((var_a**0.5) * (var_b**0.5))


def discover_pairs(
    snapshots: list[dict[str, Any]],
    markets: dict[str, dict[str, Any]],
    min_overlap: int = 10,
    top: int = 50,
    min_correlation: float = 0.3,
) -> list[CandidatePair]:
    """Discover candidate pairs from snapshot data using correlation.

    This function is fully deterministic: same input produces same output.

    Args:
        snapshots: List of snapshot dicts from DAO.get_snapshots()
        markets: Dict mapping market_id -> market metadata
        min_overlap: Minimum overlapping snapshot times required
        top: Maximum number of pairs to return
        min_correlation: Minimum absolute correlation to consider

    Returns:
        List of CandidatePair sorted by absolute correlation (descending),
        then by overlap count (descending), then by market_a_id, market_b_id.
    """
    # Group snapshots by market_id -> {snapshot_time -> price}
    market_prices: dict[str, dict[str, float]] = {}
    for snap in snapshots:
        mid = snap["market_id"]
        t = snap["snapshot_time"]
        price = snap.get("yes_price", 0.0)
        if price > 0:
            if mid not in market_prices:
                market_prices[mid] = {}
            market_prices[mid][t] = price

    # Get all market IDs with sufficient data
    market_ids = sorted(mid for mid, prices in market_prices.items() if len(prices) >= min_overlap)

    logger.debug(f"Analyzing {len(market_ids)} markets with >= {min_overlap} snapshots")

    candidates: list[CandidatePair] = []

    # Compare all pairs (O(n^2) but constrained by market count)
    for i, market_a_id in enumerate(market_ids):
        prices_a_dict = market_prices[market_a_id]
        times_a = set(prices_a_dict.keys())

        for market_b_id in market_ids[i + 1 :]:
            prices_b_dict = market_prices[market_b_id]
            times_b = set(prices_b_dict.keys())

            # Find overlapping times
            common_times = sorted(times_a & times_b)
            if len(common_times) < min_overlap:
                continue

            # Build aligned price series
            prices_a = [prices_a_dict[t] for t in common_times]
            prices_b = [prices_b_dict[t] for t in common_times]

            # Compute correlation
            corr = compute_correlation(prices_a, prices_b)
            abs_corr = abs(corr)

            if abs_corr < min_correlation:
                continue

            # Compute spread statistics
            spreads = [a - b for a, b in zip(prices_a, prices_b, strict=True)]
            avg_spread = sum(spreads) / len(spreads) if spreads else 0.0
            max_abs_spread = max(abs(s) for s in spreads) if spreads else 0.0

            # Get market metadata
            market_a = markets.get(market_a_id, {})
            market_b = markets.get(market_b_id, {})

            candidate = CandidatePair(
                market_a_id=market_a_id,
                market_b_id=market_b_id,
                market_a_question=market_a.get("question", ""),
                market_b_question=market_b.get("question", ""),
                correlation=corr,
                overlap_count=len(common_times),
                avg_spread=avg_spread,
                max_abs_spread=max_abs_spread,
            )
            candidates.append(candidate)

    # Sort deterministically: by abs(correlation) desc, overlap desc, then IDs
    candidates.sort(
        key=lambda c: (-abs(c.correlation), -c.overlap_count, c.market_a_id, c.market_b_id)
    )

    logger.info(f"Discovered {len(candidates)} candidate pairs (returning top {top})")
    return candidates[:top]


def validate_pair_overlap(
    pair: PairConfig,
    snapshots: list[dict[str, Any]],
    min_overlap: int = 10,
) -> dict[str, Any]:
    """Validate that a pair has sufficient overlap in snapshot data.

    Args:
        pair: Pair configuration to validate
        snapshots: List of snapshot dicts
        min_overlap: Minimum required overlap

    Returns:
        Dict with validation results:
            - valid: bool
            - a_count: int (snapshots for market A)
            - b_count: int (snapshots for market B)
            - overlap_count: int (overlapping snapshot times)
            - reason: str (if invalid)
    """
    # Group by market_id -> set of snapshot_times
    a_times: set[str] = set()
    b_times: set[str] = set()

    for snap in snapshots:
        mid = snap["market_id"]
        t = snap["snapshot_time"]
        if mid == pair.market_a_id:
            a_times.add(t)
        elif mid == pair.market_b_id:
            b_times.add(t)

    overlap = a_times & b_times
    overlap_count = len(overlap)

    result: dict[str, Any] = {
        "valid": True,
        "a_count": len(a_times),
        "b_count": len(b_times),
        "overlap_count": overlap_count,
        "reason": "",
    }

    if len(a_times) == 0:
        result["valid"] = False
        result["reason"] = f"Market A ({pair.market_a_id}) has no snapshots"
    elif len(b_times) == 0:
        result["valid"] = False
        result["reason"] = f"Market B ({pair.market_b_id}) has no snapshots"
    elif overlap_count < min_overlap:
        result["valid"] = False
        result["reason"] = f"Insufficient overlap: {overlap_count} < {min_overlap} required"

    return result
