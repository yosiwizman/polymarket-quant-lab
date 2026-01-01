"""Drift detection and metrics for reconciliation.

Phase 5.5: Robust drift detection comparing WSS cache vs REST orderbooks.

Drift is detected based on configurable thresholds for:
- Mid price difference (bps)
- Spread difference (bps)
- Depth difference (percentage)

This replaces the overly strict 0.1% threshold from Phase 5.4 that caused
100% false positive drift rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmq.markets.orderbook import OrderBookData


# Default thresholds chosen to reduce false positives
DEFAULT_MID_DIFF_BPS = 25.0  # 25 bps (0.25%)
DEFAULT_SPREAD_DIFF_BPS = 25.0  # 25 bps absolute spread difference
DEFAULT_DEPTH_DIFF_PCT = 50.0  # 50% depth difference
DEFAULT_DEPTH_LEVELS = 3  # Compare top N levels


@dataclass
class DriftThresholds:
    """Configuration for drift detection thresholds.

    All thresholds are OR conditions - drift is detected if ANY threshold
    is exceeded.
    """

    mid_diff_bps: float = DEFAULT_MID_DIFF_BPS
    spread_diff_bps: float = DEFAULT_SPREAD_DIFF_BPS
    depth_diff_pct: float = DEFAULT_DEPTH_DIFF_PCT
    depth_levels: int = DEFAULT_DEPTH_LEVELS


@dataclass
class DriftMetrics:
    """Computed drift metrics between two orderbooks.

    All metrics are optional since orderbooks may be incomplete.
    """

    # Mid price difference in basis points
    mid_diff_bps: float | None = None

    # Spread difference in basis points (absolute, not relative)
    spread_diff_bps: float | None = None

    # Depth difference as percentage of REST depth
    depth_diff_pct: float | None = None

    # Whether drift was detected (based on thresholds)
    has_drift: bool = False

    # Which threshold triggered drift (for debugging)
    drift_reason: str | None = None


def _to_decimal(value: float | None) -> Decimal | None:
    """Convert float to Decimal for precise comparison."""
    if value is None:
        return None
    return Decimal(str(value))


def _compute_mid_diff_bps(
    wss_mid: float | None, rest_mid: float | None
) -> float | None:
    """Compute mid price difference in basis points.

    Formula: abs(wss_mid - rest_mid) / rest_mid * 10000

    Returns None if either mid is missing or rest_mid is zero.
    """
    if wss_mid is None or rest_mid is None or rest_mid == 0:
        return None

    # Use Decimal for precision
    wss = _to_decimal(wss_mid)
    rest = _to_decimal(rest_mid)
    if wss is None or rest is None or rest == 0:
        return None

    diff = abs(wss - rest)
    bps = (diff / rest) * Decimal("10000")
    return float(bps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _compute_spread_diff_bps(
    wss_spread_bps: float | None, rest_spread_bps: float | None
) -> float | None:
    """Compute absolute spread difference in basis points.

    This is an absolute difference, not relative - if WSS has 50 bps spread
    and REST has 30 bps spread, the diff is 20 bps.
    """
    if wss_spread_bps is None or rest_spread_bps is None:
        return None

    diff = abs(wss_spread_bps - rest_spread_bps)
    return round(diff, 2)


def _compute_depth_sum(ob: OrderBookData, levels: int) -> tuple[float, float]:
    """Compute sum of depth (size × price) for top N levels.

    Returns (bid_depth, ask_depth) in notional terms.
    """
    bid_depth = 0.0
    ask_depth = 0.0

    for bid in ob.bids[:levels]:
        bid_depth += bid.price * bid.size

    for ask in ob.asks[:levels]:
        ask_depth += ask.price * ask.size

    return bid_depth, ask_depth


def _compute_depth_diff_pct(
    wss_ob: OrderBookData, rest_ob: OrderBookData, levels: int
) -> float | None:
    """Compute depth difference as percentage of REST depth.

    Uses total notional across top N levels of both sides.
    Returns None if REST depth is zero.
    """
    wss_bid, wss_ask = _compute_depth_sum(wss_ob, levels)
    rest_bid, rest_ask = _compute_depth_sum(rest_ob, levels)

    wss_total = wss_bid + wss_ask
    rest_total = rest_bid + rest_ask

    if rest_total == 0:
        return None

    diff = abs(wss_total - rest_total)
    pct = (diff / rest_total) * 100
    return round(pct, 2)


def compute_drift_metrics(
    wss_ob: OrderBookData,
    rest_ob: OrderBookData,
    thresholds: DriftThresholds | None = None,
) -> DriftMetrics:
    """Compute drift metrics between WSS cached orderbook and REST orderbook.

    Args:
        wss_ob: Orderbook from WSS cache
        rest_ob: Orderbook from REST API (source of truth)
        thresholds: Drift detection thresholds (uses defaults if None)

    Returns:
        DriftMetrics with computed values and has_drift flag
    """
    if thresholds is None:
        thresholds = DriftThresholds()

    metrics = DriftMetrics()

    # Compute mid price difference
    metrics.mid_diff_bps = _compute_mid_diff_bps(wss_ob.mid_price, rest_ob.mid_price)

    # Compute spread difference
    metrics.spread_diff_bps = _compute_spread_diff_bps(
        wss_ob.spread_bps, rest_ob.spread_bps
    )

    # Compute depth difference
    metrics.depth_diff_pct = _compute_depth_diff_pct(
        wss_ob, rest_ob, thresholds.depth_levels
    )

    # Check thresholds (OR conditions)
    drift_reasons: list[str] = []

    if metrics.mid_diff_bps is not None and metrics.mid_diff_bps > thresholds.mid_diff_bps:
        drift_reasons.append(f"mid={metrics.mid_diff_bps:.1f}bps")

    if (
        metrics.spread_diff_bps is not None
        and metrics.spread_diff_bps > thresholds.spread_diff_bps
    ):
        drift_reasons.append(f"spread={metrics.spread_diff_bps:.1f}bps")

    if (
        metrics.depth_diff_pct is not None
        and metrics.depth_diff_pct > thresholds.depth_diff_pct
    ):
        drift_reasons.append(f"depth={metrics.depth_diff_pct:.1f}%")

    if drift_reasons:
        metrics.has_drift = True
        metrics.drift_reason = ", ".join(drift_reasons)

    return metrics
