"""Strategy scorecard for evaluating backtest results.

Computes a score (0-100) based on metrics like PnL, drawdown,
win rate, and data quality. Produces pass/fail reasons and
recommended risk limits.
"""

from dataclasses import dataclass, field
from typing import Any

from pmq.logging import get_logger

logger = get_logger("governance.scorecard")


@dataclass
class RiskLimits:
    """Recommended risk limits for a strategy."""

    max_notional_per_market: float = 500.0
    max_total_notional: float = 5000.0
    max_positions: int = 20
    max_trades_per_hour: int = 50
    stop_loss_pct: float = 0.10  # 10% drawdown triggers stop
    min_data_quality_pct: float = 80.0  # Minimum coverage required


@dataclass
class StrategyScorecard:
    """Scorecard result from strategy evaluation."""

    score: float  # 0-100
    passed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_limits: RiskLimits = field(default_factory=RiskLimits)
    metrics_used: dict[str, Any] = field(default_factory=dict)


# Scoring weights and thresholds
SCORE_WEIGHTS = {
    "pnl": 25,
    "drawdown": 20,
    "win_rate": 15,
    "sharpe": 20,
    "trades_per_day": 10,
    "data_quality": 10,
}

# Minimum thresholds for passing
MIN_THRESHOLDS = {
    "pnl": 0.0,  # Must be non-negative
    "max_drawdown": 0.25,  # Max 25% drawdown
    "win_rate": 0.40,  # At least 40% win rate
    "sharpe": 0.5,  # Sharpe-like ratio >= 0.5
    "min_trades": 5,  # At least 5 trades
    "data_quality_pct": 70.0,  # At least 70% coverage
}


def compute_scorecard(
    total_pnl: float,
    max_drawdown: float,
    win_rate: float,
    sharpe_ratio: float,
    total_trades: int,
    trades_per_day: float,
    capital_utilization: float,
    initial_balance: float,
    data_quality_pct: float | None = None,
    _missing_intervals: int = 0,  # Reserved for future use
    largest_gap_seconds: float = 0.0,
) -> StrategyScorecard:
    """Compute a scorecard from backtest metrics.

    Args:
        total_pnl: Total profit/loss from backtest
        max_drawdown: Maximum drawdown as decimal (0.1 = 10%)
        win_rate: Win rate as decimal (0.5 = 50%)
        sharpe_ratio: Sharpe-like ratio
        total_trades: Total number of trades
        trades_per_day: Average trades per day
        capital_utilization: Average capital utilization
        initial_balance: Starting balance
        data_quality_pct: Data coverage percentage (optional)
        missing_intervals: Number of missing data intervals
        largest_gap_seconds: Largest gap in data

    Returns:
        StrategyScorecard with score, pass/fail, and recommendations
    """
    reasons: list[str] = []
    warnings: list[str] = []
    scores: dict[str, float] = {}

    # Track metrics used
    metrics_used = {
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": total_trades,
        "trades_per_day": trades_per_day,
        "capital_utilization": capital_utilization,
        "data_quality_pct": data_quality_pct,
    }

    # --- PnL Score (25 points) ---
    pnl_pct = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0
    if total_pnl < MIN_THRESHOLDS["pnl"]:
        scores["pnl"] = 0
        reasons.append(f"FAIL: Negative PnL (${total_pnl:.2f})")
    elif pnl_pct >= 20:
        scores["pnl"] = SCORE_WEIGHTS["pnl"]
    elif pnl_pct >= 10:
        scores["pnl"] = SCORE_WEIGHTS["pnl"] * 0.8
    elif pnl_pct >= 5:
        scores["pnl"] = SCORE_WEIGHTS["pnl"] * 0.6
    elif pnl_pct >= 0:
        scores["pnl"] = SCORE_WEIGHTS["pnl"] * 0.4
        warnings.append(f"Low PnL return: {pnl_pct:.1f}%")
    else:
        scores["pnl"] = 0

    # --- Drawdown Score (20 points) ---
    if max_drawdown > MIN_THRESHOLDS["max_drawdown"]:
        scores["drawdown"] = 0
        reasons.append(f"FAIL: Max drawdown too high ({max_drawdown:.1%})")
    elif max_drawdown <= 0.05:
        scores["drawdown"] = SCORE_WEIGHTS["drawdown"]
    elif max_drawdown <= 0.10:
        scores["drawdown"] = SCORE_WEIGHTS["drawdown"] * 0.8
    elif max_drawdown <= 0.15:
        scores["drawdown"] = SCORE_WEIGHTS["drawdown"] * 0.6
    elif max_drawdown <= 0.20:
        scores["drawdown"] = SCORE_WEIGHTS["drawdown"] * 0.4
        warnings.append(f"High drawdown: {max_drawdown:.1%}")
    else:
        scores["drawdown"] = SCORE_WEIGHTS["drawdown"] * 0.2
        warnings.append(f"Very high drawdown: {max_drawdown:.1%}")

    # --- Win Rate Score (15 points) ---
    if win_rate < MIN_THRESHOLDS["win_rate"]:
        scores["win_rate"] = 0
        reasons.append(f"FAIL: Win rate too low ({win_rate:.1%})")
    elif win_rate >= 0.70:
        scores["win_rate"] = SCORE_WEIGHTS["win_rate"]
    elif win_rate >= 0.60:
        scores["win_rate"] = SCORE_WEIGHTS["win_rate"] * 0.8
    elif win_rate >= 0.50:
        scores["win_rate"] = SCORE_WEIGHTS["win_rate"] * 0.6
    else:
        scores["win_rate"] = SCORE_WEIGHTS["win_rate"] * 0.4

    # --- Sharpe Score (20 points) ---
    if sharpe_ratio < MIN_THRESHOLDS["sharpe"]:
        scores["sharpe"] = 0
        reasons.append(f"FAIL: Sharpe ratio too low ({sharpe_ratio:.2f})")
    elif sharpe_ratio >= 2.0:
        scores["sharpe"] = SCORE_WEIGHTS["sharpe"]
    elif sharpe_ratio >= 1.5:
        scores["sharpe"] = SCORE_WEIGHTS["sharpe"] * 0.8
    elif sharpe_ratio >= 1.0:
        scores["sharpe"] = SCORE_WEIGHTS["sharpe"] * 0.6
    else:
        scores["sharpe"] = SCORE_WEIGHTS["sharpe"] * 0.4

    # --- Trades/Day Score (10 points) ---
    if total_trades < MIN_THRESHOLDS["min_trades"]:
        scores["trades_per_day"] = 0
        reasons.append(f"FAIL: Too few trades ({total_trades})")
    elif trades_per_day >= 5:
        scores["trades_per_day"] = SCORE_WEIGHTS["trades_per_day"]
    elif trades_per_day >= 2:
        scores["trades_per_day"] = SCORE_WEIGHTS["trades_per_day"] * 0.7
    elif trades_per_day >= 1:
        scores["trades_per_day"] = SCORE_WEIGHTS["trades_per_day"] * 0.4
    else:
        scores["trades_per_day"] = SCORE_WEIGHTS["trades_per_day"] * 0.2
        warnings.append(f"Low trading frequency: {trades_per_day:.1f}/day")

    # --- Data Quality Score (10 points) ---
    if data_quality_pct is not None:
        if data_quality_pct < MIN_THRESHOLDS["data_quality_pct"]:
            scores["data_quality"] = 0
            reasons.append(f"FAIL: Data quality too low ({data_quality_pct:.1f}%)")
        elif data_quality_pct >= 95:
            scores["data_quality"] = SCORE_WEIGHTS["data_quality"]
        elif data_quality_pct >= 85:
            scores["data_quality"] = SCORE_WEIGHTS["data_quality"] * 0.8
        elif data_quality_pct >= 75:
            scores["data_quality"] = SCORE_WEIGHTS["data_quality"] * 0.6
        else:
            scores["data_quality"] = SCORE_WEIGHTS["data_quality"] * 0.4
            warnings.append(f"Low data quality: {data_quality_pct:.1f}%")

        if largest_gap_seconds > 3600:  # More than 1 hour gap
            warnings.append(f"Large data gap detected: {largest_gap_seconds/60:.0f} minutes")
    else:
        # No data quality info - partial score
        scores["data_quality"] = SCORE_WEIGHTS["data_quality"] * 0.5
        warnings.append("No data quality report available")

    # --- Calculate Total Score ---
    total_score = sum(scores.values())

    # --- Determine Pass/Fail ---
    # Must pass all critical thresholds AND score >= 60
    has_failures = any(r.startswith("FAIL:") for r in reasons)
    passed = not has_failures and total_score >= 60

    if not passed and not has_failures:
        reasons.append(f"FAIL: Score too low ({total_score:.0f}/100, need 60)")

    if passed:
        reasons.insert(0, f"PASS: Score {total_score:.0f}/100")

    # --- Calculate Recommended Limits ---
    limits = _compute_recommended_limits(
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        trades_per_day=trades_per_day,
        capital_utilization=capital_utilization,
        initial_balance=initial_balance,
        data_quality_pct=data_quality_pct,
    )

    return StrategyScorecard(
        score=total_score,
        passed=passed,
        reasons=reasons,
        warnings=warnings,
        recommended_limits=limits,
        metrics_used=metrics_used,
    )


def _compute_recommended_limits(
    total_pnl: float,
    max_drawdown: float,
    win_rate: float,
    _trades_per_day: float,  # Reserved for future limit calculations
    _capital_utilization: float,  # Reserved for future limit calculations
    initial_balance: float,
    data_quality_pct: float | None,
) -> RiskLimits:
    """Compute recommended risk limits based on backtest performance.

    Higher performance -> more generous limits.
    """
    # Base limits
    base_notional = 500.0
    base_total = 5000.0
    base_positions = 20
    base_trades_hr = 50

    # Performance multiplier (0.5 to 2.0)
    pnl_pct = (total_pnl / initial_balance) if initial_balance > 0 else 0
    perf_mult = 1.0 + min(max(pnl_pct, -0.5), 1.0)  # Clamp to 0.5-2.0

    # Drawdown penalty
    dd_mult = 1.0 - min(max_drawdown, 0.5)  # Lower limits for high drawdown

    # Win rate bonus
    wr_mult = 0.8 + (win_rate * 0.4)  # 0.8-1.2

    # Combined multiplier
    combined = perf_mult * dd_mult * wr_mult

    # Data quality affects minimum coverage requirement
    min_quality = 80.0
    if data_quality_pct is not None and data_quality_pct < 90:
        min_quality = 85.0  # Stricter if data quality was borderline

    return RiskLimits(
        max_notional_per_market=base_notional * combined,
        max_total_notional=base_total * combined,
        max_positions=int(base_positions * combined),
        max_trades_per_hour=int(base_trades_hr * combined),
        stop_loss_pct=min(max_drawdown * 1.5, 0.20),  # 1.5x observed drawdown, max 20%
        min_data_quality_pct=min_quality,
    )
