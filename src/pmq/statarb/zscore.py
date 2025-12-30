"""Z-score based mean-reversion engine for statistical arbitrage.

This module provides deterministic z-score signal generation:
- OLS beta computation for hedge ratio
- Spread calculation with hedge ratio
- Rolling z-score computation
- Entry/exit signal generation

All functions are deterministic: same input produces same output.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pmq.logging import get_logger

logger = get_logger("statarb.zscore")


class SignalAction(Enum):
    """Signal action types."""

    ENTER_LONG = "enter_long"  # Long A, short B (spread too low, expect reversion up)
    ENTER_SHORT = "enter_short"  # Short A, long B (spread too high, expect reversion down)
    EXIT = "exit"  # Close position (z-score reverted)
    HOLD = "hold"  # No action


@dataclass(frozen=True)
class ZScoreSignal:
    """A z-score based trading signal."""

    time: str
    pair_name: str
    market_a_id: str
    market_b_id: str
    action: SignalAction
    z_score: float
    spread: float
    price_a: float
    price_b: float
    beta: float
    reason: str = ""


@dataclass
class FittedParams:
    """Fitted parameters from TRAIN data for a pair."""

    pair_name: str
    market_a_id: str
    market_b_id: str
    beta: float  # OLS hedge ratio
    spread_mean: float  # Mean of spread on TRAIN
    spread_std: float  # Std of spread on TRAIN
    train_count: int  # Number of TRAIN observations
    is_valid: bool = True
    error_msg: str = ""


def ols_beta(prices_a: list[float], prices_b: list[float]) -> float:
    """Compute OLS beta (hedge ratio) via simple linear regression.

    Regresses prices_a on prices_b: prices_a = beta * prices_b + alpha + error

    Args:
        prices_a: Dependent variable prices (market A)
        prices_b: Independent variable prices (market B)

    Returns:
        Beta coefficient. Returns 1.0 if insufficient data or zero variance.
    """
    n = len(prices_a)
    if n < 3 or n != len(prices_b):
        return 1.0

    # Compute means
    mean_a = sum(prices_a) / n
    mean_b = sum(prices_b) / n

    # Compute covariance and variance
    cov = 0.0
    var_b = 0.0
    for a, b in zip(prices_a, prices_b, strict=True):
        da = a - mean_a
        db = b - mean_b
        cov += da * db
        var_b += db * db

    if var_b < 1e-10:
        # Zero variance in B means constant prices, default to beta=1
        return 1.0

    beta = cov / var_b
    return beta


def spread_series(
    prices_a: list[float],
    prices_b: list[float],
    beta: float,
) -> list[float]:
    """Compute spread series: spread = priceA - beta * priceB.

    Args:
        prices_a: Prices for market A
        prices_b: Prices for market B (same length)
        beta: Hedge ratio

    Returns:
        List of spread values
    """
    if len(prices_a) != len(prices_b):
        raise ValueError("prices_a and prices_b must have same length")
    return [a - beta * b for a, b in zip(prices_a, prices_b, strict=True)]


def rolling_mean_std(
    values: list[float],
    lookback: int,
) -> tuple[list[float], list[float]]:
    """Compute rolling mean and standard deviation.

    Args:
        values: Input series
        lookback: Rolling window size

    Returns:
        Tuple of (rolling_means, rolling_stds).
        For indices < lookback-1, uses all available data.
    """
    n = len(values)
    means: list[float] = []
    stds: list[float] = []

    for i in range(n):
        # Use at least 2 points, up to lookback
        start = max(0, i - lookback + 1)
        window = values[start : i + 1]

        if len(window) < 2:
            means.append(window[0] if window else 0.0)
            stds.append(0.0)
            continue

        window_mean = sum(window) / len(window)
        variance = sum((x - window_mean) ** 2 for x in window) / len(window)
        window_std = variance**0.5

        means.append(window_mean)
        stds.append(window_std)

    return means, stds


def zscore_series(
    spreads: list[float],
    lookback: int,
) -> list[float]:
    """Compute rolling z-score series.

    z = (spread - rolling_mean) / rolling_std

    Args:
        spreads: Spread values
        lookback: Rolling window size for mean/std calculation

    Returns:
        List of z-scores. Returns 0.0 when std is near zero.
    """
    means, stds = rolling_mean_std(spreads, lookback)

    zscores: list[float] = []
    for spread, mean, std in zip(spreads, means, stds, strict=True):
        if std < 1e-10:
            zscores.append(0.0)
        else:
            zscores.append((spread - mean) / std)

    return zscores


def fit_pair_params(
    prices_a: list[float],
    prices_b: list[float],
    pair_name: str,
    market_a_id: str,
    market_b_id: str,
) -> FittedParams:
    """Fit parameters for a pair from TRAIN data.

    Computes:
    - OLS beta (hedge ratio)
    - Spread mean and std on TRAIN

    Args:
        prices_a: TRAIN prices for market A
        prices_b: TRAIN prices for market B
        pair_name: Human-readable pair name
        market_a_id: Market A identifier
        market_b_id: Market B identifier

    Returns:
        FittedParams with fitted values
    """
    if len(prices_a) < 10 or len(prices_b) < 10:
        return FittedParams(
            pair_name=pair_name,
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            beta=1.0,
            spread_mean=0.0,
            spread_std=1.0,
            train_count=len(prices_a),
            is_valid=False,
            error_msg=f"Insufficient TRAIN data: {len(prices_a)} points",
        )

    if len(prices_a) != len(prices_b):
        return FittedParams(
            pair_name=pair_name,
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            beta=1.0,
            spread_mean=0.0,
            spread_std=1.0,
            train_count=len(prices_a),
            is_valid=False,
            error_msg="Mismatched price series lengths",
        )

    # Compute beta
    beta = ols_beta(prices_a, prices_b)

    # Compute spreads
    spreads = spread_series(prices_a, prices_b, beta)

    # Compute spread mean and std
    n = len(spreads)
    spread_mean = sum(spreads) / n
    variance = sum((s - spread_mean) ** 2 for s in spreads) / n
    spread_std = variance**0.5

    if spread_std < 1e-10:
        return FittedParams(
            pair_name=pair_name,
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            beta=beta,
            spread_mean=spread_mean,
            spread_std=1.0,
            train_count=n,
            is_valid=False,
            error_msg="Near-zero spread standard deviation",
        )

    return FittedParams(
        pair_name=pair_name,
        market_a_id=market_a_id,
        market_b_id=market_b_id,
        beta=beta,
        spread_mean=spread_mean,
        spread_std=spread_std,
        train_count=n,
        is_valid=True,
    )


@dataclass
class PositionState:
    """Track position state for a pair."""

    is_open: bool = False
    direction: str = ""  # "long" or "short" (long = long A/short B, short = short A/long B)
    entry_time: str = ""
    entry_z: float = 0.0
    entry_spread: float = 0.0
    bars_held: int = 0


def generate_signals(
    times: list[str],
    prices_a: list[float],
    prices_b: list[float],
    fitted_params: FittedParams,
    lookback: int,  # noqa: ARG001 - kept for API compatibility
    entry_z: float,
    exit_z: float,
    max_hold_bars: int,
    cooldown_bars: int = 0,
) -> list[ZScoreSignal]:
    """Generate z-score based entry/exit signals.

    Signal logic:
    - ENTER_LONG when z <= -entry_z (spread is low, expect reversion up)
    - ENTER_SHORT when z >= entry_z (spread is high, expect reversion down)
    - EXIT when |z| <= exit_z OR bars_held >= max_hold_bars

    Args:
        times: Timestamp for each observation
        prices_a: Prices for market A
        prices_b: Prices for market B
        fitted_params: Pre-fitted params from TRAIN
        lookback: Rolling window for z-score (uses fitted mean/std as anchor)
        entry_z: Z-score threshold to enter (absolute value)
        exit_z: Z-score threshold to exit (absolute value)
        max_hold_bars: Maximum bars to hold before forced exit
        cooldown_bars: Bars to wait after exit before new entry

    Returns:
        List of ZScoreSignals
    """
    if not fitted_params.is_valid:
        return []

    n = len(times)
    if n != len(prices_a) or n != len(prices_b):
        logger.warning("Mismatched series lengths in generate_signals")
        return []

    # Compute spreads and z-scores using fitted params
    spreads = spread_series(prices_a, prices_b, fitted_params.beta)

    # Z-score using TRAIN mean/std (more stable than rolling)
    zscores = [
        (s - fitted_params.spread_mean) / fitted_params.spread_std
        for s in spreads
    ]

    signals: list[ZScoreSignal] = []
    position = PositionState()
    cooldown_remaining = 0

    for i in range(n):
        t = times[i]
        z = zscores[i]
        spread = spreads[i]
        price_a = prices_a[i]
        price_b = prices_b[i]

        # Update cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Check for exit if in position
        if position.is_open:
            position.bars_held += 1

            should_exit = False
            exit_reason = ""

            # Exit on z-score reversion
            if abs(z) <= exit_z:
                should_exit = True
                exit_reason = f"Z-score reverted to {z:.2f}"
            # Exit on max hold
            elif position.bars_held >= max_hold_bars:
                should_exit = True
                exit_reason = f"Max hold reached ({max_hold_bars} bars)"

            if should_exit:
                signals.append(ZScoreSignal(
                    time=t,
                    pair_name=fitted_params.pair_name,
                    market_a_id=fitted_params.market_a_id,
                    market_b_id=fitted_params.market_b_id,
                    action=SignalAction.EXIT,
                    z_score=z,
                    spread=spread,
                    price_a=price_a,
                    price_b=price_b,
                    beta=fitted_params.beta,
                    reason=exit_reason,
                ))
                position = PositionState()
                cooldown_remaining = cooldown_bars
                continue

        # Check for entry if not in position and no cooldown
        if not position.is_open and cooldown_remaining == 0:
            if z <= -entry_z:
                # Spread is low, expect reversion up -> Long A, Short B
                signals.append(ZScoreSignal(
                    time=t,
                    pair_name=fitted_params.pair_name,
                    market_a_id=fitted_params.market_a_id,
                    market_b_id=fitted_params.market_b_id,
                    action=SignalAction.ENTER_LONG,
                    z_score=z,
                    spread=spread,
                    price_a=price_a,
                    price_b=price_b,
                    beta=fitted_params.beta,
                    reason=f"Z-score {z:.2f} <= -{entry_z}",
                ))
                position = PositionState(
                    is_open=True,
                    direction="long",
                    entry_time=t,
                    entry_z=z,
                    entry_spread=spread,
                    bars_held=0,
                )
            elif z >= entry_z:
                # Spread is high, expect reversion down -> Short A, Long B
                signals.append(ZScoreSignal(
                    time=t,
                    pair_name=fitted_params.pair_name,
                    market_a_id=fitted_params.market_a_id,
                    market_b_id=fitted_params.market_b_id,
                    action=SignalAction.ENTER_SHORT,
                    z_score=z,
                    spread=spread,
                    price_a=price_a,
                    price_b=price_b,
                    beta=fitted_params.beta,
                    reason=f"Z-score {z:.2f} >= {entry_z}",
                ))
                position = PositionState(
                    is_open=True,
                    direction="short",
                    entry_time=t,
                    entry_z=z,
                    entry_spread=spread,
                    bars_held=0,
                )

    # Force exit at end if still in position
    if position.is_open and n > 0:
        signals.append(ZScoreSignal(
            time=times[-1],
            pair_name=fitted_params.pair_name,
            market_a_id=fitted_params.market_a_id,
            market_b_id=fitted_params.market_b_id,
            action=SignalAction.EXIT,
            z_score=zscores[-1],
            spread=spreads[-1],
            price_a=prices_a[-1],
            price_b=prices_b[-1],
            beta=fitted_params.beta,
            reason="End of period forced exit",
        ))

    return signals


def signals_to_dict(signals: list[ZScoreSignal]) -> list[dict[str, Any]]:
    """Convert signals to list of dicts for serialization."""
    return [
        {
            "time": s.time,
            "pair_name": s.pair_name,
            "market_a_id": s.market_a_id,
            "market_b_id": s.market_b_id,
            "action": s.action.value,
            "z_score": s.z_score,
            "spread": s.spread,
            "price_a": s.price_a,
            "price_b": s.price_b,
            "beta": s.beta,
            "reason": s.reason,
        }
        for s in signals
    ]
