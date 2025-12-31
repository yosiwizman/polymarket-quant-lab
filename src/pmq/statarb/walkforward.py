"""Walk-forward evaluation for statistical arbitrage.

This module provides walk-forward (out-of-sample) evaluation:
- Split snapshot times into TRAIN and TEST segments
- Fit model parameters on TRAIN data only
- Evaluate strategy on TEST data only
- Prevents overfitting by ensuring no data leakage

All functions are deterministic: same input produces same output.
"""

from dataclasses import dataclass
from typing import Any

from pmq.logging import get_logger
from pmq.statarb.pairs_config import PairConfig
from pmq.statarb.zscore import (
    FittedParams,
    SignalAction,
    ZScoreSignal,
    fit_pair_params,
    generate_signals,
)

logger = get_logger("statarb.walkforward")


@dataclass
class WalkForwardSplit:
    """Result of splitting times into TRAIN and TEST."""

    train_times: list[str]
    test_times: list[str]
    train_count: int
    test_count: int
    total_count: int
    first_train: str
    last_train: str
    first_test: str
    last_test: str


def split_times(
    times: list[str],
    train_count: int,
    test_count: int,
) -> WalkForwardSplit:
    """Split times into TRAIN and TEST segments.

    Times are split chronologically:
    - First `train_count` times go to TRAIN
    - Next `test_count` times go to TEST

    If there aren't enough times for both, we use all available,
    prioritizing TEST (so we at least have out-of-sample data).

    Args:
        times: Sorted list of timestamps (oldest first)
        train_count: Number of times for TRAIN segment
        test_count: Number of times for TEST segment

    Returns:
        WalkForwardSplit with train and test times
    """
    total = len(times)

    if total == 0:
        return WalkForwardSplit(
            train_times=[],
            test_times=[],
            train_count=0,
            test_count=0,
            total_count=0,
            first_train="",
            last_train="",
            first_test="",
            last_test="",
        )

    # If we have fewer times than requested, scale proportionally
    requested_total = train_count + test_count
    if total < requested_total:
        # Scale down proportionally
        train_ratio = train_count / requested_total
        actual_train = int(total * train_ratio)
        actual_test = total - actual_train

        # Ensure at least some train data if possible
        if actual_train == 0 and total >= 2:
            actual_train = 1
            actual_test = total - 1
    else:
        actual_train = train_count
        actual_test = test_count

    # Split times
    train_times = times[:actual_train]
    test_times = times[actual_train : actual_train + actual_test]

    return WalkForwardSplit(
        train_times=train_times,
        test_times=test_times,
        train_count=len(train_times),
        test_count=len(test_times),
        total_count=total,
        first_train=train_times[0] if train_times else "",
        last_train=train_times[-1] if train_times else "",
        first_test=test_times[0] if test_times else "",
        last_test=test_times[-1] if test_times else "",
    )


@dataclass
class PairTrainData:
    """Training data for a single pair."""

    pair: PairConfig
    times: list[str]
    prices_a: list[float]
    prices_b: list[float]


@dataclass
class PairTestData:
    """Test data for a single pair."""

    pair: PairConfig
    times: list[str]
    prices_a: list[float]
    prices_b: list[float]
    fitted_params: FittedParams


@dataclass
class WalkForwardMetrics:
    """Metrics from walk-forward evaluation."""

    total_pnl: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    avg_trades_per_pair: float
    total_fees: float
    net_pnl: float
    entry_count: int
    exit_count: int


@dataclass
class WalkForwardResult:
    """Complete result from walk-forward evaluation."""

    split: WalkForwardSplit
    fitted_params: dict[str, FittedParams]  # pair_name -> params
    signals: list[ZScoreSignal]
    train_metrics: WalkForwardMetrics | None
    test_metrics: WalkForwardMetrics
    pair_summaries: list[dict[str, Any]]
    config_used: dict[str, Any]


def extract_pair_prices(
    snapshots: list[dict[str, Any]],
    pair: PairConfig,
    times: list[str],
) -> tuple[list[str], list[float], list[float]]:
    """Extract aligned price series for a pair from snapshots.

    Args:
        snapshots: List of snapshot dicts
        pair: Pair configuration
        times: Times to extract (filters snapshots)

    Returns:
        Tuple of (common_times, prices_a, prices_b) where all lists have same length.
        Only times where BOTH markets have data are included.
    """
    # Index snapshots by (market_id, time) -> price
    price_index: dict[tuple[str, str], float] = {}
    for snap in snapshots:
        key = (snap["market_id"], snap["snapshot_time"])
        price_index[key] = snap.get("yes_price", 0.0)

    # Find times where both markets have data
    times_set = set(times)
    common_times: list[str] = []
    prices_a: list[float] = []
    prices_b: list[float] = []

    for t in sorted(times_set):
        key_a = (pair.market_a_id, t)
        key_b = (pair.market_b_id, t)

        if key_a in price_index and key_b in price_index:
            pa = price_index[key_a]
            pb = price_index[key_b]
            if pa > 0 and pb > 0:
                common_times.append(t)
                prices_a.append(pa)
                prices_b.append(pb)

    return common_times, prices_a, prices_b


def fit_all_pairs(
    snapshots: list[dict[str, Any]],
    pairs: list[PairConfig],
    train_times: list[str],
) -> dict[str, FittedParams]:
    """Fit parameters for all pairs using TRAIN data only.

    Args:
        snapshots: All snapshots
        pairs: List of pairs to fit
        train_times: TRAIN time window

    Returns:
        Dict mapping pair_name -> FittedParams
    """
    result: dict[str, FittedParams] = {}

    for pair in pairs:
        common_times, prices_a, prices_b = extract_pair_prices(snapshots, pair, train_times)

        params = fit_pair_params(
            prices_a=prices_a,
            prices_b=prices_b,
            pair_name=pair.name,
            market_a_id=pair.market_a_id,
            market_b_id=pair.market_b_id,
        )

        result[pair.name] = params

        if params.is_valid:
            logger.debug(
                f"Fitted {pair.name}: beta={params.beta:.4f}, "
                f"mean={params.spread_mean:.4f}, std={params.spread_std:.4f}"
            )
        else:
            logger.debug(f"Failed to fit {pair.name}: {params.error_msg}")

    return result


def evaluate_test_period(
    snapshots: list[dict[str, Any]],
    pairs: list[PairConfig],
    fitted_params: dict[str, FittedParams],
    test_times: list[str],
    lookback: int,
    entry_z: float,
    exit_z: float,
    max_hold_bars: int,
    cooldown_bars: int,
    fee_bps: float,
    slippage_bps: float,
    quantity_per_trade: float,
) -> tuple[list[ZScoreSignal], WalkForwardMetrics, list[dict[str, Any]]]:
    """Evaluate strategy on TEST period using fitted params.

    Args:
        snapshots: All snapshots
        pairs: List of pairs to evaluate
        fitted_params: Pre-fitted parameters from TRAIN
        test_times: TEST time window
        lookback: Z-score lookback (for diagnostics, uses fitted mean/std)
        entry_z: Entry z-score threshold
        exit_z: Exit z-score threshold
        max_hold_bars: Max bars before forced exit
        cooldown_bars: Cooldown after exit
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        quantity_per_trade: Quantity per trade

    Returns:
        Tuple of (all_signals, metrics, pair_summaries)
    """
    all_signals: list[ZScoreSignal] = []
    pair_summaries: list[dict[str, Any]] = []

    # Generate signals for each pair
    for pair in pairs:
        params = fitted_params.get(pair.name)
        if not params or not params.is_valid:
            pair_summaries.append(
                {
                    "pair_name": pair.name,
                    "status": "skipped",
                    "reason": params.error_msg if params else "No fitted params",
                    "signals": 0,
                }
            )
            continue

        common_times, prices_a, prices_b = extract_pair_prices(snapshots, pair, test_times)

        if len(common_times) < 5:
            pair_summaries.append(
                {
                    "pair_name": pair.name,
                    "status": "skipped",
                    "reason": f"Insufficient TEST data: {len(common_times)} points",
                    "signals": 0,
                }
            )
            continue

        signals = generate_signals(
            times=common_times,
            prices_a=prices_a,
            prices_b=prices_b,
            fitted_params=params,
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            max_hold_bars=max_hold_bars,
            cooldown_bars=cooldown_bars,
        )

        all_signals.extend(signals)

        pair_summaries.append(
            {
                "pair_name": pair.name,
                "status": "evaluated",
                "beta": params.beta,
                "spread_mean": params.spread_mean,
                "spread_std": params.spread_std,
                "test_points": len(common_times),
                "signals": len(signals),
                "entries": sum(
                    1
                    for s in signals
                    if s.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT)
                ),
                "exits": sum(1 for s in signals if s.action == SignalAction.EXIT),
            }
        )

    # Compute metrics from signals
    metrics = compute_metrics_from_signals(
        all_signals,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        quantity_per_trade=quantity_per_trade,
    )

    return all_signals, metrics, pair_summaries


def compute_metrics_from_signals(
    signals: list[ZScoreSignal],
    fee_bps: float,
    slippage_bps: float,
    quantity_per_trade: float,
) -> WalkForwardMetrics:
    """Compute performance metrics from signals.

    Simple PnL model:
    - On entry: pay spread + slippage + fee
    - On exit: receive spread difference (mean reversion profit/loss) - slippage - fee

    Args:
        signals: List of signals
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        quantity_per_trade: Quantity per trade

    Returns:
        WalkForwardMetrics
    """
    if not signals:
        return WalkForwardMetrics(
            total_pnl=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            total_trades=0,
            avg_trades_per_pair=0.0,
            total_fees=0.0,
            net_pnl=0.0,
            entry_count=0,
            exit_count=0,
        )

    # Track PnL per round trip
    pnls: list[float] = []
    total_fees = 0.0

    # Track open positions: pair_name -> entry_signal
    open_positions: dict[str, ZScoreSignal] = {}

    entry_count = 0
    exit_count = 0

    # Process signals chronologically
    sorted_signals = sorted(signals, key=lambda s: (s.time, s.pair_name))

    for sig in sorted_signals:
        fee_rate = fee_bps / 10000
        slip_rate = slippage_bps / 10000
        notional = quantity_per_trade * (sig.price_a + sig.price_b) / 2

        if sig.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
            entry_count += 1

            # Pay entry costs
            entry_cost = notional * (fee_rate + slip_rate)
            total_fees += entry_cost

            # Store entry for later matching
            open_positions[sig.pair_name] = sig

        elif sig.action == SignalAction.EXIT:
            exit_count += 1

            # Pay exit costs
            exit_cost = notional * (fee_rate + slip_rate)
            total_fees += exit_cost

            # Calculate PnL if we have matching entry
            entry_sig = open_positions.pop(sig.pair_name, None)
            if entry_sig:
                # Mean reversion PnL: spread moved toward mean
                # Long position: profit when spread increases
                # Short position: profit when spread decreases
                spread_change = sig.spread - entry_sig.spread

                if entry_sig.action == SignalAction.ENTER_LONG:
                    # We went long spread (long A, short B), profit if spread up
                    trade_pnl = spread_change * quantity_per_trade
                else:
                    # We went short spread (short A, long B), profit if spread down
                    trade_pnl = -spread_change * quantity_per_trade

                # Subtract transaction costs
                round_trip_fees = (entry_cost if entry_sig else 0) + exit_cost
                net_trade_pnl = trade_pnl - round_trip_fees
                pnls.append(net_trade_pnl)

    # Calculate summary metrics
    total_pnl = sum(pnls) if pnls else 0.0
    net_pnl = total_pnl  # Already net of fees

    # Win rate
    if pnls:
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
    else:
        win_rate = 0.0

    # Sharpe-like ratio
    if len(pnls) >= 2:
        mean_pnl = total_pnl / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        std_pnl = variance**0.5
        sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Max drawdown from cumulative PnL
    max_drawdown = 0.0
    if pnls:
        cumsum = 0.0
        peak = 0.0
        for p in pnls:
            cumsum += p
            if cumsum > peak:
                peak = cumsum
            dd = (peak - cumsum) / max(peak, 1.0)
            if dd > max_drawdown:
                max_drawdown = dd

    # Unique pairs
    unique_pairs = len({s.pair_name for s in signals})
    avg_trades = len(pnls) / unique_pairs if unique_pairs > 0 else 0.0

    return WalkForwardMetrics(
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        total_trades=len(pnls),
        avg_trades_per_pair=avg_trades,
        total_fees=total_fees,
        net_pnl=net_pnl,
        entry_count=entry_count,
        exit_count=exit_count,
    )


def run_walk_forward(
    snapshots: list[dict[str, Any]],
    pairs: list[PairConfig],
    train_count: int,
    test_count: int,
    lookback: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_hold_bars: int = 60,
    cooldown_bars: int = 5,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    quantity_per_trade: float = 10.0,
) -> WalkForwardResult:
    """Run complete walk-forward evaluation.

    1. Split times into TRAIN and TEST
    2. Fit parameters on TRAIN
    3. Generate signals and compute metrics on TEST

    Args:
        snapshots: All snapshots (will be filtered by time)
        pairs: List of pairs to evaluate
        train_count: Number of snapshots for TRAIN
        test_count: Number of snapshots for TEST
        lookback: Z-score lookback
        entry_z: Entry threshold
        exit_z: Exit threshold
        max_hold_bars: Max hold period
        cooldown_bars: Cooldown after exit
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        quantity_per_trade: Quantity per trade

    Returns:
        WalkForwardResult with full evaluation data
    """
    # Get all unique times from snapshots
    all_times = sorted({s["snapshot_time"] for s in snapshots})

    # Split times
    split = split_times(all_times, train_count, test_count)

    logger.info(
        f"Walk-forward split: {split.train_count} TRAIN, {split.test_count} TEST "
        f"(total {split.total_count})"
    )

    if split.train_count == 0:
        logger.warning("No TRAIN data available")
        return WalkForwardResult(
            split=split,
            fitted_params={},
            signals=[],
            train_metrics=None,
            test_metrics=WalkForwardMetrics(
                total_pnl=0,
                sharpe_ratio=0,
                win_rate=0,
                max_drawdown=0,
                total_trades=0,
                avg_trades_per_pair=0,
                total_fees=0,
                net_pnl=0,
                entry_count=0,
                exit_count=0,
            ),
            pair_summaries=[],
            config_used={},
        )

    # Fit on TRAIN
    fitted_params = fit_all_pairs(snapshots, pairs, split.train_times)

    valid_pairs = sum(1 for p in fitted_params.values() if p.is_valid)
    logger.info(f"Fitted {valid_pairs}/{len(pairs)} pairs successfully")

    # Evaluate on TEST
    signals, test_metrics, pair_summaries = evaluate_test_period(
        snapshots=snapshots,
        pairs=pairs,
        fitted_params=fitted_params,
        test_times=split.test_times,
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        max_hold_bars=max_hold_bars,
        cooldown_bars=cooldown_bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        quantity_per_trade=quantity_per_trade,
    )

    logger.info(
        f"TEST metrics: PnL=${test_metrics.total_pnl:.2f}, "
        f"Sharpe={test_metrics.sharpe_ratio:.2f}, "
        f"WR={test_metrics.win_rate:.1%}, "
        f"trades={test_metrics.total_trades}"
    )

    config_used = {
        "train_count": train_count,
        "test_count": test_count,
        "lookback": lookback,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "max_hold_bars": max_hold_bars,
        "cooldown_bars": cooldown_bars,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "quantity_per_trade": quantity_per_trade,
    }

    return WalkForwardResult(
        split=split,
        fitted_params=fitted_params,
        signals=signals,
        train_metrics=None,  # Could compute TRAIN metrics separately if needed
        test_metrics=test_metrics,
        pair_summaries=pair_summaries,
        config_used=config_used,
    )


def result_to_dict(result: WalkForwardResult) -> dict[str, Any]:
    """Convert WalkForwardResult to dict for serialization."""
    return {
        "split": {
            "train_count": result.split.train_count,
            "test_count": result.split.test_count,
            "total_count": result.split.total_count,
            "first_train": result.split.first_train,
            "last_train": result.split.last_train,
            "first_test": result.split.first_test,
            "last_test": result.split.last_test,
        },
        "fitted_params": {
            name: {
                "beta": p.beta,
                "spread_mean": p.spread_mean,
                "spread_std": p.spread_std,
                "train_count": p.train_count,
                "is_valid": p.is_valid,
                "error_msg": p.error_msg,
            }
            for name, p in result.fitted_params.items()
        },
        "test_metrics": {
            "total_pnl": result.test_metrics.total_pnl,
            "sharpe_ratio": result.test_metrics.sharpe_ratio,
            "win_rate": result.test_metrics.win_rate,
            "max_drawdown": result.test_metrics.max_drawdown,
            "total_trades": result.test_metrics.total_trades,
            "avg_trades_per_pair": result.test_metrics.avg_trades_per_pair,
            "total_fees": result.test_metrics.total_fees,
            "net_pnl": result.test_metrics.net_pnl,
            "entry_count": result.test_metrics.entry_count,
            "exit_count": result.test_metrics.exit_count,
        },
        "pair_summaries": result.pair_summaries,
        "config_used": result.config_used,
        "signal_count": len(result.signals),
    }
