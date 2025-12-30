"""Grid search tuning for statistical arbitrage parameters.

This module provides:
- Grid configuration loading from YAML
- Parallel-safe grid search over parameter combinations
- Leaderboard generation and best config export

All functions are deterministic: same input produces same output ordering.
"""

import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pmq.logging import get_logger
from pmq.statarb.pairs_config import PairConfig
from pmq.statarb.walkforward import run_walk_forward

logger = get_logger("statarb.tuning")


@dataclass
class GridConfig:
    """Configuration for grid search."""

    lookback: list[int] = field(default_factory=lambda: [20, 30, 50])
    entry_z: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    exit_z: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    max_hold_bars: list[int] = field(default_factory=lambda: [30, 60, 120])
    cooldown_bars: list[int] = field(default_factory=lambda: [5])
    fee_bps: list[float] = field(default_factory=lambda: [0.0])
    slippage_bps: list[float] = field(default_factory=lambda: [0.0])

    @property
    def total_combinations(self) -> int:
        """Total number of parameter combinations."""
        return (
            len(self.lookback)
            * len(self.entry_z)
            * len(self.exit_z)
            * len(self.max_hold_bars)
            * len(self.cooldown_bars)
            * len(self.fee_bps)
            * len(self.slippage_bps)
        )


def load_grid_config(path: Path | str) -> GridConfig:
    """Load grid configuration from YAML file.

    Expected format:
    ```yaml
    grid:
      lookback: [20, 30, 50]
      entry_z: [1.5, 2.0, 2.5]
      exit_z: [0.3, 0.5, 0.7]
      max_hold_bars: [30, 60, 120]
      cooldown_bars: [5]
      fee_bps: [0.0]
      slippage_bps: [0.0]
    ```

    Args:
        path: Path to grid config YAML file

    Returns:
        GridConfig with loaded parameters
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Grid config not found at {path}, using defaults")
        return GridConfig()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "grid" not in data:
        logger.warning(f"No 'grid' key in {path}, using defaults")
        return GridConfig()

    grid_data = data["grid"]
    return GridConfig(
        lookback=grid_data.get("lookback", GridConfig().lookback),
        entry_z=grid_data.get("entry_z", GridConfig().entry_z),
        exit_z=grid_data.get("exit_z", GridConfig().exit_z),
        max_hold_bars=grid_data.get("max_hold_bars", GridConfig().max_hold_bars),
        cooldown_bars=grid_data.get("cooldown_bars", GridConfig().cooldown_bars),
        fee_bps=grid_data.get("fee_bps", GridConfig().fee_bps),
        slippage_bps=grid_data.get("slippage_bps", GridConfig().slippage_bps),
    )


@dataclass
class TuningResult:
    """Result from a single parameter combination."""

    params: dict[str, Any]
    pnl: float
    sharpe: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    net_pnl: float
    rank: int = 0  # Set after sorting


@dataclass
class TuningLeaderboard:
    """Leaderboard of tuning results."""

    results: list[TuningResult]
    best_params: dict[str, Any]
    grid_config: GridConfig
    train_count: int
    test_count: int
    total_combinations: int
    pairs_count: int


def generate_param_combinations(grid: GridConfig) -> list[dict[str, Any]]:
    """Generate all parameter combinations from grid config.

    Combinations are sorted deterministically to ensure reproducible ordering.

    Args:
        grid: Grid configuration

    Returns:
        List of parameter dicts, sorted deterministically
    """
    combinations = list(itertools.product(
        sorted(grid.lookback),
        sorted(grid.entry_z),
        sorted(grid.exit_z),
        sorted(grid.max_hold_bars),
        sorted(grid.cooldown_bars),
        sorted(grid.fee_bps),
        sorted(grid.slippage_bps),
    ))

    return [
        {
            "lookback": lookback,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "max_hold_bars": max_hold_bars,
            "cooldown_bars": cooldown_bars,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
        }
        for lookback, entry_z, exit_z, max_hold_bars, cooldown_bars, fee_bps, slippage_bps
        in combinations
    ]


def run_grid_search(
    snapshots: list[dict[str, Any]],
    pairs: list[PairConfig],
    train_count: int,
    test_count: int,
    grid: GridConfig,
    quantity_per_trade: float = 10.0,
    top_k: int = 10,
) -> TuningLeaderboard:
    """Run grid search over parameter combinations.

    For each combination:
    1. Run walk-forward evaluation with TRAIN/TEST split
    2. Record TEST metrics only (to prevent overfitting)
    3. Rank by Sharpe ratio (primary), then PnL (secondary)

    Args:
        snapshots: All snapshot data
        pairs: Pairs to evaluate
        train_count: TRAIN snapshot count
        test_count: TEST snapshot count
        grid: Grid configuration
        quantity_per_trade: Quantity per trade
        top_k: Number of top results to keep

    Returns:
        TuningLeaderboard with ranked results
    """
    param_combinations = generate_param_combinations(grid)
    total = len(param_combinations)

    logger.info(f"Starting grid search: {total} combinations")

    results: list[TuningResult] = []

    for i, params in enumerate(param_combinations):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Running combination {i + 1}/{total}")

        try:
            wf_result = run_walk_forward(
                snapshots=snapshots,
                pairs=pairs,
                train_count=train_count,
                test_count=test_count,
                lookback=params["lookback"],
                entry_z=params["entry_z"],
                exit_z=params["exit_z"],
                max_hold_bars=params["max_hold_bars"],
                cooldown_bars=params["cooldown_bars"],
                fee_bps=params["fee_bps"],
                slippage_bps=params["slippage_bps"],
                quantity_per_trade=quantity_per_trade,
            )

            results.append(TuningResult(
                params=params,
                pnl=wf_result.test_metrics.total_pnl,
                sharpe=wf_result.test_metrics.sharpe_ratio,
                win_rate=wf_result.test_metrics.win_rate,
                max_drawdown=wf_result.test_metrics.max_drawdown,
                total_trades=wf_result.test_metrics.total_trades,
                net_pnl=wf_result.test_metrics.net_pnl,
            ))
        except Exception as e:
            logger.warning(f"Failed combination {i + 1}: {e}")
            results.append(TuningResult(
                params=params,
                pnl=0.0,
                sharpe=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                total_trades=0,
                net_pnl=0.0,
            ))

    # Sort by Sharpe (desc), then PnL (desc), then param values for determinism
    results.sort(
        key=lambda r: (
            -r.sharpe,
            -r.pnl,
            r.params["lookback"],
            r.params["entry_z"],
            r.params["exit_z"],
            r.params["max_hold_bars"],
        )
    )

    # Assign ranks
    for i, result in enumerate(results):
        result.rank = i + 1

    # Get top k
    top_results = results[:top_k]
    best_params = top_results[0].params if top_results else {}

    logger.info(
        f"Grid search complete. Best: Sharpe={top_results[0].sharpe:.3f}, "
        f"PnL=${top_results[0].pnl:.2f}" if top_results else "No valid results"
    )

    return TuningLeaderboard(
        results=top_results,
        best_params=best_params,
        grid_config=grid,
        train_count=train_count,
        test_count=test_count,
        total_combinations=total,
        pairs_count=len(pairs),
    )


def save_leaderboard_csv(leaderboard: TuningLeaderboard, path: Path | str) -> None:
    """Save leaderboard to CSV file.

    Args:
        leaderboard: Tuning leaderboard
        path: Output CSV path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "rank",
            "sharpe",
            "pnl",
            "net_pnl",
            "win_rate",
            "max_drawdown",
            "trades",
            "lookback",
            "entry_z",
            "exit_z",
            "max_hold_bars",
            "cooldown_bars",
            "fee_bps",
            "slippage_bps",
        ])

        # Results
        for result in leaderboard.results:
            writer.writerow([
                result.rank,
                f"{result.sharpe:.4f}",
                f"{result.pnl:.2f}",
                f"{result.net_pnl:.2f}",
                f"{result.win_rate:.4f}",
                f"{result.max_drawdown:.4f}",
                result.total_trades,
                result.params["lookback"],
                result.params["entry_z"],
                result.params["exit_z"],
                result.params["max_hold_bars"],
                result.params["cooldown_bars"],
                result.params["fee_bps"],
                result.params["slippage_bps"],
            ])

    logger.info(f"Saved leaderboard to {path}")


def export_best_config(leaderboard: TuningLeaderboard, path: Path | str) -> None:
    """Export best configuration to YAML file.

    Args:
        leaderboard: Tuning leaderboard
        path: Output YAML path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not leaderboard.best_params:
        logger.warning("No best params to export")
        return

    config = {
        "# Generated by pmq statarb tune": None,
        "statarb": {
            "lookback": leaderboard.best_params["lookback"],
            "entry_z": leaderboard.best_params["entry_z"],
            "exit_z": leaderboard.best_params["exit_z"],
            "max_hold_bars": leaderboard.best_params["max_hold_bars"],
            "cooldown_bars": leaderboard.best_params["cooldown_bars"],
            "fee_bps": leaderboard.best_params["fee_bps"],
            "slippage_bps": leaderboard.best_params["slippage_bps"],
        },
        "tuning_metadata": {
            "train_count": leaderboard.train_count,
            "test_count": leaderboard.test_count,
            "combinations_tested": leaderboard.total_combinations,
            "pairs_count": leaderboard.pairs_count,
            "best_sharpe": leaderboard.results[0].sharpe if leaderboard.results else 0,
            "best_pnl": leaderboard.results[0].pnl if leaderboard.results else 0,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        # Write header comment manually
        f.write("# Generated by pmq statarb tune\n")
        f.write("# Best configuration from grid search\n\n")
        yaml.dump(
            {"statarb": config["statarb"], "tuning_metadata": config["tuning_metadata"]},
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    logger.info(f"Exported best config to {path}")


def leaderboard_to_dict(leaderboard: TuningLeaderboard) -> dict[str, Any]:
    """Convert leaderboard to dict for JSON serialization."""
    return {
        "results": [
            {
                "rank": r.rank,
                "params": r.params,
                "sharpe": r.sharpe,
                "pnl": r.pnl,
                "net_pnl": r.net_pnl,
                "win_rate": r.win_rate,
                "max_drawdown": r.max_drawdown,
                "total_trades": r.total_trades,
            }
            for r in leaderboard.results
        ],
        "best_params": leaderboard.best_params,
        "metadata": {
            "train_count": leaderboard.train_count,
            "test_count": leaderboard.test_count,
            "total_combinations": leaderboard.total_combinations,
            "pairs_count": leaderboard.pairs_count,
        },
    }
