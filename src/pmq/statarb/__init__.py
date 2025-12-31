"""Statistical arbitrage module for pairs trading.

Phase 4.3: Z-score based mean-reversion with walk-forward evaluation.
Phase 4.6: Realistic costs and market constraints.
"""

from pmq.statarb.constraints import (
    ConstraintResult,
    FilterReason,
    apply_market_constraints,
    constraint_result_to_dict,
)
from pmq.statarb.discovery import (
    CandidatePair,
    compute_correlation,
    discover_pairs,
    validate_pair_overlap,
)
from pmq.statarb.pairs_config import (
    PairConfig,
    PairsConfigError,
    PairsConfigResult,
    generate_pairs_yaml,
    load_validated_pairs_config,
    validate_pairs_config,
)
from pmq.statarb.tuning import (
    GridConfig,
    TuningLeaderboard,
    TuningResult,
    export_best_config,
    generate_param_combinations,
    load_grid_config,
    run_grid_search,
    save_leaderboard_csv,
)
from pmq.statarb.walkforward import (
    WalkForwardMetrics,
    WalkForwardResult,
    WalkForwardSplit,
    extract_pair_prices,
    fit_all_pairs,
    run_walk_forward,
    split_times,
)
from pmq.statarb.zscore import (
    FittedParams,
    SignalAction,
    ZScoreSignal,
    fit_pair_params,
    generate_signals,
    ols_beta,
    spread_series,
    zscore_series,
)

__all__ = [
    # Constraints (Phase 4.6)
    "ConstraintResult",
    "FilterReason",
    "apply_market_constraints",
    "constraint_result_to_dict",
    # Discovery
    "CandidatePair",
    "compute_correlation",
    "discover_pairs",
    "validate_pair_overlap",
    # Pairs config
    "PairConfig",
    "PairsConfigError",
    "PairsConfigResult",
    "generate_pairs_yaml",
    "load_validated_pairs_config",
    "validate_pairs_config",
    # Z-score engine
    "FittedParams",
    "SignalAction",
    "ZScoreSignal",
    "fit_pair_params",
    "generate_signals",
    "ols_beta",
    "spread_series",
    "zscore_series",
    # Walk-forward
    "WalkForwardMetrics",
    "WalkForwardResult",
    "WalkForwardSplit",
    "extract_pair_prices",
    "fit_all_pairs",
    "run_walk_forward",
    "split_times",
    # Tuning
    "GridConfig",
    "TuningLeaderboard",
    "TuningResult",
    "export_best_config",
    "generate_param_combinations",
    "load_grid_config",
    "run_grid_search",
    "save_leaderboard_csv",
]
