"""Statistical arbitrage module for pairs trading."""

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

__all__ = [
    "CandidatePair",
    "PairConfig",
    "PairsConfigError",
    "PairsConfigResult",
    "compute_correlation",
    "discover_pairs",
    "generate_pairs_yaml",
    "load_validated_pairs_config",
    "validate_pair_overlap",
    "validate_pairs_config",
]
