"""Statistical arbitrage module for pairs trading."""

from pmq.statarb.pairs_config import (
    PairConfig,
    PairsConfigError,
    PairsConfigResult,
    load_validated_pairs_config,
    validate_pairs_config,
)

__all__ = [
    "PairConfig",
    "PairsConfigError",
    "PairsConfigResult",
    "load_validated_pairs_config",
    "validate_pairs_config",
]
