"""Pairs configuration validation for statistical arbitrage.

Provides strong validation and helpful error messages for pairs config files.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pmq.logging import get_logger

logger = get_logger("statarb.pairs_config")

# Valid market ID pattern:
# - Numeric IDs (e.g., "976939")
# - Hex condition IDs (e.g., "0x...")
# - UUIDs (e.g., "abc-def-...")
MARKET_ID_PATTERN = re.compile(r"^\d+$|^0x[a-fA-F0-9]{40,}$|^[a-fA-F0-9-]{36,}$")


class PairsConfigError(Exception):
    """Error in pairs configuration."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        """Initialize with message and optional list of specific errors."""
        self.errors = errors or []
        full_message = message
        if self.errors:
            full_message += "\n" + "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(full_message)


@dataclass(frozen=True)
class PairConfig:
    """Validated configuration for a single stat-arb pair."""

    market_a_id: str
    market_b_id: str
    name: str
    correlation: float = 1.0
    enabled: bool = True
    min_liquidity: float | None = None
    max_spread: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "market_a_id": self.market_a_id,
            "market_b_id": self.market_b_id,
            "name": self.name,
            "correlation": self.correlation,
            "enabled": self.enabled,
        }
        if self.min_liquidity is not None:
            d["min_liquidity"] = self.min_liquidity
        if self.max_spread is not None:
            d["max_spread"] = self.max_spread
        return d


@dataclass
class PairsConfigResult:
    """Result of loading and validating pairs configuration."""

    pairs: list[PairConfig] = field(default_factory=list)
    enabled_pairs: list[PairConfig] = field(default_factory=list)
    disabled_pairs: list[PairConfig] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config_path: str = ""
    config_hash: str = ""

    @property
    def has_enabled_pairs(self) -> bool:
        """Check if there are any enabled pairs."""
        return len(self.enabled_pairs) > 0


def _validate_market_id(market_id: str, field_name: str, pair_index: int) -> list[str]:
    """Validate a market ID and return list of errors."""
    errors: list[str] = []
    if not market_id:
        errors.append(f"Pair {pair_index + 1}: {field_name} is required")
    elif not isinstance(market_id, str):
        errors.append(f"Pair {pair_index + 1}: {field_name} must be a string")
    elif not MARKET_ID_PATTERN.match(market_id):
        errors.append(
            f"Pair {pair_index + 1}: {field_name} '{market_id[:20]}...' "
            f"does not look like a valid market ID. "
            f"Expected format: 0x... (condition ID) or UUID"
        )
    return errors


def _validate_pair(data: dict[str, Any], index: int) -> tuple[PairConfig | None, list[str]]:
    """Validate a single pair configuration.

    Args:
        data: Raw pair data from YAML
        index: Pair index for error messages

    Returns:
        Tuple of (PairConfig if valid, list of errors)
    """
    errors: list[str] = []

    # Required fields
    market_a_id = data.get("market_a_id", "")
    market_b_id = data.get("market_b_id", "")

    errors.extend(_validate_market_id(market_a_id, "market_a_id", index))
    errors.extend(_validate_market_id(market_b_id, "market_b_id", index))

    # Check for duplicate IDs
    if market_a_id and market_b_id and market_a_id == market_b_id:
        errors.append(f"Pair {index + 1}: market_a_id and market_b_id cannot be the same")

    # Name (optional but recommended)
    name = data.get("name", "")
    if not name:
        name = f"Pair_{index + 1}"

    # Correlation (optional, default 1.0)
    correlation = data.get("correlation", 1.0)
    if not isinstance(correlation, (int, float)):
        errors.append(f"Pair {index + 1}: correlation must be a number")
        correlation = 1.0
    elif correlation < -1.0 or correlation > 1.0:
        errors.append(f"Pair {index + 1}: correlation must be between -1.0 and 1.0")

    # Enabled (optional, default True)
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        errors.append(f"Pair {index + 1}: enabled must be true or false")
        enabled = True

    # Optional constraints
    min_liquidity = data.get("min_liquidity")
    if min_liquidity is not None and not isinstance(min_liquidity, (int, float)):
        errors.append(f"Pair {index + 1}: min_liquidity must be a number")
        min_liquidity = None

    max_spread = data.get("max_spread")
    if max_spread is not None and not isinstance(max_spread, (int, float)):
        errors.append(f"Pair {index + 1}: max_spread must be a number")
        max_spread = None

    if errors:
        return None, errors

    return (
        PairConfig(
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            name=name,
            correlation=float(correlation),
            enabled=enabled,
            min_liquidity=float(min_liquidity) if min_liquidity else None,
            max_spread=float(max_spread) if max_spread else None,
        ),
        [],
    )


def validate_pairs_config(data: dict[str, Any]) -> tuple[list[PairConfig], list[str]]:
    """Validate pairs configuration data.

    Args:
        data: Parsed YAML data

    Returns:
        Tuple of (list of valid PairConfig, list of errors)
    """
    errors: list[str] = []
    pairs: list[PairConfig] = []

    if not data:
        errors.append("Configuration file is empty")
        return pairs, errors

    if not isinstance(data, dict):
        errors.append("Configuration must be a YAML mapping")
        return pairs, errors

    pairs_data = data.get("pairs")
    if pairs_data is None:
        errors.append("Missing 'pairs' key in configuration")
        return pairs, errors

    if not isinstance(pairs_data, list):
        errors.append("'pairs' must be a list")
        return pairs, errors

    if len(pairs_data) == 0:
        errors.append("'pairs' list is empty")
        return pairs, errors

    # Check for duplicates
    seen_ids: set[tuple[str, str]] = set()

    for i, pair_data in enumerate(pairs_data):
        if not isinstance(pair_data, dict):
            errors.append(f"Pair {i + 1}: must be a mapping with market_a_id, market_b_id, etc.")
            continue

        pair, pair_errors = _validate_pair(pair_data, i)
        if pair_errors:
            errors.extend(pair_errors)
            continue

        if pair is None:
            continue

        # Check for duplicates (order-independent)
        pair_key = tuple(sorted([pair.market_a_id, pair.market_b_id]))
        if pair_key in seen_ids:
            errors.append(
                f"Pair {i + 1}: Duplicate pair - markets {pair.market_a_id[:16]}... "
                f"and {pair.market_b_id[:16]}... already defined"
            )
            continue
        seen_ids.add(pair_key)

        pairs.append(pair)

    return pairs, errors


def _compute_config_hash(content: str) -> str:
    """Compute a hash of the config content for reproducibility tracking."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def load_validated_pairs_config(path: Path | str) -> PairsConfigResult:
    """Load and validate pairs configuration from a YAML file.

    Args:
        path: Path to the pairs configuration file

    Returns:
        PairsConfigResult with validated pairs and metadata

    Raises:
        PairsConfigError: If the configuration is invalid
    """
    path = Path(path)

    if not path.exists():
        raise PairsConfigError(
            f"Pairs config file not found: {path}\n\n"
            f"To generate a starter pairs file, run:\n"
            f"  pmq statarb pairs suggest --last-times 30 --interval 60 --out {path}\n\n"
            f"Or create manually following the example in config/pairs.yml"
        )

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        raise PairsConfigError(f"Failed to read pairs config file {path}: {e}") from e

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise PairsConfigError(f"Invalid YAML in {path}: {e}") from e

    pairs, errors = validate_pairs_config(data)

    if errors:
        raise PairsConfigError(f"Invalid pairs configuration in {path}:", errors)

    result = PairsConfigResult(
        pairs=pairs,
        enabled_pairs=[p for p in pairs if p.enabled],
        disabled_pairs=[p for p in pairs if not p.enabled],
        config_path=str(path),
        config_hash=_compute_config_hash(content),
    )

    # Add warnings
    if result.disabled_pairs:
        result.warnings.append(
            f"{len(result.disabled_pairs)} pair(s) are disabled and will be skipped"
        )

    if not result.has_enabled_pairs:
        raise PairsConfigError(
            f"No enabled pairs in {path}. "
            f"All {len(pairs)} pair(s) have 'enabled: false'. "
            f"Enable at least one pair to run statarb."
        )

    logger.info(
        f"Loaded {len(result.enabled_pairs)} enabled pairs from {path} (hash: {result.config_hash})"
    )

    return result


def generate_pairs_yaml(pairs: list[PairConfig], header_comment: str = "") -> str:
    """Generate YAML content for pairs configuration.

    Args:
        pairs: List of pair configurations
        header_comment: Optional header comment

    Returns:
        YAML string
    """
    lines = [
        "# Statistical Arbitrage Pairs Configuration",
        "#",
        "# Define pairs of correlated markets to monitor for spread divergence.",
        "# When the spread exceeds entry_threshold, a signal is generated.",
        "#",
        "# Schema:",
        "#   market_a_id: First market ID (required)",
        "#   market_b_id: Second market ID (required)",
        "#   name: Human-readable name (optional)",
        "#   correlation: 1.0 for same direction, -1.0 for inverse (default: 1.0)",
        "#   enabled: true/false to enable/disable pair (default: true)",
        "#   min_liquidity: Minimum liquidity threshold (optional)",
        "#   max_spread: Maximum spread to consider (optional)",
        "#",
    ]

    if header_comment:
        lines.append(f"# {header_comment}")
        lines.append("#")

    lines.append("")
    lines.append("pairs:")

    if not pairs:
        lines.append("  # No pairs configured. Add pairs like:")
        lines.append('  # - market_a_id: "0x..."')
        lines.append('  #   market_b_id: "0x..."')
        lines.append('  #   name: "My Pair"')
        lines.append("  #   correlation: 1.0")
        lines.append("  #   enabled: true")
    else:
        for pair in pairs:
            lines.append(f'  - market_a_id: "{pair.market_a_id}"')
            lines.append(f'    market_b_id: "{pair.market_b_id}"')
            lines.append(f'    name: "{pair.name}"')
            lines.append(f"    correlation: {pair.correlation}")
            lines.append(f"    enabled: {str(pair.enabled).lower()}")
            if pair.min_liquidity is not None:
                lines.append(f"    min_liquidity: {pair.min_liquidity}")
            if pair.max_spread is not None:
                lines.append(f"    max_spread: {pair.max_spread}")
            lines.append("")

    return "\n".join(lines)
