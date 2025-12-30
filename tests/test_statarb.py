"""Tests for statarb pairs configuration and CLI commands."""

import tempfile
from pathlib import Path

import pytest
import yaml

from pmq.statarb import (
    PairConfig,
    PairsConfigError,
    PairsConfigResult,
    load_validated_pairs_config,
    validate_pairs_config,
)
from pmq.statarb.pairs_config import generate_pairs_yaml


class TestPairsConfigValidation:
    """Tests for pairs configuration validation."""

    def test_validate_valid_config(self):
        """Test validation of a valid pairs config."""
        data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "789012",
                    "name": "Test Pair",
                    "correlation": 1.0,
                    "enabled": True,
                }
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 0
        assert len(pairs) == 1
        assert pairs[0].market_a_id == "123456"
        assert pairs[0].market_b_id == "789012"
        assert pairs[0].name == "Test Pair"
        assert pairs[0].correlation == 1.0
        assert pairs[0].enabled is True

    def test_validate_config_with_hex_ids(self):
        """Test validation accepts hex condition IDs."""
        data = {
            "pairs": [
                {
                    "market_a_id": "0x1234567890abcdef1234567890abcdef12345678",
                    "market_b_id": "0xfedcba0987654321fedcba0987654321fedcba09",
                    "name": "Hex ID Pair",
                }
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 0
        assert len(pairs) == 1

    def test_validate_config_with_uuid_ids(self):
        """Test validation accepts UUID-like IDs."""
        data = {
            "pairs": [
                {
                    "market_a_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "market_b_id": "f9e8d7c6-b5a4-3210-fedc-ba0987654321",
                    "name": "UUID Pair",
                }
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 0
        assert len(pairs) == 1

    def test_validate_missing_pairs_key(self):
        """Test error when pairs key is missing."""
        # Empty dict is treated as empty config
        data = {}
        pairs, errors = validate_pairs_config(data)
        assert len(errors) == 1
        assert len(pairs) == 0

        # Dict without pairs key is missing key
        data_no_pairs = {"other_key": "value"}
        pairs2, errors2 = validate_pairs_config(data_no_pairs)
        assert len(errors2) == 1
        assert "Missing 'pairs' key" in errors2[0]
        assert len(pairs2) == 0

    def test_validate_empty_pairs_list(self):
        """Test error when pairs list is empty."""
        data = {"pairs": []}
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 1
        assert "'pairs' list is empty" in errors[0]

    def test_validate_missing_market_ids(self):
        """Test error when market IDs are missing."""
        data = {"pairs": [{"name": "Incomplete Pair"}]}
        pairs, errors = validate_pairs_config(data)

        assert len(errors) >= 2
        assert any("market_a_id is required" in e for e in errors)
        assert any("market_b_id is required" in e for e in errors)

    def test_validate_same_market_ids(self):
        """Test error when both market IDs are the same."""
        data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "123456",
                    "name": "Same IDs",
                }
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 1
        assert "cannot be the same" in errors[0]

    def test_validate_duplicate_pairs(self):
        """Test error on duplicate pairs."""
        data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "789012",
                    "name": "Pair 1",
                },
                {
                    "market_a_id": "789012",  # Reversed IDs should still be duplicate
                    "market_b_id": "123456",
                    "name": "Pair 2",
                },
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 1
        assert "Duplicate pair" in errors[0]

    def test_validate_correlation_bounds(self):
        """Test error when correlation is out of bounds."""
        data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "789012",
                    "correlation": 1.5,  # Out of bounds
                }
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 1
        assert "between -1.0 and 1.0" in errors[0]

    def test_validate_disabled_pair(self):
        """Test validation of disabled pairs."""
        data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "789012",
                    "enabled": False,
                }
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 0
        assert len(pairs) == 1
        assert pairs[0].enabled is False

    def test_validate_optional_fields(self):
        """Test validation with optional fields."""
        data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "789012",
                    "min_liquidity": 100.0,
                    "max_spread": 0.05,
                }
            ]
        }
        pairs, errors = validate_pairs_config(data)

        assert len(errors) == 0
        assert pairs[0].min_liquidity == 100.0
        assert pairs[0].max_spread == 0.05


class TestLoadValidatedPairsConfig:
    """Tests for loading pairs config from file."""

    def test_load_valid_file(self):
        """Test loading a valid YAML file."""
        config_data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "789012",
                    "name": "Test Pair",
                    "correlation": 1.0,
                    "enabled": True,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            path = Path(f.name)

        try:
            result = load_validated_pairs_config(path)

            assert isinstance(result, PairsConfigResult)
            assert len(result.enabled_pairs) == 1
            assert result.config_path == str(path)
            assert len(result.config_hash) > 0
        finally:
            path.unlink()

    def test_load_missing_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(PairsConfigError) as exc_info:
            load_validated_pairs_config(Path("/nonexistent/path.yml"))

        assert "not found" in str(exc_info.value)
        assert "pmq statarb pairs suggest" in str(exc_info.value)

    def test_load_invalid_yaml(self):
        """Test error with invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(PairsConfigError) as exc_info:
                load_validated_pairs_config(path)

            assert "Invalid YAML" in str(exc_info.value)
        finally:
            path.unlink()

    def test_load_no_enabled_pairs(self):
        """Test error when all pairs are disabled."""
        config_data = {
            "pairs": [
                {
                    "market_a_id": "123456",
                    "market_b_id": "789012",
                    "enabled": False,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(config_data, f)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(PairsConfigError) as exc_info:
                load_validated_pairs_config(path)

            assert "No enabled pairs" in str(exc_info.value)
        finally:
            path.unlink()

    def test_result_has_enabled_pairs_property(self):
        """Test the has_enabled_pairs property."""
        result = PairsConfigResult(
            pairs=[],
            enabled_pairs=[],
            disabled_pairs=[],
        )
        assert result.has_enabled_pairs is False

        result_with_pairs = PairsConfigResult(
            pairs=[
                PairConfig(
                    market_a_id="123",
                    market_b_id="456",
                    name="test",
                )
            ],
            enabled_pairs=[
                PairConfig(
                    market_a_id="123",
                    market_b_id="456",
                    name="test",
                )
            ],
            disabled_pairs=[],
        )
        assert result_with_pairs.has_enabled_pairs is True


class TestGeneratePairsYaml:
    """Tests for YAML generation."""

    def test_generate_empty_pairs(self):
        """Test generating YAML with no pairs."""
        yaml_content = generate_pairs_yaml([])

        assert "pairs:" in yaml_content
        assert "# No pairs configured" in yaml_content

    def test_generate_with_pairs(self):
        """Test generating YAML with pairs."""
        pairs = [
            PairConfig(
                market_a_id="123456",
                market_b_id="789012",
                name="Test Pair",
                correlation=1.0,
                enabled=True,
            )
        ]

        yaml_content = generate_pairs_yaml(pairs)

        assert 'market_a_id: "123456"' in yaml_content
        assert 'market_b_id: "789012"' in yaml_content
        assert 'name: "Test Pair"' in yaml_content
        assert "correlation: 1.0" in yaml_content
        assert "enabled: true" in yaml_content

    def test_generate_with_header_comment(self):
        """Test generating YAML with custom header."""
        pairs = [
            PairConfig(
                market_a_id="123",
                market_b_id="456",
                name="Test",
            )
        ]

        yaml_content = generate_pairs_yaml(pairs, header_comment="Custom header")

        assert "# Custom header" in yaml_content

    def test_generated_yaml_is_valid(self):
        """Test that generated YAML can be parsed back."""
        pairs = [
            PairConfig(
                market_a_id="123456",
                market_b_id="789012",
                name="Test Pair",
                correlation=-0.5,
                enabled=True,
                min_liquidity=100.0,
            )
        ]

        yaml_content = generate_pairs_yaml(pairs)
        parsed = yaml.safe_load(yaml_content)

        assert "pairs" in parsed
        assert len(parsed["pairs"]) == 1
        assert parsed["pairs"][0]["market_a_id"] == "123456"


class TestPairConfigDataclass:
    """Tests for PairConfig dataclass."""

    def test_create_pair_config(self):
        """Test creating a PairConfig."""
        pair = PairConfig(
            market_a_id="123",
            market_b_id="456",
            name="Test",
            correlation=1.0,
            enabled=True,
        )

        assert pair.market_a_id == "123"
        assert pair.market_b_id == "456"
        assert pair.name == "Test"
        assert pair.correlation == 1.0
        assert pair.enabled is True

    def test_pair_config_to_dict(self):
        """Test converting PairConfig to dict."""
        pair = PairConfig(
            market_a_id="123",
            market_b_id="456",
            name="Test",
            correlation=1.0,
            enabled=True,
            min_liquidity=100.0,
        )

        d = pair.to_dict()

        assert d["market_a_id"] == "123"
        assert d["market_b_id"] == "456"
        assert d["name"] == "Test"
        assert d["correlation"] == 1.0
        assert d["enabled"] is True
        assert d["min_liquidity"] == 100.0
        assert "max_spread" not in d  # None values excluded

    def test_pair_config_frozen(self):
        """Test that PairConfig is immutable (frozen)."""
        pair = PairConfig(
            market_a_id="123",
            market_b_id="456",
            name="Test",
        )

        # Frozen dataclass should raise an error on modification
        with pytest.raises(AttributeError):
            pair.name = "Modified"  # type: ignore


class TestEvaluationPipelineIntegration:
    """Tests for evaluation pipeline statarb integration."""

    def test_eval_run_statarb_without_pairs_raises(self):
        """Test that eval run for statarb without pairs raises ValueError."""
        # Note: This test requires the full evaluation pipeline setup
        # For now, we test the validation logic directly
        from pmq.evaluation.pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline()

        # This should raise ValueError for statarb without pairs
        with pytest.raises(ValueError) as exc_info:
            pipeline.run(
                strategy_name="statarb",
                strategy_version="v1",
                window_mode="last_times",
                window_value=5,
                interval_seconds=60,
                pairs_config=None,  # Missing pairs config
            )

        assert "statarb strategy requires --pairs" in str(exc_info.value)

    def test_eval_run_statarb_with_invalid_pairs_raises(self):
        """Test that eval run with invalid pairs config raises ValueError."""
        from pmq.evaluation.pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline()

        # This should raise ValueError for non-existent pairs file
        with pytest.raises(ValueError) as exc_info:
            pipeline.run(
                strategy_name="statarb",
                strategy_version="v1",
                window_mode="last_times",
                window_value=5,
                interval_seconds=60,
                pairs_config="/nonexistent/pairs.yml",
            )

        assert "Invalid pairs config" in str(exc_info.value)


class TestDiscoveryModule:
    """Tests for the pair discovery module."""

    def test_compute_correlation_positive(self):
        """Test correlation computation for positively correlated series."""
        from pmq.statarb.discovery import compute_correlation

        # Perfect positive correlation
        prices_a = [0.5, 0.6, 0.7, 0.8, 0.9]
        prices_b = [0.5, 0.6, 0.7, 0.8, 0.9]
        corr = compute_correlation(prices_a, prices_b)
        assert corr == pytest.approx(1.0, abs=0.001)

    def test_compute_correlation_negative(self):
        """Test correlation computation for negatively correlated series."""
        from pmq.statarb.discovery import compute_correlation

        # Perfect negative correlation
        prices_a = [0.5, 0.6, 0.7, 0.8, 0.9]
        prices_b = [0.9, 0.8, 0.7, 0.6, 0.5]
        corr = compute_correlation(prices_a, prices_b)
        assert corr == pytest.approx(-1.0, abs=0.001)

    def test_compute_correlation_insufficient_data(self):
        """Test correlation returns 0 for insufficient data."""
        from pmq.statarb.discovery import compute_correlation

        # Less than 3 data points
        prices_a = [0.5, 0.6]
        prices_b = [0.5, 0.6]
        corr = compute_correlation(prices_a, prices_b)
        assert corr == 0.0

    def test_compute_correlation_constant_prices(self):
        """Test correlation returns 0 for constant prices."""
        from pmq.statarb.discovery import compute_correlation

        # Constant prices (zero variance)
        prices_a = [0.5, 0.5, 0.5, 0.5]
        prices_b = [0.6, 0.7, 0.8, 0.9]
        corr = compute_correlation(prices_a, prices_b)
        assert corr == 0.0

    def test_discover_pairs_deterministic(self):
        """Test that discover_pairs produces deterministic output."""
        from pmq.statarb.discovery import discover_pairs

        # Create snapshot data with known correlation
        snapshots = [
            {"market_id": "A", "snapshot_time": "t1", "yes_price": 0.5},
            {"market_id": "A", "snapshot_time": "t2", "yes_price": 0.6},
            {"market_id": "A", "snapshot_time": "t3", "yes_price": 0.7},
            {"market_id": "A", "snapshot_time": "t4", "yes_price": 0.8},
            {"market_id": "A", "snapshot_time": "t5", "yes_price": 0.9},
            {"market_id": "B", "snapshot_time": "t1", "yes_price": 0.5},
            {"market_id": "B", "snapshot_time": "t2", "yes_price": 0.6},
            {"market_id": "B", "snapshot_time": "t3", "yes_price": 0.7},
            {"market_id": "B", "snapshot_time": "t4", "yes_price": 0.8},
            {"market_id": "B", "snapshot_time": "t5", "yes_price": 0.9},
        ]
        markets = {
            "A": {"question": "Market A?"},
            "B": {"question": "Market B?"},
        }

        # Run twice and ensure same output
        result1 = discover_pairs(snapshots, markets, min_overlap=3, top=10, min_correlation=0.3)
        result2 = discover_pairs(snapshots, markets, min_overlap=3, top=10, min_correlation=0.3)

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2, strict=True):
            assert c1.market_a_id == c2.market_a_id
            assert c1.market_b_id == c2.market_b_id
            assert c1.correlation == c2.correlation

    def test_discover_pairs_respects_min_overlap(self):
        """Test that discover_pairs respects min_overlap requirement."""
        from pmq.statarb.discovery import discover_pairs

        # Only 3 overlapping times
        snapshots = [
            {"market_id": "A", "snapshot_time": "t1", "yes_price": 0.5},
            {"market_id": "A", "snapshot_time": "t2", "yes_price": 0.6},
            {"market_id": "A", "snapshot_time": "t3", "yes_price": 0.7},
            {"market_id": "B", "snapshot_time": "t1", "yes_price": 0.5},
            {"market_id": "B", "snapshot_time": "t2", "yes_price": 0.6},
            {"market_id": "B", "snapshot_time": "t3", "yes_price": 0.7},
        ]
        markets = {"A": {}, "B": {}}

        # min_overlap=5 should find nothing
        result = discover_pairs(snapshots, markets, min_overlap=5, top=10, min_correlation=0.3)
        assert len(result) == 0

        # min_overlap=3 should find the pair
        result = discover_pairs(snapshots, markets, min_overlap=3, top=10, min_correlation=0.3)
        assert len(result) == 1

    def test_validate_pair_overlap_valid(self):
        """Test validate_pair_overlap returns valid for sufficient overlap."""
        from pmq.statarb.discovery import validate_pair_overlap

        pair = PairConfig(
            market_a_id="A",
            market_b_id="B",
            name="Test",
        )
        snapshots = [
            {"market_id": "A", "snapshot_time": "t1"},
            {"market_id": "A", "snapshot_time": "t2"},
            {"market_id": "A", "snapshot_time": "t3"},
            {"market_id": "B", "snapshot_time": "t1"},
            {"market_id": "B", "snapshot_time": "t2"},
            {"market_id": "B", "snapshot_time": "t3"},
        ]

        result = validate_pair_overlap(pair, snapshots, min_overlap=3)
        assert result["valid"] is True
        assert result["overlap_count"] == 3

    def test_validate_pair_overlap_insufficient(self):
        """Test validate_pair_overlap returns invalid for insufficient overlap."""
        from pmq.statarb.discovery import validate_pair_overlap

        pair = PairConfig(
            market_a_id="A",
            market_b_id="B",
            name="Test",
        )
        snapshots = [
            {"market_id": "A", "snapshot_time": "t1"},
            {"market_id": "A", "snapshot_time": "t2"},
            {"market_id": "B", "snapshot_time": "t3"},  # Different time
            {"market_id": "B", "snapshot_time": "t4"},  # Different time
        ]

        result = validate_pair_overlap(pair, snapshots, min_overlap=3)
        assert result["valid"] is False
        assert "Insufficient overlap" in result["reason"]

    def test_validate_pair_overlap_missing_market(self):
        """Test validate_pair_overlap returns invalid for missing market."""
        from pmq.statarb.discovery import validate_pair_overlap

        pair = PairConfig(
            market_a_id="A",
            market_b_id="C",  # Market C not in snapshots
            name="Test",
        )
        snapshots = [
            {"market_id": "A", "snapshot_time": "t1"},
            {"market_id": "B", "snapshot_time": "t1"},
        ]

        result = validate_pair_overlap(pair, snapshots, min_overlap=1)
        assert result["valid"] is False
        assert "no snapshots" in result["reason"]

    def test_candidate_pair_to_pair_config(self):
        """Test CandidatePair conversion to PairConfig."""
        from pmq.statarb.discovery import CandidatePair

        candidate = CandidatePair(
            market_a_id="123",
            market_b_id="456",
            market_a_question="Question A?",
            market_b_question="Question B?",
            correlation=0.85,
            overlap_count=20,
            avg_spread=0.05,
            max_abs_spread=0.15,
        )

        pair_config = candidate.to_pair_config()

        assert pair_config.market_a_id == "123"
        assert pair_config.market_b_id == "456"
        assert pair_config.correlation == 0.85
        assert pair_config.enabled is True
