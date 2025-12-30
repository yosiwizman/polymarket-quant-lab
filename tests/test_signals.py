"""Tests for signal detection strategies."""


import pytest
from pydantic import ValidationError

from pmq.config import ArbitrageConfig, StatArbConfig
from pmq.models import ArbitrageSignal, GammaMarket, StatArbSignal
from pmq.strategies.arb import ArbitrageScanner, find_arbitrage_opportunities
from pmq.strategies.statarb import StatArbPair, StatArbScanner


@pytest.fixture
def arb_config():
    """Arbitrage configuration for testing."""
    return ArbitrageConfig(threshold=0.99, min_liquidity=100.0)


@pytest.fixture
def statarb_config():
    """Stat-arb configuration for testing."""
    return StatArbConfig(entry_threshold=0.10, exit_threshold=0.02)


def create_market(
    market_id: str,
    yes_price: float,
    no_price: float,
    liquidity: float = 1000.0,
    active: bool = True,
    closed: bool = False,
) -> GammaMarket:
    """Helper to create a market for testing."""
    return GammaMarket(
        id=market_id,
        question=f"Test market {market_id}",
        slug=f"test-{market_id}",
        active=active,
        closed=closed,
        outcome_prices=f"[{yes_price}, {no_price}]",
        liquidity=liquidity,
        volume=10000.0,
        volume24hr=1000.0,
    )


class TestArbitrageScanner:
    """Tests for ArbitrageScanner."""

    def test_scanner_init(self, arb_config):
        """Test scanner initialization."""
        scanner = ArbitrageScanner(config=arb_config)
        assert scanner.threshold == 0.99
        assert scanner.min_liquidity == 100.0

    def test_scan_market_with_arbitrage(self, arb_config):
        """Test detection of arbitrage opportunity."""
        scanner = ArbitrageScanner(config=arb_config)

        # Market with YES=0.45, NO=0.45, sum=0.90 < 0.99
        market = create_market("arb_market", yes_price=0.45, no_price=0.45)

        signal = scanner.scan_market(market)

        assert signal is not None
        assert isinstance(signal, ArbitrageSignal)
        assert signal.market_id == "arb_market"
        assert signal.yes_price == 0.45
        assert signal.no_price == 0.45
        assert signal.combined_price == 0.90
        assert signal.profit_potential == pytest.approx(0.10, abs=0.001)

    def test_scan_market_no_arbitrage(self, arb_config):
        """Test no signal when no arbitrage exists."""
        scanner = ArbitrageScanner(config=arb_config)

        # Market with YES=0.50, NO=0.50, sum=1.00 >= 0.99
        market = create_market("fair_market", yes_price=0.50, no_price=0.50)

        signal = scanner.scan_market(market)

        assert signal is None

    def test_scan_market_inactive(self, arb_config):
        """Test that inactive markets are skipped."""
        scanner = ArbitrageScanner(config=arb_config)

        market = create_market(
            "inactive_market", yes_price=0.40, no_price=0.40, active=False
        )

        signal = scanner.scan_market(market)

        assert signal is None

    def test_scan_market_closed(self, arb_config):
        """Test that closed markets are skipped."""
        scanner = ArbitrageScanner(config=arb_config)

        market = create_market(
            "closed_market", yes_price=0.40, no_price=0.40, closed=True
        )

        signal = scanner.scan_market(market)

        assert signal is None

    def test_scan_market_low_liquidity(self, arb_config):
        """Test that low liquidity markets are skipped."""
        scanner = ArbitrageScanner(config=arb_config)

        market = create_market(
            "low_liq_market", yes_price=0.40, no_price=0.40, liquidity=50.0
        )

        signal = scanner.scan_market(market)

        assert signal is None

    def test_scan_markets_multiple(self, arb_config):
        """Test scanning multiple markets."""
        scanner = ArbitrageScanner(config=arb_config)

        markets = [
            create_market("m1", yes_price=0.45, no_price=0.45),  # Arb
            create_market("m2", yes_price=0.50, no_price=0.50),  # No arb
            create_market("m3", yes_price=0.40, no_price=0.40),  # Arb (better)
            create_market("m4", yes_price=0.60, no_price=0.60),  # No arb
        ]

        signals = scanner.scan_markets(markets)

        assert len(signals) == 2
        # Should be sorted by profit potential (highest first)
        assert signals[0].market_id == "m3"  # 0.20 profit
        assert signals[1].market_id == "m1"  # 0.10 profit

    def test_scan_markets_top_n(self, arb_config):
        """Test limiting results with top_n."""
        scanner = ArbitrageScanner(config=arb_config)

        markets = [
            create_market("m1", yes_price=0.45, no_price=0.45),
            create_market("m2", yes_price=0.40, no_price=0.40),
            create_market("m3", yes_price=0.35, no_price=0.35),
        ]

        signals = scanner.scan_markets(markets, top_n=2)

        assert len(signals) == 2

    def test_find_arbitrage_opportunities_function(self):
        """Test convenience function."""
        markets = [
            create_market("m1", yes_price=0.45, no_price=0.45),
            create_market("m2", yes_price=0.50, no_price=0.50),
        ]

        signals = find_arbitrage_opportunities(
            markets, threshold=0.95, min_liquidity=50.0, top_n=10
        )

        assert len(signals) == 1


class TestStatArbScanner:
    """Tests for StatArbScanner."""

    def test_scanner_init(self, statarb_config):
        """Test scanner initialization."""
        scanner = StatArbScanner(config=statarb_config, pairs=[])
        assert scanner.entry_threshold == 0.10
        assert scanner.exit_threshold == 0.02

    def test_compute_spread_positive_correlation(self, statarb_config):
        """Test spread computation for positively correlated markets."""
        scanner = StatArbScanner(config=statarb_config, pairs=[])

        spread = scanner.compute_spread(0.60, 0.50, correlation=1.0)
        assert spread == pytest.approx(0.10, abs=0.001)

    def test_compute_spread_negative_correlation(self, statarb_config):
        """Test spread computation for negatively correlated markets."""
        scanner = StatArbScanner(config=statarb_config, pairs=[])

        # For inverse correlation, spread = A - (1 - B)
        spread = scanner.compute_spread(0.60, 0.60, correlation=-1.0)
        assert spread == pytest.approx(0.20, abs=0.001)

    def test_scan_pair_with_signal(self, statarb_config):
        """Test detection of stat-arb signal."""
        pair = StatArbPair("market_a", "market_b", name="test_pair")
        scanner = StatArbScanner(config=statarb_config, pairs=[pair])

        markets = {
            "market_a": create_market("market_a", yes_price=0.65, no_price=0.35),
            "market_b": create_market("market_b", yes_price=0.50, no_price=0.50),
        }

        signal = scanner.scan_pair(pair, markets)

        assert signal is not None
        assert isinstance(signal, StatArbSignal)
        assert signal.spread == pytest.approx(0.15, abs=0.001)
        assert signal.direction == "LONG_B_SHORT_A"

    def test_scan_pair_no_signal(self, statarb_config):
        """Test no signal when spread is within threshold."""
        pair = StatArbPair("market_a", "market_b")
        scanner = StatArbScanner(config=statarb_config, pairs=[pair])

        markets = {
            "market_a": create_market("market_a", yes_price=0.52, no_price=0.48),
            "market_b": create_market("market_b", yes_price=0.50, no_price=0.50),
        }

        signal = scanner.scan_pair(pair, markets)

        assert signal is None

    def test_scan_pair_missing_market(self, statarb_config):
        """Test handling of missing market in pair."""
        pair = StatArbPair("market_a", "market_b")
        scanner = StatArbScanner(config=statarb_config, pairs=[pair])

        markets = {
            "market_a": create_market("market_a", yes_price=0.60, no_price=0.40),
            # market_b is missing
        }

        signal = scanner.scan_pair(pair, markets)

        assert signal is None

    def test_scan_pairs_multiple(self, statarb_config):
        """Test scanning multiple pairs."""
        pairs = [
            StatArbPair("m1", "m2", name="pair1"),
            StatArbPair("m3", "m4", name="pair2"),
        ]
        scanner = StatArbScanner(config=statarb_config, pairs=pairs)

        markets = [
            create_market("m1", yes_price=0.70, no_price=0.30),
            create_market("m2", yes_price=0.50, no_price=0.50),
            create_market("m3", yes_price=0.55, no_price=0.45),
            create_market("m4", yes_price=0.50, no_price=0.50),
        ]

        signals = scanner.scan_pairs(markets)

        assert len(signals) == 1  # Only pair1 has spread > 0.10
        assert signals[0].market_a_id == "m1"

    def test_add_pair(self, statarb_config):
        """Test adding a pair dynamically."""
        scanner = StatArbScanner(config=statarb_config, pairs=[])
        assert len(scanner.pairs) == 0

        scanner.add_pair(StatArbPair("a", "b"))
        assert len(scanner.pairs) == 1


class TestArbitrageSignalModel:
    """Tests for ArbitrageSignal model."""

    def test_signal_is_valid(self):
        """Test is_valid property."""
        signal = ArbitrageSignal(
            market_id="test",
            market_question="Test?",
            yes_price=0.45,
            no_price=0.45,
            combined_price=0.90,
            profit_potential=0.10,
            liquidity=1000.0,
        )

        assert signal.is_valid is True

    def test_signal_is_invalid(self):
        """Test is_valid when combined >= 1."""
        signal = ArbitrageSignal(
            market_id="test",
            market_question="Test?",
            yes_price=0.50,
            no_price=0.50,
            combined_price=1.00,
            profit_potential=0.00,
            liquidity=1000.0,
        )

        assert signal.is_valid is False

    def test_signal_frozen(self):
        """Test that signal is immutable."""
        signal = ArbitrageSignal(
            market_id="test",
            market_question="Test?",
            yes_price=0.45,
            no_price=0.45,
            combined_price=0.90,
            profit_potential=0.10,
            liquidity=1000.0,
        )

        with pytest.raises(ValidationError):
            signal.market_id = "new_id"
