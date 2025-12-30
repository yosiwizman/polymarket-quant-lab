"""Statistical arbitrage signal detection for correlated market pairs.

Stat-arb identifies pairs of correlated markets and trades the spread
when it diverges beyond a threshold, expecting mean reversion.

Entry: abs(spread) > entry_threshold
Exit: abs(spread) < exit_threshold
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pmq.config import StatArbConfig, get_settings, load_pairs_config
from pmq.logging import get_logger, log_trade_event
from pmq.models import GammaMarket, StatArbSignal

logger = get_logger("strategies.statarb")


class StatArbPair:
    """Configuration for a correlated market pair."""

    def __init__(
        self,
        market_a_id: str,
        market_b_id: str,
        name: str = "",
        correlation: float = 1.0,
    ) -> None:
        """Initialize a stat-arb pair.

        Args:
            market_a_id: ID of first market
            market_b_id: ID of second market
            name: Human-readable name for the pair
            correlation: Expected correlation (1.0 = same direction, -1.0 = inverse)
        """
        self.market_a_id = market_a_id
        self.market_b_id = market_b_id
        self.name = name or f"{market_a_id[:8]}../{market_b_id[:8]}.."
        self.correlation = correlation

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StatArbPair":
        """Create pair from dictionary config."""
        return cls(
            market_a_id=data["market_a_id"],
            market_b_id=data["market_b_id"],
            name=data.get("name", ""),
            correlation=data.get("correlation", 1.0),
        )


class StatArbScanner:
    """Scanner for statistical arbitrage opportunities between market pairs.

    Monitors configured pairs and signals when spreads diverge beyond thresholds.
    """

    def __init__(
        self,
        config: StatArbConfig | None = None,
        pairs: list[StatArbPair] | None = None,
    ) -> None:
        """Initialize the stat-arb scanner.

        Args:
            config: Stat-arb configuration
            pairs: List of pairs to monitor (loads from config file if not provided)
        """
        self._config = config or get_settings().statarb
        self._pairs = pairs or self._load_pairs()
        logger.debug(
            f"StatArbScanner initialized: entry={self._config.entry_threshold}, "
            f"exit={self._config.exit_threshold}, pairs={len(self._pairs)}"
        )

    def _load_pairs(self) -> list[StatArbPair]:
        """Load pairs from configuration file."""
        pairs_data = load_pairs_config(self._config.pairs_file)
        pairs = []
        for data in pairs_data:
            try:
                pair = StatArbPair.from_dict(data)
                pairs.append(pair)
            except (KeyError, TypeError) as e:
                logger.warning(f"Invalid pair config: {e}")
        return pairs

    @property
    def entry_threshold(self) -> float:
        """Get entry threshold."""
        return self._config.entry_threshold

    @property
    def exit_threshold(self) -> float:
        """Get exit threshold."""
        return self._config.exit_threshold

    @property
    def pairs(self) -> list[StatArbPair]:
        """Get configured pairs."""
        return self._pairs

    def add_pair(self, pair: StatArbPair) -> None:
        """Add a pair to monitor.

        Args:
            pair: Pair configuration
        """
        self._pairs.append(pair)

    def compute_spread(
        self,
        price_a: float,
        price_b: float,
        correlation: float = 1.0,
    ) -> float:
        """Compute spread between two prices.

        Args:
            price_a: Price of market A
            price_b: Price of market B
            correlation: Expected correlation

        Returns:
            Spread value (adjusted for correlation)
        """
        if correlation < 0:
            # Inverse correlation: expect prices to move opposite
            return price_a - (1.0 - price_b)
        return price_a - price_b

    def scan_pair(
        self,
        pair: StatArbPair,
        markets: dict[str, GammaMarket],
    ) -> StatArbSignal | None:
        """Scan a single pair for stat-arb opportunity.

        Args:
            pair: Pair configuration
            markets: Dict mapping market ID to market data

        Returns:
            StatArbSignal if opportunity exists, None otherwise
        """
        market_a = markets.get(pair.market_a_id)
        market_b = markets.get(pair.market_b_id)

        if not market_a or not market_b:
            return None

        # Skip if either market is inactive
        if not market_a.active or not market_b.active:
            return None
        if market_a.closed or market_b.closed:
            return None

        price_a = market_a.yes_price
        price_b = market_b.yes_price

        if price_a <= 0 or price_b <= 0:
            return None

        spread = self.compute_spread(price_a, price_b, pair.correlation)
        abs_spread = abs(spread)

        if abs_spread > self._config.entry_threshold:
            # Determine direction
            if spread > 0:
                direction = "LONG_B_SHORT_A"  # Buy B, sell A
            else:
                direction = "LONG_A_SHORT_B"  # Buy A, sell B

            signal = StatArbSignal(
                market_a_id=pair.market_a_id,
                market_b_id=pair.market_b_id,
                market_a_question=market_a.question,
                market_b_question=market_b.question,
                price_a=price_a,
                price_b=price_b,
                spread=spread,
                entry_threshold=self._config.entry_threshold,
                exit_threshold=self._config.exit_threshold,
                direction=direction,
                detected_at=datetime.now(timezone.utc),
            )

            log_trade_event(
                "STATARB_SIGNAL",
                pair.market_a_id,
                pair_name=pair.name,
                spread=spread,
                direction=direction,
            )

            return signal

        return None

    def scan_pairs(
        self,
        markets: dict[str, GammaMarket] | list[GammaMarket],
    ) -> list[StatArbSignal]:
        """Scan all configured pairs for stat-arb opportunities.

        Args:
            markets: Dict or list of markets

        Returns:
            List of stat-arb signals
        """
        # Convert list to dict if needed
        if isinstance(markets, list):
            markets = {m.id: m for m in markets}

        signals: list[StatArbSignal] = []

        for pair in self._pairs:
            signal = self.scan_pair(pair, markets)
            if signal is not None:
                signals.append(signal)

        # Sort by absolute spread (highest divergence first)
        signals.sort(key=lambda s: abs(s.spread), reverse=True)

        logger.info(f"Found {len(signals)} stat-arb signals from {len(self._pairs)} pairs")
        return signals

    def scan_from_db(
        self,
        markets_data: list[dict[str, Any]],
    ) -> list[StatArbSignal]:
        """Scan pairs using market data from database.

        Args:
            markets_data: List of market dicts from DAO

        Returns:
            List of stat-arb signals
        """
        # Build lookup dict
        markets_lookup: dict[str, dict[str, Any]] = {m["id"]: m for m in markets_data}

        signals: list[StatArbSignal] = []

        for pair in self._pairs:
            market_a = markets_lookup.get(pair.market_a_id)
            market_b = markets_lookup.get(pair.market_b_id)

            if not market_a or not market_b:
                continue

            if not market_a.get("active") or not market_b.get("active"):
                continue
            if market_a.get("closed") or market_b.get("closed"):
                continue

            price_a = market_a.get("last_price_yes", 0.0)
            price_b = market_b.get("last_price_yes", 0.0)

            if price_a <= 0 or price_b <= 0:
                continue

            spread = self.compute_spread(price_a, price_b, pair.correlation)
            abs_spread = abs(spread)

            if abs_spread > self._config.entry_threshold:
                direction = "LONG_B_SHORT_A" if spread > 0 else "LONG_A_SHORT_B"

                signal = StatArbSignal(
                    market_a_id=pair.market_a_id,
                    market_b_id=pair.market_b_id,
                    market_a_question=market_a.get("question", ""),
                    market_b_question=market_b.get("question", ""),
                    price_a=price_a,
                    price_b=price_b,
                    spread=spread,
                    entry_threshold=self._config.entry_threshold,
                    exit_threshold=self._config.exit_threshold,
                    direction=direction,
                    detected_at=datetime.now(timezone.utc),
                )
                signals.append(signal)

        signals.sort(key=lambda s: abs(s.spread), reverse=True)
        return signals
