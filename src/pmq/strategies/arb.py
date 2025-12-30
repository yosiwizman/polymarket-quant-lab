"""Deterministic arbitrage signal detection.

Arbitrage in prediction markets occurs when the combined price of YES and NO
tokens is less than 1.0, meaning you can buy both and guarantee a profit
when the market resolves.

Signal: YES_price + NO_price < threshold (default 0.99)
Profit potential: 1.0 - (YES_price + NO_price)
"""

from datetime import UTC, datetime
from typing import Any

from pmq.config import ArbitrageConfig, get_settings
from pmq.logging import get_logger, log_trade_event
from pmq.models import ArbitrageSignal, GammaMarket

logger = get_logger("strategies.arb")


class ArbitrageScanner:
    """Scanner for deterministic arbitrage opportunities.

    Detects markets where YES + NO prices sum to less than 1.0,
    indicating a risk-free profit opportunity.
    """

    def __init__(self, config: ArbitrageConfig | None = None) -> None:
        """Initialize the arbitrage scanner.

        Args:
            config: Arbitrage configuration (uses global settings if not provided)
        """
        self._config = config or get_settings().arbitrage
        logger.debug(
            f"ArbitrageScanner initialized: threshold={self._config.threshold}, "
            f"min_liquidity={self._config.min_liquidity}"
        )

    @property
    def threshold(self) -> float:
        """Get the arbitrage threshold."""
        return self._config.threshold

    @property
    def min_liquidity(self) -> float:
        """Get minimum liquidity requirement."""
        return self._config.min_liquidity

    def scan_market(self, market: GammaMarket) -> ArbitrageSignal | None:
        """Scan a single market for arbitrage opportunity.

        Args:
            market: Market to analyze

        Returns:
            ArbitrageSignal if opportunity exists, None otherwise
        """
        # Skip inactive or closed markets
        if not market.active or market.closed:
            return None

        # Get prices
        yes_price = market.yes_price
        no_price = market.no_price

        # Skip if prices are invalid
        if yes_price <= 0 or no_price <= 0:
            return None
        if yes_price >= 1 or no_price >= 1:
            return None

        combined_price = yes_price + no_price

        # Check for arbitrage
        if combined_price < self._config.threshold:
            # Check liquidity requirement
            if market.liquidity < self._config.min_liquidity:
                logger.debug(
                    f"Arb found but low liquidity: {market.id} "
                    f"(liq={market.liquidity:.2f} < {self._config.min_liquidity})"
                )
                return None

            profit_potential = 1.0 - combined_price

            signal = ArbitrageSignal(
                market_id=market.id,
                market_question=market.question,
                yes_price=yes_price,
                no_price=no_price,
                combined_price=combined_price,
                profit_potential=profit_potential,
                liquidity=market.liquidity,
                detected_at=datetime.now(UTC),
            )

            log_trade_event(
                "ARBITRAGE_SIGNAL",
                market.id,
                yes_price=yes_price,
                no_price=no_price,
                combined=combined_price,
                profit=profit_potential,
            )

            return signal

        return None

    def scan_markets(
        self,
        markets: list[GammaMarket],
        top_n: int | None = None,
    ) -> list[ArbitrageSignal]:
        """Scan multiple markets for arbitrage opportunities.

        Args:
            markets: List of markets to analyze
            top_n: Return only top N signals by profit potential

        Returns:
            List of arbitrage signals, sorted by profit potential (descending)
        """
        signals: list[ArbitrageSignal] = []

        for market in markets:
            signal = self.scan_market(market)
            if signal is not None:
                signals.append(signal)

        # Sort by profit potential (highest first)
        signals.sort(key=lambda s: s.profit_potential, reverse=True)

        if top_n is not None:
            signals = signals[:top_n]

        logger.info(f"Found {len(signals)} arbitrage signals from {len(markets)} markets")
        return signals

    def scan_from_db(
        self,
        markets_data: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[ArbitrageSignal]:
        """Scan markets from database format.

        Args:
            markets_data: List of market dicts from DAO
            top_n: Return only top N signals

        Returns:
            List of arbitrage signals
        """
        signals: list[ArbitrageSignal] = []

        for data in markets_data:
            # Skip inactive/closed
            if not data.get("active") or data.get("closed"):
                continue

            yes_price = data.get("last_price_yes", 0.0)
            no_price = data.get("last_price_no", 0.0)

            if yes_price <= 0 or no_price <= 0:
                continue
            if yes_price >= 1 or no_price >= 1:
                continue

            combined_price = yes_price + no_price

            if combined_price < self._config.threshold:
                liquidity = data.get("liquidity", 0.0)
                if liquidity < self._config.min_liquidity:
                    continue

                signal = ArbitrageSignal(
                    market_id=data["id"],
                    market_question=data.get("question", ""),
                    yes_price=yes_price,
                    no_price=no_price,
                    combined_price=combined_price,
                    profit_potential=1.0 - combined_price,
                    liquidity=liquidity,
                    detected_at=datetime.now(UTC),
                )
                signals.append(signal)

        signals.sort(key=lambda s: s.profit_potential, reverse=True)

        if top_n is not None:
            signals = signals[:top_n]

        return signals


def find_arbitrage_opportunities(
    markets: list[GammaMarket],
    threshold: float = 0.99,
    min_liquidity: float = 100.0,
    top_n: int | None = None,
) -> list[ArbitrageSignal]:
    """Convenience function to find arbitrage opportunities.

    Args:
        markets: List of markets to scan
        threshold: Combined price threshold
        min_liquidity: Minimum liquidity in USD
        top_n: Return only top N signals

    Returns:
        List of arbitrage signals
    """
    config = ArbitrageConfig(threshold=threshold, min_liquidity=min_liquidity)
    scanner = ArbitrageScanner(config=config)
    return scanner.scan_markets(markets, top_n=top_n)
