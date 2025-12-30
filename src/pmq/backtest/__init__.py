"""Deterministic backtesting framework for strategy evaluation.

This module provides tools for running historical backtests using
captured market snapshots. Backtests are deterministic and repeatable.

Key components:
- BacktestEngine: Core engine for replaying historical data
- BacktestRunner: Orchestrates strategy execution
- MetricsCalculator: Computes performance metrics
"""

from pmq.backtest.engine import BacktestEngine, BacktestPosition
from pmq.backtest.metrics import MetricsCalculator
from pmq.backtest.runner import BacktestRunner

__all__ = [
    "BacktestEngine",
    "BacktestPosition",
    "BacktestRunner",
    "MetricsCalculator",
]
