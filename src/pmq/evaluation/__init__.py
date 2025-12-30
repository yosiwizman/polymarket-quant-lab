"""Evaluation pipeline module for end-to-end strategy validation.

This module provides automated evaluation of strategies through:
- Data quality checks
- Backtesting
- Approval evaluation
- Optional paper trading smoke tests
"""

from pmq.evaluation.pipeline import EvaluationPipeline, EvaluationResult
from pmq.evaluation.reporter import EvaluationReporter

__all__ = [
    "EvaluationPipeline",
    "EvaluationReporter",
    "EvaluationResult",
]
