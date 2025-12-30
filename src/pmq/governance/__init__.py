"""Governance module for strategy approval and risk management.

This module provides:
- Strategy scorecards computed from backtest results
- Approval registry for formal strategy authorization
- Risk gate enforcement for paper/live trading
"""

from pmq.governance.risk_gate import RiskGate, RiskLimits, limits_to_dict
from pmq.governance.scorecard import StrategyScorecard, compute_scorecard

__all__ = [
    "StrategyScorecard",
    "compute_scorecard",
    "RiskGate",
    "RiskLimits",
    "limits_to_dict",
]
