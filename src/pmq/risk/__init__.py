"""Risk management module for approval TTL and execution controls.

Phase 9: Provides scoped approval with time-to-live for safe paper trading.

This module provides:
- Scoped approvals with explicit TTL (time-to-live)
- File-based storage for approvals (exports/approvals/)
- Safe-by-default behavior: NOT approved unless explicit valid approval exists
"""

from pmq.risk.approval import (
    Approval,
    ApprovalStore,
    approve,
    get_approval,
    is_approved,
    list_approvals,
    revoke,
)

__all__ = [
    "Approval",
    "ApprovalStore",
    "approve",
    "get_approval",
    "revoke",
    "is_approved",
    "list_approvals",
]
