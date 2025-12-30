"""API routes for the operator console."""

import contextlib
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from pmq import __version__
from pmq.models import SignalType
from pmq.storage import DAO

router = APIRouter()


def get_dao() -> DAO:
    """Get DAO instance."""
    return DAO()


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/api/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status with timestamp and version
    """
    return {
        "ok": True,
        "time": datetime.now(UTC).isoformat(),
        "version": __version__,
    }


@router.get("/api/signals")
def get_signals(
    limit: int = Query(default=50, ge=1, le=500),
    signal_type: str | None = Query(default=None, alias="type"),
) -> dict[str, Any]:
    """Get recent signals.

    Args:
        limit: Maximum number of signals to return
        signal_type: Filter by type (ARBITRAGE or STAT_ARB)

    Returns:
        List of signals
    """
    dao = get_dao()
    type_filter = None
    if signal_type:
        with contextlib.suppress(ValueError):
            type_filter = SignalType(signal_type.upper())

    signals = dao.get_recent_signals(signal_type=type_filter, limit=limit)
    return {"signals": signals, "count": len(signals)}


@router.get("/api/trades")
def get_trades(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Get recent paper trades.

    Args:
        limit: Maximum number of trades to return

    Returns:
        List of trades
    """
    dao = get_dao()
    trades = dao.get_trades_for_export(limit=limit)
    return {"trades": trades, "count": len(trades)}


@router.get("/api/positions")
def get_positions() -> dict[str, Any]:
    """Get current paper positions.

    Returns:
        List of open positions
    """
    dao = get_dao()
    positions = dao.get_positions_for_export()
    return {"positions": positions, "count": len(positions)}


@router.get("/api/summary")
def get_summary() -> dict[str, Any]:
    """Get trading summary statistics.

    Returns:
        Summary with stats, runtime state, and counts
    """
    dao = get_dao()

    stats = dao.get_trading_stats()
    runtime_state = dao.get_all_runtime_state()

    return {
        "stats": stats,
        "runtime": runtime_state,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# =============================================================================
# Snapshot Endpoints (Phase 2.5)
# =============================================================================


@router.get("/api/snapshots/summary")
def get_snapshots_summary() -> dict[str, Any]:
    """Get overall snapshot statistics.

    Returns:
        Summary with snapshot counts and time range
    """
    dao = get_dao()
    summary = dao.get_snapshot_summary()
    return {
        "summary": summary,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/api/snapshots/coverage")
def get_snapshots_coverage(
    from_time: str = Query(alias="from"),
    to_time: str = Query(alias="to"),
) -> dict[str, Any]:
    """Get snapshot coverage for a time window.

    Args:
        from_time: Start time (ISO or YYYY-MM-DD)
        to_time: End time (ISO or YYYY-MM-DD)

    Returns:
        Coverage stats for the window
    """
    dao = get_dao()
    # Normalize dates
    if "T" not in from_time:
        from_time = f"{from_time}T00:00:00+00:00"
    if "T" not in to_time:
        to_time = f"{to_time}T23:59:59+00:00"

    coverage = dao.get_snapshot_coverage(from_time, to_time)
    return {"coverage": coverage}


@router.get("/api/snapshots/quality/latest")
def get_latest_quality_report(
    mode: str | None = Query(
        default=None, description="Window mode: explicit, last_minutes, last_times"
    ),
    value: int | None = Query(
        default=None, description="Minutes or times count (for rolling modes)"
    ),
    interval: int = Query(default=60, description="Expected interval in seconds"),
) -> dict[str, Any]:
    """Get quality report - stored or on-demand.

    Modes:
    - None (default): Return latest stored report
    - last_minutes: Check quality for last N minutes (requires value)
    - last_times: Check quality for last K distinct snapshot times (requires value)

    Returns:
        Quality report with status, maturity, and readiness
    """
    from pmq.quality import QualityChecker

    dao = get_dao()

    # Handle rolling window modes
    if mode == "last_minutes" and value:
        checker = QualityChecker(dao=dao)
        result = checker.check_last_minutes(value, interval)
        report = {
            "window_from": result.window_from,
            "window_to": result.window_to,
            "window_mode": result.window_mode,
            "coverage_pct": result.coverage_pct,
            "missing_intervals": result.missing_intervals,
            "duplicate_count": result.duplicate_count,
            "status": result.status,
            "maturity_score": result.maturity_score,
            "ready_for_scorecard": result.ready_for_scorecard,
            "distinct_times": result.distinct_times,
            "expected_times": result.expected_times,
            "largest_gap_seconds": result.largest_gap_seconds,
            "markets_seen": result.markets_seen,
            "snapshots_written": result.snapshots_written,
        }
    elif mode == "last_times" and value:
        checker = QualityChecker(dao=dao)
        result = checker.check_last_times(value, interval)
        report = {
            "window_from": result.window_from,
            "window_to": result.window_to,
            "window_mode": result.window_mode,
            "coverage_pct": result.coverage_pct,
            "missing_intervals": result.missing_intervals,
            "duplicate_count": result.duplicate_count,
            "status": result.status,
            "maturity_score": result.maturity_score,
            "ready_for_scorecard": result.ready_for_scorecard,
            "distinct_times": result.distinct_times,
            "expected_times": result.expected_times,
            "largest_gap_seconds": result.largest_gap_seconds,
            "markets_seen": result.markets_seen,
            "snapshots_written": result.snapshots_written,
        }
    else:
        # Default: return latest stored report
        report = dao.get_latest_quality_report()

    # Determine status badge
    if report:
        ready = report.get("ready_for_scorecard", False)
        coverage_pct = report.get("coverage_pct", 0)
        missing = report.get("missing_intervals", 0)
        duplicates = report.get("duplicate_count", 0)

        if ready and coverage_pct >= 95 and missing <= 5 and duplicates == 0:
            status = "healthy"
        elif coverage_pct >= 80 and missing <= 20:
            status = "degraded"
        else:
            status = "unhealthy"

        readiness = "ready" if ready else "not_ready"
    else:
        status = "unknown"
        readiness = "unknown"

    return {
        "report": report,
        "status": status,
        "readiness": readiness,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# =============================================================================
# Governance Endpoints (Phase 3)
# =============================================================================


@router.get("/api/governance/approvals")
def get_approvals(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Get strategy approvals.

    Args:
        status: Optional filter (APPROVED, REVOKED, PENDING)
        limit: Maximum results

    Returns:
        List of approval records
    """
    dao = get_dao()
    approvals = dao.get_approvals(status=status, limit=limit)
    return {
        "approvals": approvals,
        "count": len(approvals),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/api/governance/risk-events")
def get_risk_events(
    severity: str | None = Query(default=None),
    event_type: str | None = Query(default=None, alias="type"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Get risk events.

    Args:
        severity: Optional filter (INFO, WARN, CRITICAL)
        event_type: Optional filter by event type
        limit: Maximum results

    Returns:
        List of risk events
    """
    dao = get_dao()
    events = dao.get_risk_events(severity=severity, event_type=event_type, limit=limit)
    return {
        "events": events,
        "count": len(events),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/api/governance/summary")
def get_governance_summary() -> dict[str, Any]:
    """Get governance summary.

    Returns:
        Summary with approval and risk event counts
    """
    dao = get_dao()
    approvals = dao.get_approvals(status="APPROVED", limit=100)
    recent_events = dao.get_risk_events(limit=50)

    # Count by severity
    severity_counts = {"INFO": 0, "WARN": 0, "CRITICAL": 0}
    for event in recent_events:
        sev = event.get("severity", "INFO")
        if sev in severity_counts:
            severity_counts[sev] += 1

    return {
        "approved_strategies": len(approvals),
        "recent_risk_events": len(recent_events),
        "severity_counts": severity_counts,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# =============================================================================
# Dashboard (HTML)
# =============================================================================


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    """Render the operator dashboard.

    Args:
        request: FastAPI request object

    Returns:
        Rendered HTML dashboard
    """
    dao = get_dao()

    # Get data for dashboard
    stats = dao.get_trading_stats()
    runtime_state = dao.get_all_runtime_state()
    recent_signals = dao.get_recent_signals(limit=10)
    recent_trades = dao.get_trades_for_export(limit=10)
    positions = dao.get_positions_for_export()

    # Snapshot data (Phase 2.5)
    snapshot_summary = dao.get_snapshot_summary()
    quality_report = dao.get_latest_quality_report()

    # Compute quality status + maturity
    if quality_report:
        coverage_pct = quality_report.get("coverage_pct", 0)
        missing = quality_report.get("missing_intervals", 0)
        duplicates = quality_report.get("duplicate_count", 0)
        maturity_score = quality_report.get("maturity_score", 0)
        ready_for_scorecard = quality_report.get("ready_for_scorecard", False)

        if ready_for_scorecard and coverage_pct >= 95 and missing <= 5 and duplicates == 0:
            quality_status = "healthy"
        elif coverage_pct >= 80 and missing <= 20:
            quality_status = "degraded"
        else:
            quality_status = "unhealthy"
    else:
        quality_status = "unknown"
        maturity_score = 0
        ready_for_scorecard = False

    # Governance data (Phase 3)
    approvals = dao.get_approvals(status="APPROVED", limit=10)
    risk_events = dao.get_risk_events(limit=10)

    templates = request.app.state.templates
    response: HTMLResponse = templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "version": __version__,
            "stats": stats,
            "runtime": runtime_state,
            "signals": recent_signals,
            "trades": recent_trades,
            "positions": positions,
            "snapshot_summary": snapshot_summary,
            "quality_report": quality_report,
            "quality_status": quality_status,
            "maturity_score": maturity_score,
            "ready_for_scorecard": ready_for_scorecard,
            "approvals": approvals,
            "risk_events": risk_events,
            "now": datetime.now(UTC).isoformat(),
        },
    )
    return response
