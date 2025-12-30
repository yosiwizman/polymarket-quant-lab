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
def get_latest_quality_report() -> dict[str, Any]:
    """Get the most recent quality report.

    Returns:
        Latest quality report or empty if none exists
    """
    dao = get_dao()
    report = dao.get_latest_quality_report()

    # Determine status badge
    if report:
        coverage_pct = report.get("coverage_pct", 0)
        missing = report.get("missing_intervals", 0)
        duplicates = report.get("duplicate_count", 0)

        if coverage_pct >= 95 and missing <= 5 and duplicates == 0:
            status = "healthy"
        elif coverage_pct >= 80 and missing <= 20:
            status = "degraded"
        else:
            status = "unhealthy"
    else:
        status = "unknown"

    return {
        "report": report,
        "status": status,
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

    # Compute quality status
    if quality_report:
        coverage_pct = quality_report.get("coverage_pct", 0)
        missing = quality_report.get("missing_intervals", 0)
        duplicates = quality_report.get("duplicate_count", 0)
        if coverage_pct >= 95 and missing <= 5 and duplicates == 0:
            quality_status = "healthy"
        elif coverage_pct >= 80 and missing <= 20:
            quality_status = "degraded"
        else:
            quality_status = "unhealthy"
    else:
        quality_status = "unknown"

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
            "now": datetime.now(UTC).isoformat(),
        },
    )
    return response
