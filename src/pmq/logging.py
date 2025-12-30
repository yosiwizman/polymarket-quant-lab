"""Structured logging configuration for audit trails."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Create module-level logger
logger = logging.getLogger("pmq")


class StructuredFormatter(logging.Formatter):
    """JSON-like structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured output."""
        timestamp = datetime.now(timezone.utc).isoformat()
        level = record.levelname
        message = record.getMessage()

        # Build structured log entry
        entry = f"[{timestamp}] [{level}] {record.name}: {message}"

        # Add extra fields if present
        extra_fields: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            }:
                extra_fields[key] = value

        if extra_fields:
            entry += f" | {extra_fields}"

        return entry


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure application logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger for pmq
    logger.setLevel(log_level)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    logger.debug("Logging configured", extra={"level": level, "log_file": str(log_file)})


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for the specified module.

    Args:
        name: Module name (will be prefixed with 'pmq.')

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"pmq.{name}")


# Audit logger for trading operations
audit_logger = get_logger("audit")


def log_trade_event(
    event_type: str,
    market_id: str,
    **details: Any,
) -> None:
    """Log a trade-related event for audit purposes.

    Args:
        event_type: Type of event (SIGNAL, PAPER_TRADE, POSITION_UPDATE, etc.)
        market_id: Market identifier
        **details: Additional event details
    """
    audit_logger.info(
        f"{event_type} | market={market_id}",
        extra={"event_type": event_type, "market_id": market_id, **details},
    )
