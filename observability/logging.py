"""
Structured logging for the pipeline.

Emits JSON log events (via stdlib `logging` with a JSON formatter) so events
can be queried, aggregated, and dashboarded. Every event carries at minimum
`event`, `agent`, and `run_id` so a single pipeline invocation can be traced
from requirement to final artifact.

Configure once at startup; subsequent `get_logger("agent_name")` calls return
ready-to-use loggers with the agent name bound into every event.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_CONFIGURED = False
_LEVEL = logging.INFO


class JSONFormatter(logging.Formatter):
    """Renders log records as single-line JSON objects."""

    STANDARD_ATTRS = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":       datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":    record.levelname,
            "event":    record.getMessage(),
            "logger":   record.name,
        }
        # Pull any extra fields the caller passed via the `extra=` kwarg.
        for key, value in record.__dict__.items():
            if key not in self.STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON formatter on the root logger. Safe to call repeatedly."""
    global _CONFIGURED, _LEVEL
    _LEVEL = level
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any pre-existing handlers so we don't double-log.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    _CONFIGURED = True


class _AgentLogger:
    """Thin wrapper that attaches `agent` and `run_id` to every event."""

    def __init__(self, name: str, bound: dict[str, Any] | None = None) -> None:
        self._log = logging.getLogger(name)
        self._bound = dict(bound or {})
        self._bound.setdefault("agent", name)

    def bind(self, **kwargs: Any) -> "_AgentLogger":
        merged = {**self._bound, **kwargs}
        return _AgentLogger(self._log.name, merged)

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        extra = {**self._bound, **fields}
        self._log.log(level, event, extra=extra)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def critical(self, event: str, **fields: Any) -> None:
        self._emit(logging.CRITICAL, event, **fields)


def get_logger(agent_name: str) -> _AgentLogger:
    """Return a logger bound to this agent. Auto-configures on first use."""
    if not _CONFIGURED:
        configure_logging(_LEVEL)
    return _AgentLogger(agent_name)


def bind_run_context(logger: _AgentLogger, run_id: str, **extra: Any) -> _AgentLogger:
    """Bind a run_id (and any extra context) into the logger's default fields."""
    return logger.bind(run_id=run_id, **extra)
