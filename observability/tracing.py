"""
Lightweight tracing primitives.

`new_run_id` produces a UUID at the pipeline's root; it's propagated through
ProjectState so downstream nodes can attach their events to the same trace.
`trace_span` is a context manager that logs span_start / span_end events with
duration — the minimum viable distributed trace without a full LangSmith
or OpenTelemetry integration.

Swap these in for `@traceable` from LangSmith when an API key is available —
the interface is intentionally minimal so the call sites don't have to change.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from observability.logging import get_logger


def new_run_id() -> str:
    """Generate a fresh run identifier."""
    return str(uuid.uuid4())


@contextmanager
def trace_span(agent: str, span_name: str, run_id: str, **attrs: Any) -> Iterator[None]:
    """Log `span_start` on enter and `span_end` with duration_ms on exit.

    Use at every significant boundary: agent node, LLM call, tool invocation.
    Exceptions are logged as `span_error` with the exception type and message.
    """
    log = get_logger(agent).bind(run_id=run_id)
    start = time.monotonic()
    log.info("span_start", span=span_name, **attrs)
    try:
        yield
    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        log.error(
            "span_error",
            span=span_name,
            duration_ms=duration_ms,
            exc_type=type(e).__name__,
            exc_msg=str(e),
        )
        raise
    else:
        duration_ms = int((time.monotonic() - start) * 1000)
        log.info("span_end", span=span_name, duration_ms=duration_ms)
