"""
Lightweight tracing primitives.

`new_run_id` produces a UUID at the pipeline's root; it's propagated through
ProjectState so downstream nodes can attach their events to the same trace.
`trace_span` is a context manager that logs span_start / span_end events with
duration — the minimum viable distributed trace without a full LangSmith
or OpenTelemetry integration.

`configure_langsmith()` is opt-in: when ``LANGCHAIN_API_KEY`` (or its
``LANGSMITH_API_KEY`` alias) is present in the environment, every node
decorated with ``@traceable`` (from langsmith) is forwarded as a span to
the LangSmith dashboard. When the env var is not set, the decorator is a
no-op shim and the local JSON logs remain the only trace surface.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from observability.logging import get_logger

try:
    from langsmith import traceable as _ls_traceable  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — optional dep
    _ls_traceable = None


def new_run_id() -> str:
    """Generate a fresh run identifier."""
    return str(uuid.uuid4())


def configure_langsmith() -> bool:
    """Mirror LANGSMITH_* env vars onto the LANGCHAIN_* names the SDK reads.

    Returns True when LangSmith looks configured (an API key is present),
    False otherwise. Safe to call unconditionally at pipeline startup.
    """
    if os.environ.get("LANGSMITH_ENDPOINT") and not os.environ.get("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]
    if os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    if os.environ.get("LANGSMITH_PROJECT") and not os.environ.get("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "multi-agent-dev-team")
        return True
    return False


def traceable(name: str | None = None, run_type: str = "chain") -> Callable[[Callable], Callable]:
    """Wrap a function so it appears as a span in the LangSmith dashboard.

    Falls back to a no-op decorator when the optional ``langsmith`` package
    is not installed, so unconfigured environments don't fail at import
    time. The local ``trace_span`` JSON logs are emitted regardless.
    """
    if _ls_traceable is None:
        def _noop(fn: Callable) -> Callable:
            return fn
        return _noop
    return _ls_traceable(name=name, run_type=run_type)


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
