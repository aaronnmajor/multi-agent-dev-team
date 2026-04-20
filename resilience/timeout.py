"""
Pipeline-level timeout using a thread-based watchdog.

Wraps a callable in a thread and terminates with a PipelineTimeoutError if
execution exceeds the configured ceiling. Pure-Python implementation (no
asyncio dependency required) since the rest of the pipeline is synchronous.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

DEFAULT_PIPELINE_TIMEOUT_S = 300  # 5 minutes per run.


class PipelineTimeoutError(Exception):
    """Raised when the top-level pipeline run exceeds its wall-clock budget."""


def with_timeout(
    func: Callable[..., T],
    timeout_s: float = DEFAULT_PIPELINE_TIMEOUT_S,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run `func(*args, **kwargs)` with a wall-clock timeout.

    The watchdog is best-effort — Python cannot safely kill the underlying
    thread, so a timeout causes the caller to see a PipelineTimeoutError while
    the worker thread may continue briefly in the background. For the pipeline
    this is acceptable: the worker will exit as soon as its next LLM/tool call
    completes (each of those has its own internal timeout).
    """
    result: list[T] = []
    error: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(func(*args, **kwargs))
        except BaseException as e:
            error.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        raise PipelineTimeoutError(
            f"Pipeline exceeded timeout of {timeout_s}s and was abandoned."
        )
    if error:
        raise error[0]
    if not result:
        raise PipelineTimeoutError("Pipeline produced no result (unknown reason).")
    return result[0]
