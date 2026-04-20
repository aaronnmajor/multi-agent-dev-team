"""
Retry decorator with exponential backoff and full jitter.

Wrap the LLM call (or any other transient-error-prone operation) with this
decorator; do NOT wrap entire agent nodes — a retry that re-runs the whole
node throws away reasoning tokens already spent.

Classifies via the `TransientError` exception from `exceptions.py`.
Permanent and Degradable errors propagate immediately (no retry).
"""

from __future__ import annotations

import random
import time
from functools import wraps
from typing import Callable

from exceptions import TransientError
from observability.logging import get_logger

_log = get_logger("resilience")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
):
    """Decorator factory: retry on TransientError with exponential backoff + full jitter.

    Args:
        max_attempts: total attempts including the first (so max_attempts=3 means 2 retries).
        base_delay: initial backoff delay in seconds.
        max_delay: cap on the exponential delay.
        sleep: injectable sleep function (swap for a fake in tests).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    last_error = e
                    if attempt == max_attempts - 1:
                        _log.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        raise
                    exp_delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, exp_delay)
                    _log.warning(
                        "retry_scheduled",
                        func=func.__name__,
                        attempt=attempt + 1,
                        delay_s=round(jitter, 2),
                        error=str(e),
                    )
                    sleep(jitter)
            # Should be unreachable — loop either returns or raises.
            raise last_error if last_error else RuntimeError("retry loop fell through")
        return wrapper
    return decorator
