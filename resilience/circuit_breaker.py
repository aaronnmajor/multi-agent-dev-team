"""
Circuit breaker for systemic downstream failures.

Three states: CLOSED -> OPEN -> HALF_OPEN -> (CLOSED on success, OPEN on failure).

Each agent should own its own breaker instance so a QA outage does not trip
the PM breaker. Thresholds default to 3 consecutive TransientErrors within a
30-second cool-down, following the Week 4 course notes' recommendation.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable

from exceptions import DegradableError, TransientError
from observability.logging import get_logger

_log = get_logger("resilience")


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        cool_down: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.cool_down = cool_down
        self._clock = clock
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._opened_at: float = 0.0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke `func` through the breaker.

        Raises DegradableError when the breaker is OPEN (fast fail).
        TransientErrors from `func` increment the failure count and may trip
        the breaker; other exceptions propagate unchanged.
        """
        if self.state == CircuitState.OPEN:
            if self._clock() - self._opened_at >= self.cool_down:
                self.state = CircuitState.HALF_OPEN
                _log.info("circuit_breaker_half_open", breaker=self.name)
            else:
                raise DegradableError(f"circuit_open:{self.name}")

        try:
            result = func(*args, **kwargs)
        except TransientError:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_success(self) -> None:
        if self.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            _log.info("circuit_breaker_closed", breaker=self.name)
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def _record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self._opened_at = self._clock()
            _log.warning(
                "circuit_breaker_opened",
                breaker=self.name,
                failure_count=self.failure_count,
            )
