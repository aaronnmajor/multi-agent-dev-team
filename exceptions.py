"""
Structured exception hierarchy for the pipeline.

Classify errors at the point they are raised so downstream handlers (retry,
circuit breaker, graceful degradation) can decide the correct response.
"""


class AgentError(Exception):
    """Base class for all pipeline-specific errors."""


class TransientError(AgentError):
    """Temporary failure that may succeed on retry (rate limit, timeout, network blip)."""


class PermanentError(AgentError):
    """Permanent failure that will not succeed on retry (auth, bad schema, config)."""


class DegradableError(AgentError):
    """Component unavailable; caller should apply a fallback and continue."""
