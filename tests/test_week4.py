"""
Tests for Week 4: observability, resilience, cost optimisation.
All tests are deterministic and make no LLM calls.
"""

from __future__ import annotations

import json
import time

import pytest

from caching.response_cache import ResponseCache
from config import AGENT_MODELS, get_model
from exceptions import AgentError, DegradableError, PermanentError, TransientError
from observability.cost import MODEL_PRICES, CostTracker, TokenUsage
from observability.logging import JSONFormatter, get_logger
from observability.tracing import new_run_id, trace_span
from resilience.circuit_breaker import CircuitBreaker, CircuitState
from resilience.retry import retry_with_backoff
from resilience.timeout import PipelineTimeoutError, with_timeout


# ─────────────────────────────────────────────────────────────────────────────
# Exception hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestExceptions:
    def test_all_descend_from_agent_error(self):
        assert issubclass(TransientError, AgentError)
        assert issubclass(PermanentError, AgentError)
        assert issubclass(DegradableError, AgentError)


# ─────────────────────────────────────────────────────────────────────────────
# Config — model selection
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_all_three_agents_have_models(self):
        assert set(AGENT_MODELS) == {"pm", "coder", "qa"}

    def test_get_model_returns_configured_value(self):
        assert get_model("pm") == AGENT_MODELS["pm"]

    def test_get_model_raises_on_unknown_agent(self):
        with pytest.raises(KeyError):
            get_model("stranger")


# ─────────────────────────────────────────────────────────────────────────────
# Observability — cost tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenUsage:
    def test_cost_computation_uses_model_price(self):
        u = TokenUsage(agent="pm", model="gpt-4o-mini", prompt=1000, completion=500)
        p_price, c_price = MODEL_PRICES["gpt-4o-mini"]
        expected = (1000 * p_price + 500 * c_price) / 1000.0
        assert u.cost_usd == pytest.approx(expected)

    def test_cached_tokens_discount_billable_prompt(self):
        u_nocache = TokenUsage(agent="pm", model="gpt-4o", prompt=1000, completion=500)
        u_cached  = TokenUsage(agent="pm", model="gpt-4o", prompt=1000, completion=500, cached=400)
        assert u_cached.cost_usd < u_nocache.cost_usd

    def test_unknown_model_uses_default_pricing(self):
        u = TokenUsage(agent="pm", model="model-not-in-registry", prompt=100, completion=100)
        # Should not raise, and should return a finite cost.
        assert u.cost_usd > 0

    def test_total_tokens(self):
        u = TokenUsage(agent="coder", model="gpt-4o", prompt=100, completion=50)
        assert u.total_tokens == 150


class TestCostTracker:
    def test_report_sums_per_agent(self):
        tracker = CostTracker(run_id="r1")
        tracker.record(TokenUsage(agent="pm", model="gpt-4o-mini", prompt=100, completion=50))
        tracker.record(TokenUsage(agent="pm", model="gpt-4o-mini", prompt=200, completion=100))
        tracker.record(TokenUsage(agent="qa", model="gpt-4o-mini", prompt=50, completion=25))
        report = tracker.report()
        assert report["by_agent"]["pm"]["prompt"] == 300
        assert report["by_agent"]["pm"]["completion"] == 150
        assert report["by_agent"]["qa"]["prompt"] == 50
        assert report["total_tokens"] == 525
        assert report["total_cost_usd"] > 0
        assert report["run_id"] == "r1"


# ─────────────────────────────────────────────────────────────────────────────
# Observability — logging format
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONFormatter:
    def test_format_produces_valid_json_with_required_fields(self):
        import logging
        record = logging.LogRecord(
            name="pm", level=logging.INFO, pathname="", lineno=0,
            msg="agent_node_start", args=(), exc_info=None,
        )
        # Simulate passing `extra=` fields.
        record.agent = "pm"
        record.run_id = "abc123"
        out = JSONFormatter().format(record)
        parsed = json.loads(out)
        assert parsed["event"] == "agent_node_start"
        assert parsed["level"] == "INFO"
        assert parsed["agent"] == "pm"
        assert parsed["run_id"] == "abc123"
        assert "ts" in parsed


class TestTracing:
    def test_new_run_id_returns_uuid_string(self):
        rid = new_run_id()
        assert isinstance(rid, str)
        # UUID4 hex length with dashes.
        assert len(rid) == 36

    def test_trace_span_runs_its_body(self):
        calls = []
        with trace_span("pm", "test_span", run_id="r1"):
            calls.append("inside")
        assert calls == ["inside"]

    def test_trace_span_propagates_exceptions(self):
        with pytest.raises(ValueError):
            with trace_span("pm", "test_span", run_id="r1"):
                raise ValueError("boom")


# ─────────────────────────────────────────────────────────────────────────────
# Resilience — retry
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryWithBackoff:
    def test_succeeds_without_retry(self):
        calls = [0]
        @retry_with_backoff(max_attempts=3, sleep=lambda _: None)
        def fn():
            calls[0] += 1
            return "ok"
        assert fn() == "ok"
        assert calls[0] == 1

    def test_retries_transient_error_then_succeeds(self):
        calls = [0]
        @retry_with_backoff(max_attempts=3, sleep=lambda _: None)
        def fn():
            calls[0] += 1
            if calls[0] < 3:
                raise TransientError("transient")
            return "ok"
        assert fn() == "ok"
        assert calls[0] == 3

    def test_raises_after_max_attempts(self):
        calls = [0]
        @retry_with_backoff(max_attempts=3, sleep=lambda _: None)
        def fn():
            calls[0] += 1
            raise TransientError("still failing")
        with pytest.raises(TransientError):
            fn()
        assert calls[0] == 3

    def test_does_not_retry_permanent_error(self):
        calls = [0]
        @retry_with_backoff(max_attempts=3, sleep=lambda _: None)
        def fn():
            calls[0] += 1
            raise PermanentError("bad key")
        with pytest.raises(PermanentError):
            fn()
        assert calls[0] == 1  # no retry


# ─────────────────────────────────────────────────────────────────────────────
# Resilience — circuit breaker
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=2)

        def fail():
            raise TransientError("x")

        for _ in range(2):
            with pytest.raises(TransientError):
                cb.call(fail)
        assert cb.state == CircuitState.OPEN

    def test_open_breaker_fast_fails(self):
        cb = CircuitBreaker(failure_threshold=1)
        with pytest.raises(TransientError):
            cb.call(lambda: (_ for _ in ()).throw(TransientError("boom")))
        assert cb.state == CircuitState.OPEN
        # Next call fails fast without invoking the function:
        with pytest.raises(DegradableError):
            cb.call(lambda: "should-not-run")

    def test_transitions_to_half_open_after_cool_down(self):
        clock = [0.0]
        cb = CircuitBreaker(failure_threshold=1, cool_down=10.0, clock=lambda: clock[0])
        with pytest.raises(TransientError):
            cb.call(lambda: (_ for _ in ()).throw(TransientError("boom")))
        assert cb.state == CircuitState.OPEN
        clock[0] = 15.0  # jump past cool_down
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        with pytest.raises(TransientError):
            cb.call(lambda: (_ for _ in ()).throw(TransientError("x")))
        assert cb.failure_count == 1
        cb.call(lambda: "ok")
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED


# ─────────────────────────────────────────────────────────────────────────────
# Resilience — pipeline timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestWithTimeout:
    def test_returns_on_time(self):
        def fast():
            return 42
        assert with_timeout(fast, timeout_s=2.0) == 42

    def test_raises_on_slow_function(self):
        def slow():
            time.sleep(3.0)
            return "done"
        with pytest.raises(PipelineTimeoutError):
            with_timeout(slow, timeout_s=0.5)

    def test_propagates_inner_exception(self):
        def boom():
            raise ValueError("inner")
        with pytest.raises(ValueError, match="inner"):
            with_timeout(boom, timeout_s=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Caching — response cache
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseCache:
    def test_miss_returns_none(self):
        c = ResponseCache()
        assert c.get("sys", [{"role": "user", "content": "hi"}]) is None
        assert c.misses == 1

    def test_hit_returns_stored_value(self):
        c = ResponseCache()
        c.set("sys", [{"role": "user", "content": "hi"}], "cached-response")
        result = c.get("sys", [{"role": "user", "content": "hi"}])
        assert result == "cached-response"
        assert c.hits == 1

    def test_different_input_is_a_miss(self):
        c = ResponseCache()
        c.set("sys", [{"role": "user", "content": "A"}], "response-A")
        assert c.get("sys", [{"role": "user", "content": "B"}]) is None

    def test_hit_rate(self):
        c = ResponseCache()
        c.set("sys", [{"role": "user", "content": "A"}], "rA")
        c.get("sys", [{"role": "user", "content": "A"}])  # hit
        c.get("sys", [{"role": "user", "content": "A"}])  # hit
        c.get("sys", [{"role": "user", "content": "B"}])  # miss
        assert c.hit_rate() == pytest.approx(2 / 3)

    def test_ttl_expiry(self):
        c = ResponseCache(ttl_seconds=0.05)
        c.set("sys", [{"role": "user", "content": "A"}], "rA")
        time.sleep(0.1)
        assert c.get("sys", [{"role": "user", "content": "A"}]) is None
