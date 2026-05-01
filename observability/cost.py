"""
Per-agent token tracking and cost estimation.

Pricing is stored per model in MODEL_PRICES (USD per 1k tokens, input/output).
Each LLM call records a TokenUsage; the CostTracker aggregates them into a
per-agent and per-run report printable at the end of a pipeline invocation.

A per-run CostTracker is kept in a module-level registry keyed by `run_id`
so the PM/Coder/QA agents can record usage without explicitly threading the
tracker through ProjectState. `tracker_for(run_id)` returns (and lazily
creates) the tracker for a run; `write_report(run_id, out_dir)` flushes the
final report JSON to disk and removes the tracker from the registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from observability.logging import get_logger

# Prices as of early 2026. Update as providers change rates.
# Format: model_name -> (input_per_1k, output_per_1k)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o":          (0.0025, 0.010),
    "gpt-4o-mini":     (0.00015, 0.0006),
    "gpt-4.1-mini":    (0.0004, 0.0016),
    "claude-opus-4":   (0.015, 0.075),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-haiku-4":  (0.00025, 0.00125),
}


@dataclass
class TokenUsage:
    agent:      str
    model:      str
    prompt:     int = 0
    completion: int = 0
    cached:     int = 0  # prompt tokens served from provider cache

    @property
    def total_tokens(self) -> int:
        return self.prompt + self.completion

    @property
    def cost_usd(self) -> float:
        p_price, c_price = MODEL_PRICES.get(self.model, (0.005, 0.015))
        billable_prompt = max(0, self.prompt - self.cached)
        return (billable_prompt * p_price + self.completion * c_price) / 1000.0


class CostTracker:
    """Accumulate TokenUsage instances across a pipeline run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id
        self._usages: list[TokenUsage] = []
        self._log = get_logger("cost")

    def record(self, usage: TokenUsage) -> None:
        self._usages.append(usage)
        self._log.info(
            "token_usage",
            run_id=self.run_id,
            agent=usage.agent,
            model=usage.model,
            prompt=usage.prompt,
            completion=usage.completion,
            cached=usage.cached,
            cost_usd=round(usage.cost_usd, 5),
        )

    def report(self) -> dict[str, Any]:
        by_agent: dict[str, dict[str, float]] = {}
        for u in self._usages:
            slot = by_agent.setdefault(
                u.agent, {"prompt": 0, "completion": 0, "cost_usd": 0.0}
            )
            slot["prompt"] += u.prompt
            slot["completion"] += u.completion
            slot["cost_usd"] += u.cost_usd
        total_cost = sum(v["cost_usd"] for v in by_agent.values())
        return {
            "run_id":         self.run_id,
            "by_agent":       {a: {"prompt": int(v["prompt"]),
                                   "completion": int(v["completion"]),
                                   "cost_usd": round(v["cost_usd"], 5)}
                               for a, v in by_agent.items()},
            "total_tokens":   sum(u.total_tokens for u in self._usages),
            "total_cost_usd": round(total_cost, 5),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-run tracker registry
# ─────────────────────────────────────────────────────────────────────────────

_TRACKERS: dict[str, CostTracker] = {}
_REGISTRY_LOCK = Lock()


def tracker_for(run_id: str) -> CostTracker:
    """Return the CostTracker for the given run_id, creating it on first use."""
    with _REGISTRY_LOCK:
        tracker = _TRACKERS.get(run_id)
        if tracker is None:
            tracker = CostTracker(run_id=run_id)
            _TRACKERS[run_id] = tracker
        return tracker


def record_usage_from_response(
    run_id: str,
    agent: str,
    model: str,
    response: Any,
) -> None:
    """Extract usage from an OpenAI-shaped response and record it.

    Tolerates LangChain message responses (which expose ``usage_metadata`` or
    ``response_metadata['token_usage']``) and raw OpenAI completions (which
    expose ``response.usage``). Silently no-ops on shapes we don't recognise
    so the pipeline never breaks because of telemetry.
    """
    if not run_id:
        return
    prompt = completion = cached = 0
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        prompt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
    else:
        # LangChain AIMessage path.
        meta = getattr(response, "usage_metadata", None) or {}
        prompt = int(meta.get("input_tokens", 0) or 0)
        completion = int(meta.get("output_tokens", 0) or 0)
    if prompt == 0 and completion == 0:
        return
    tracker_for(run_id).record(
        TokenUsage(agent=agent, model=model, prompt=prompt, completion=completion, cached=cached)
    )


def write_report(run_id: str, out_dir: str | Path) -> Path | None:
    """Flush the report for ``run_id`` to ``out_dir/<run_id>.json``.

    Returns the file path on success, or None when there is no tracker for
    the run (e.g., the pipeline never made any LLM calls). The tracker is
    removed from the registry after writing so long-lived processes don't
    accumulate state.
    """
    with _REGISTRY_LOCK:
        tracker = _TRACKERS.pop(run_id, None)
    if tracker is None:
        return None
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{run_id}.json"
    file_path.write_text(json.dumps(tracker.report(), indent=2), encoding="utf-8")
    return file_path
