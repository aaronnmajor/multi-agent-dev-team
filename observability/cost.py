"""
Per-agent token tracking and cost estimation.

Pricing is stored per model in MODEL_PRICES (USD per 1k tokens, input/output).
Each LLM call records a TokenUsage; the CostTracker aggregates them into a
per-agent and per-run report printable at the end of a pipeline invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
