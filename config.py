"""
Per-agent model selection.

Tiered strategy: the PM agent drives requirements into specs (cascading errors
warrant a stronger model); Coder and QA operate on structured inputs and
verifiable outputs, which suits a cheaper mid-tier model.

Swap model strings here without touching agent logic. All entries must
correspond to a model present in `observability.cost.MODEL_PRICES` so the
CostTracker can produce accurate USD figures.
"""

from __future__ import annotations

import os

# Tiered defaults — PM gets the stronger model, Coder and QA get the cheaper
# tier. Override per-agent via env vars for cost-optimisation experiments
# (used by Innovation #1).
AGENT_MODELS: dict[str, str] = {
    "pm":    os.getenv("MODEL_PM",    "gpt-4o"),
    "coder": os.getenv("MODEL_CODER", "gpt-4o-mini"),
    "qa":    os.getenv("MODEL_QA",    "gpt-4o-mini"),
}


def get_model(agent: str) -> str:
    """Return the model name configured for the given agent."""
    if agent not in AGENT_MODELS:
        raise KeyError(f"No model configured for agent {agent!r}. Known: {list(AGENT_MODELS)}")
    return AGENT_MODELS[agent]
