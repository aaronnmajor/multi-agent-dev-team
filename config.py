"""
Per-agent model selection.

Tiered strategy: the PM agent drives requirements into specs (cascading errors
warrant a stronger model); Coder and QA operate on structured inputs and
verifiable outputs, which suits a cheaper mid-tier model.

Swap model strings here without touching agent logic.
"""

from __future__ import annotations

AGENT_MODELS: dict[str, str] = {
    "pm":    "gpt-4.1-mini",   # Could upgrade to gpt-4o for production.
    "coder": "gpt-4.1-mini",
    "qa":    "gpt-4.1-mini",
}


def get_model(agent: str) -> str:
    """Return the model name configured for the given agent."""
    if agent not in AGENT_MODELS:
        raise KeyError(f"No model configured for agent {agent!r}. Known: {list(AGENT_MODELS)}")
    return AGENT_MODELS[agent]
