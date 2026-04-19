"""
Smoke test for the Week 1 Coder Agent v1.0.

Verifies the agent completes a short task and returns structured output
with all four required fields (code, explanation, plan, result).

This test makes real LLM calls. Skip with `pytest -m "not llm"` if desired.
"""

from __future__ import annotations

import pytest

from agents.coder_agent import CoderAgent
from orchestration.state import AgentOutput


SMOKE_TASK = (
    "Write a one-line Python script to workspace/hello.py that prints 'hello from smoke test'. "
    "Then execute it with exec_python and report the output."
)


@pytest.mark.llm
@pytest.mark.timeout(120)
def test_coder_agent_smoke():
    """End-to-end: agent returns AgentOutput with all four required fields populated."""
    agent = CoderAgent()
    output = agent.run(SMOKE_TASK)

    assert isinstance(output, AgentOutput), "Agent must return an AgentOutput instance"

    assert output.code, "AgentOutput.code must be non-empty"
    assert output.explanation, "AgentOutput.explanation must be non-empty"
    assert output.plan, "AgentOutput.plan must be non-empty"
    assert output.result, "AgentOutput.result must be non-empty"

    assert output.iterations_used > 0, "Iteration counter should have advanced"
    assert output.iterations_used <= 10, "Should not exceed MAX_ITERATIONS"


def test_agent_output_schema():
    """AgentOutput schema has the four required fields defined by the project spec."""
    required_fields = {"code", "explanation", "plan", "result"}
    schema_fields = set(AgentOutput.model_fields.keys())
    assert required_fields.issubset(schema_fields), (
        f"AgentOutput must include the four required fields: {required_fields}. "
        f"Got: {schema_fields}"
    )
