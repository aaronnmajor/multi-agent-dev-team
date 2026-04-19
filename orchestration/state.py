"""
Pydantic models for agent state and structured output.

AgentState holds mutable state during the ReACT loop.
AgentOutput is the final validated deliverable returned to the caller.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Mutable state carried through the LangGraph StateGraph."""

    task: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    max_iters: int = 10
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    output: Optional[str] = None
    error: Optional[str] = None


class AgentOutput(BaseModel):
    """Final structured output returned by the Coder Agent.

    The four required fields per the Week 1 project spec:
      - code:        the generated Python source code
      - explanation: natural-language explanation of what was built
      - plan:        the execution plan the agent followed
      - result:      the result of running the code (stdout, test output, etc.)
    """

    code: str = Field(description="The generated Python source code.")
    explanation: str = Field(description="Natural-language explanation of what was built.")
    plan: str = Field(description="The execution plan the agent followed.")
    result: str = Field(description="Output from running the generated code.")
    iterations_used: int = Field(default=0, description="ReACT loop iterations consumed.")
    stopped_early: bool = Field(default=False, description="True if MAX_ITERATIONS was reached.")
