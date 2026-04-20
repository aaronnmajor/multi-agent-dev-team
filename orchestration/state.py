"""
State schemas for the multi-agent graph.

Week 1: AgentState / AgentOutput — single-agent ReACT loop.
Week 2: CodingTask / CodingArtifact / ProjectState — shared state across PM and Coder agents.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Week 1 — single-agent output schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Legacy Week 1 single-agent state (kept for backward compatibility)."""

    task: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    max_iters: int = 10
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    output: Optional[str] = None
    error: Optional[str] = None


class AgentOutput(BaseModel):
    """Final structured output from the Coder Agent.

    Four required fields per the Week 1 project spec:
      code, explanation, plan, result.
    """

    code: str = Field(description="Generated Python source code.")
    explanation: str = Field(description="Natural-language explanation of what was built.")
    plan: str = Field(description="Execution plan the agent followed.")
    result: str = Field(description="Output from running the generated code.")
    iterations_used: int = Field(default=0)
    stopped_early: bool = Field(default=False)


# ─────────────────────────────────────────────────────────────────────────────
# Week 2 — multi-agent shared state
# ─────────────────────────────────────────────────────────────────────────────

class CodingTask(TypedDict):
    """A single task produced by the PM agent and consumed by the Coder agent."""

    task_id: str
    title: str
    description: str
    acceptance_criteria: str
    status: str            # "pending" | "in_progress" | "done" | "failed"
    file: str              # target filename the coder should create/modify


class CodingArtifact(TypedDict):
    """A code artifact produced by the Coder agent."""

    task_id: str
    file: str
    content: str
    exec_result: str       # stdout / error from running the artifact


class QAReview(TypedDict):
    """A review produced by the QA & Debugger agent for a single artifact."""

    task_id: str
    passed: bool
    issues: list[str]         # Concrete bugs / style violations found
    suggestions: list[str]    # Recommended fixes
    summary: str              # One-line summary for the coder


class ProjectState(TypedDict):
    """Shared state for the PM + Coder + QA multi-agent graph.

    Fields written by multiple agents use `Annotated[list, add]` as a reducer
    so values accumulate rather than overwrite.
    """

    user_requirement: str                            # Input
    tech_spec: str                                   # PM -> state
    tasks: Annotated[list[CodingTask], add]          # PM -> state (reducer: append)
    artifacts: Annotated[list[CodingArtifact], add]  # Coder -> state (reducer: append)
    reviews: Annotated[list[QAReview], add]          # QA -> state (reducer: append) (W3)
    current_task_index: int                          # Coder/QA loop counter
    retry_count: int                                 # Per-task retry counter (W3)
    routing: str                                     # Flow: "coder" | "qa" | "done" | "error"
    error: str                                       # Error message if any
