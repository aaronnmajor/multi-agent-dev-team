"""
Coder Agent — Week 1 standalone + Week 2 multi-agent node.

Week 1: CoderAgent class runs a single task end-to-end and returns AgentOutput.
Week 2: coder_node(state) processes one task from ProjectState, writes an artifact back,
        and sets routing to "coder" (more tasks remain) or "done" (all tasks complete).
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_model
from memory import SemanticMemory, SlidingWindowBuffer
from observability import get_logger, trace_span
from orchestration.state import AgentOutput, CodingArtifact, ProjectState
from tools import exec_python, read_file, write_file

load_dotenv(override=True)

MODEL = get_model("coder")
_log = get_logger("coder")
MAX_ITERATIONS = 10
MAX_REACT_ITERATIONS = 8  # Week 2: per-task cap to prevent loops

SYSTEM_PROMPT = """You are an autonomous Coder Agent. Your job is to complete Python coding tasks.

## Workflow (ReACT loop)

1. Think step-by-step before acting. Explain your plan in your first response.
2. Write code using write_file, then execute it with exec_python to verify it works.
3. If execution fails, read the error message, fix the code, and try again.
4. When the task is complete, respond with a final summary (no tool call) including:
   - A brief explanation of what you built
   - The plan you followed
   - The result from running the code

## Rules

- All files go in the workspace/ directory (relative paths only).
- Use subprocess-safe code (no infinite loops, bounded output).
- Maximum 10 iterations. Use them wisely.
- When finished, do NOT call any tool. Just respond with a plain text summary.
"""


TOOLS = [read_file, write_file, exec_python]


class AgentGraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    iterations: int


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("HELICONE_BASE_URL"),
        default_headers={"Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}"},
    ).bind_tools(TOOLS)


def _agent_node(state: AgentGraphState, llm=None) -> dict:
    if llm is None:
        llm = _build_llm()
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
    }


def _should_continue(state: AgentGraphState) -> str:
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_coder_graph():
    """Construct the single-agent LangGraph used by the Week 1 CoderAgent wrapper."""
    workflow = StateGraph(AgentGraphState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


class CoderAgent:
    """Week 1 wrapper: run the graph on a task and return structured output."""

    def __init__(self) -> None:
        self.graph = build_coder_graph()
        self.short_term = SlidingWindowBuffer()
        self.long_term = SemanticMemory()

    def _build_system_prompt(self, task: str) -> str:
        if self.long_term.count() == 0:
            return SYSTEM_PROMPT
        memories = self.long_term.retrieve(task, top_k=3)
        if not memories:
            return SYSTEM_PROMPT
        memory_block = "\n".join(f"- {m}" for m in memories)
        return f"{SYSTEM_PROMPT}\n\n## Relevant memories from prior runs\n\n{memory_block}\n"

    def run(self, task: str) -> AgentOutput:
        system_prompt = self._build_system_prompt(task)
        initial_state: AgentGraphState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task),
            ],
            "task": task,
            "iterations": 0,
        }

        final_state = self.graph.invoke(initial_state, {"recursion_limit": 50})
        iterations_used = final_state.get("iterations", 0)
        stopped_early = iterations_used >= MAX_ITERATIONS

        code, result_text = self._extract_artifacts(final_state["messages"])
        explanation, plan = self._extract_explanation_and_plan(final_state["messages"])

        output = AgentOutput(
            code=code,
            explanation=explanation,
            plan=plan,
            result=result_text,
            iterations_used=iterations_used,
            stopped_early=stopped_early,
        )

        self.long_term.store(f"Task: {task}\nOutcome: {explanation[:500]}")
        return output

    @staticmethod
    def _extract_artifacts(messages: list) -> tuple[str, str]:
        code = ""
        result = ""
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                if name == "write_file" and isinstance(args, dict):
                    code = args.get("content", code)
            if getattr(msg, "type", None) == "tool" or getattr(msg, "name", None) == "exec_python":
                result = getattr(msg, "content", result)
        return code, result

    @staticmethod
    def _extract_explanation_and_plan(messages: list) -> tuple[str, str]:
        assistant_texts = [
            m.content for m in messages
            if getattr(m, "type", None) == "ai" and getattr(m, "content", None)
        ]
        if not assistant_texts:
            return "", ""
        plan = assistant_texts[0]
        explanation = assistant_texts[-1]
        return explanation, plan


# ─────────────────────────────────────────────────────────────────────────────
# Week 2: coder_node for the multi-agent ProjectState graph
# ─────────────────────────────────────────────────────────────────────────────

def _task_instruction(task: dict, spec: str) -> str:
    """Format a single coding task into an instruction for the coder ReACT loop."""
    return (
        f"Tech spec (for context):\n{spec}\n\n"
        f"---\nCurrent task: {task['title']} ({task['task_id']})\n"
        f"Description: {task['description']}\n"
        f"Acceptance criteria: {task['acceptance_criteria']}\n"
        f"Target file: {task['file']}\n\n"
        "Implement the task, write the file to the workspace, execute it (or an appropriate test), "
        "and then summarise what you did."
    )


def _format_feedback_from_latest_review(state: ProjectState, task_id: str) -> str:
    """Find the most recent QA review for this task and format it as feedback for the coder."""
    reviews = state.get("reviews", [])
    last = next((r for r in reversed(reviews) if r.get("task_id") == task_id), None)
    if last is None:
        return ""
    parts = [f"Previous QA review flagged issues (retry {state.get('retry_count', 0)} of 2):"]
    for issue in last.get("issues", []):
        parts.append(f"  - ISSUE: {issue}")
    for suggestion in last.get("suggestions", []):
        parts.append(f"  - SUGGESTION: {suggestion}")
    if last.get("summary"):
        parts.append(f"  Summary: {last['summary']}")
    return "\n".join(parts)


def coder_node(state: ProjectState) -> dict[str, Any]:
    """LangGraph node: process the current task and route to QA for review.

    Reads `tasks[current_task_index]`, runs the single-agent graph on it,
    writes the resulting artifact back, and routes to "qa". Task index is
    advanced by the QA node on pass (or on retry exhaustion), not here.
    """
    tasks = state.get("tasks", [])
    idx = state.get("current_task_index", 0)

    if idx >= len(tasks):
        return {"routing": "done"}

    task = tasks[idx]
    spec = state.get("tech_spec", "")
    instruction = _task_instruction(task, spec)

    # If this is a retry, append the previous QA feedback so the coder can fix the issues.
    feedback = _format_feedback_from_latest_review(state, task["task_id"])
    if feedback:
        instruction = f"{instruction}\n\n---\n{feedback}"

    run_id = state.get("run_id", "")
    with trace_span("coder", "coder_task", run_id, task_id=task["task_id"]):
        agent = CoderAgent()
        output = agent.run(instruction)

    artifact: CodingArtifact = {
        "task_id":     task["task_id"],
        "file":        task["file"],
        "content":     output.code,
        "exec_result": output.result,
    }

    _log.info(
        "coder_task_complete",
        run_id=run_id,
        task_id=task["task_id"],
        code_chars=len(output.code),
        iterations=output.iterations_used,
    )
    return {
        "artifacts": [artifact],
        "routing":   "qa",
    }


if __name__ == "__main__":
    DEMO_TASK = (
        "Write a Python function word_frequency(text: str) -> dict that returns a case-insensitive "
        "word-count dictionary, ignoring punctuation. Save it to word_freq.py, then write and run "
        "a small test script that calls it on 'To be or not to be, that is the question' and prints the result."
    )
    agent = CoderAgent()
    output = agent.run(DEMO_TASK)
    print(output.model_dump_json(indent=2))
