"""
Coder Agent v1.0 — Week 1 project deliverable.

A single autonomous agent that:
  - Accepts a natural-language coding task
  - Plans its approach, decomposing via chain-of-thought
  - Uses file I/O and sandboxed code execution tools
  - Returns structured Pydantic output (code, explanation, plan, result)

Implemented with LangGraph (StateGraph + ToolNode) using native function-calling.
Memory is a two-tier system (sliding-window short-term + Chroma long-term).
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from memory import SemanticMemory, SlidingWindowBuffer
from orchestration.state import AgentOutput
from tools import exec_python, read_file, write_file

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
MAX_ITERATIONS = 10

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
    """LangGraph state: accumulates messages through add_messages reducer."""

    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    iterations: int


def _build_llm() -> ChatOpenAI:
    """Build the LLM client with tools bound, routed through Helicone."""
    return ChatOpenAI(
        model=MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("HELICONE_BASE_URL"),
        default_headers={"Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}"},
    ).bind_tools(TOOLS)


def _agent_node(state: AgentGraphState, llm=None) -> dict:
    """Single LLM call. Increments iteration counter."""
    if llm is None:
        llm = _build_llm()
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
    }


def _should_continue(state: AgentGraphState) -> str:
    """Loop guard: route to tools if the LLM requested them, end otherwise or at max iter."""
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_coder_graph():
    """Construct the LangGraph StateGraph for the Coder Agent."""
    workflow = StateGraph(AgentGraphState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


class CoderAgent:
    """High-level wrapper that runs the graph and produces structured output."""

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
        """Run the agent on a task and return structured output."""
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
        """Pull the last write_file content and the last exec_python result from the message log."""
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
        """Use the first assistant text (the plan) and the last assistant text (the explanation)."""
        assistant_texts = [
            m.content for m in messages
            if getattr(m, "type", None) == "ai" and getattr(m, "content", None)
        ]
        if not assistant_texts:
            return "", ""
        plan = assistant_texts[0]
        explanation = assistant_texts[-1]
        return explanation, plan


if __name__ == "__main__":
    DEMO_TASK = (
        "Write a Python function word_frequency(text: str) -> dict that returns a case-insensitive "
        "word-count dictionary, ignoring punctuation. Save it to word_freq.py, then write and run "
        "a small test script that calls it on 'To be or not to be, that is the question' and prints the result."
    )
    agent = CoderAgent()
    output = agent.run(DEMO_TASK)
    print("=" * 60)
    print(f"CODE ({len(output.code)} chars):\n{output.code[:500]}...")
    print(f"\nPLAN:\n{output.plan[:500]}")
    print(f"\nEXPLANATION:\n{output.explanation[:500]}")
    print(f"\nRESULT:\n{output.result[:500]}")
    print(f"\nIterations: {output.iterations_used} | Stopped early: {output.stopped_early}")
