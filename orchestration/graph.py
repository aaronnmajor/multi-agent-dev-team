"""
Multi-agent orchestration graph.

Week 1: single-agent ReACT loop (agent <-> tools) — available via `build_coder_graph`.
Week 2: PM + Coder graph with conditional routing and self-loop on the coder node.
Week 3: PM + Coder + QA graph with Coder ↔ QA review loop and per-task retry cap.

Run directly to execute the Week 3 end-to-end demo:
    python -m orchestration.graph
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.coder_agent import CoderAgent, build_coder_graph, coder_node
from agents.pm_agent import pm_node
from agents.qa_agent import qa_node
from orchestration.state import ProjectState

__all__ = ["CoderAgent", "build_coder_graph", "build_graph", "create_graph"]


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routers
# ─────────────────────────────────────────────────────────────────────────────

def route_after_pm(state: ProjectState) -> str:
    """After the PM agent runs, route to the coder (tasks ready) or END (error)."""
    if state.get("routing", "") == "coder":
        return "coder"
    return END


def route_after_coder(state: ProjectState) -> str:
    """After the coder produces an artifact, route to QA for review."""
    if state.get("routing", "") == "qa":
        return "qa"
    return END  # "done" or unexpected -> finish


def route_after_qa(state: ProjectState) -> str:
    """After QA reviews, loop back to coder (retry or next task) or END (all tasks done)."""
    if state.get("routing", "") == "coder":
        return "coder"
    return END  # "done" or "error" -> finish


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Assemble the Week 3 multi-agent StateGraph (PM -> Coder <-> QA review loop)."""
    builder = StateGraph(ProjectState)
    builder.add_node("pm", pm_node)
    builder.add_node("coder", coder_node)
    builder.add_node("qa", qa_node)

    builder.add_edge(START, "pm")
    builder.add_conditional_edges("pm", route_after_pm, {"coder": "coder", END: END})
    builder.add_conditional_edges("coder", route_after_coder, {"qa": "qa", END: END})
    builder.add_conditional_edges("qa", route_after_qa, {"coder": "coder", END: END})

    return builder.compile()


def create_graph():
    """Public factory used by main.py and tests."""
    return build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DEMO_REQUIREMENT = (
        "Build a small command-line utility that counts word frequency in a text file. "
        "The user invokes it as `python word_count.py <path>` and it prints the top 5 "
        "most common words with their counts, case-insensitive, ignoring punctuation."
    )

    requirement = sys.argv[1] if len(sys.argv) > 1 else DEMO_REQUIREMENT

    graph = create_graph()
    initial_state: ProjectState = {
        "user_requirement":   requirement,
        "tech_spec":          "",
        "tasks":              [],
        "artifacts":          [],
        "reviews":            [],
        "current_task_index": 0,
        "retry_count":        0,
        "routing":            "",
        "error":              "",
    }

    final = graph.invoke(initial_state, {"recursion_limit": 100})

    print("=" * 60)
    print("TECH SPEC")
    print("=" * 60)
    print(final.get("tech_spec", "")[:1000])
    print()
    print("=" * 60)
    print(f"TASKS ({len(final.get('tasks', []))})")
    print("=" * 60)
    for t in final.get("tasks", []):
        print(f"  [{t['task_id']}] {t['title']} -> {t['file']}")
    print()
    print("=" * 60)
    print(f"ARTIFACTS ({len(final.get('artifacts', []))})")
    print("=" * 60)
    for a in final.get("artifacts", []):
        print(f"  [{a['task_id']}] {a['file']} ({len(a['content'])} chars)")
    print()
    print("=" * 60)
    print(f"QA REVIEWS ({len(final.get('reviews', []))})")
    print("=" * 60)
    for r in final.get("reviews", []):
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"  [{r['task_id']}] {verdict} - {r['summary']}")
        for issue in r.get("issues", [])[:3]:
            print(f"    - {issue}")
    if final.get("error"):
        print()
        print(f"ERROR: {final['error']}")
