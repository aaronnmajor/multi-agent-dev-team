"""
Orchestration entry point.

For Week 1 the graph is a single-agent ReACT loop (agent <-> tools).
In Week 2 this module grows to add the PM agent and handoff edges.

Run directly to execute the demo task:
    python -m orchestration.graph
"""

from __future__ import annotations

from agents.coder_agent import CoderAgent, build_coder_graph

__all__ = ["CoderAgent", "build_coder_graph"]


if __name__ == "__main__":
    DEMO_TASK = (
        "Write a Python function word_frequency(text: str) -> dict that returns a case-insensitive "
        "word-count dictionary, ignoring punctuation. Save it to word_freq.py, then write and run "
        "a small test script that calls it on 'To be or not to be, that is the question' and prints the result."
    )
    agent = CoderAgent()
    output = agent.run(DEMO_TASK)
    print(output.model_dump_json(indent=2))
