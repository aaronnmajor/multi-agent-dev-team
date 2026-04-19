# Architecture — Multi-Agent Dev Team

## Week 1: Coder Agent v1.0

Single autonomous agent that accepts a natural-language coding task and returns structured, validated output.

```
User task
    |
    v
+-----------+      +---------+
|   agent   | <--> |  tools  |
|  (LLM +   |      |  read   |
|  ReACT)   |      |  write  |
+-----------+      |  exec   |
    |              +---------+
    |
    v
AgentOutput (Pydantic)
  - code
  - explanation
  - plan
  - result
```

### Components

| Component | File | Purpose |
|---|---|---|
| Coder Agent | `agents/coder_agent.py` | LangGraph StateGraph wrapping an LLM + tool loop |
| Tools | `tools/` | `read_file`, `write_file`, `exec_python` (subprocess sandbox) |
| Memory | `memory.py` | Sliding-window short-term + Chroma long-term |
| Orchestration | `orchestration/graph.py` | StateGraph wiring; entry point |
| State | `orchestration/state.py` | Pydantic `AgentState` and `AgentOutput` |

### Loop structure

1. System prompt is built from the base prompt plus top-3 memories retrieved by semantic similarity to the task.
2. The LangGraph agent node calls the LLM. If the LLM produces tool calls, the ToolNode executes them and re-enters the agent node.
3. The loop terminates when the LLM returns a plain text response (no tool calls) or the iteration counter hits `MAX_ITERATIONS = 10`.
4. `CoderAgent.run()` extracts the final code, plan, explanation, and execution result from the message log and returns an `AgentOutput`.

### Safety

- Code execution uses `subprocess.run` with a hard 10-second default timeout (clamped to 30s max).
- Tool outputs are truncated at 2000 characters to protect the context window.
- All file I/O is confined to the `workspace/` directory.

## Roadmap

| Week | Additions |
|---|---|
| 2 | Product Manager agent, LangGraph handoffs, shared state |
| 3 | QA & Debugger agent, MCP / A2A protocols |
| 4 | Observability, cost tracking, error recovery, Docker deployment |
| 5 | Final polish, full test suite, demo recording |
