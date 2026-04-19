# Architecture — Multi-Agent Dev Team

## Week 2: Multi-Agent System v2.0 (current)

PM Agent hands off to Coder Agent via shared `ProjectState` on a LangGraph StateGraph.

```
  START
    |
    v
+-------+      (routing == "coder")     +---------+
|  pm   | ---------------------------> |  coder  | <-+
+-------+                               +---------+   |
    |                                       |         |
    | (routing == "error")                  |         |
    v                                       | (routing == "coder", more tasks)
   END                                      v
                                     (routing == "done")
                                            |
                                            v
                                           END
```

### ProjectState (TypedDict, orchestration/state.py)

| Field | Type | Written by | Reducer |
|---|---|---|---|
| `user_requirement` | `str` | caller | none |
| `tech_spec` | `str` | PM | none |
| `tasks` | `list[CodingTask]` | PM | `operator.add` (append) |
| `artifacts` | `list[CodingArtifact]` | Coder | `operator.add` (append) |
| `current_task_index` | `int` | Coder | none |
| `routing` | `str` | PM, Coder | none (overwrite each turn) |
| `error` | `str` | PM | none |

### PM Agent (agents/pm_agent.py)

- `build_tech_spec(requirement)` — LLM call: requirement -> Markdown spec with five sections (Overview, Functional, Non-Functional, File Structure, Constraints).
- `decompose_into_tasks(spec)` — LLM call: spec -> JSON task list (strips accidental markdown fences, fills defaults for missing fields, returns `[]` on parse failure).
- `pm_node(state)` — runs both in sequence, sets `routing="coder"` on success or `"error"` if zero tasks were produced.

### Coder Node (agents/coder_agent.py)

- `coder_node(state)` — picks `tasks[current_task_index]`, instantiates the Week 1 `CoderAgent`, runs it on the formatted task instruction, appends a `CodingArtifact` to state, increments the task index, and sets `routing="coder"` (more tasks) or `"done"` (queue drained).

### Routers (orchestration/graph.py)

- `route_after_pm` — `"coder"` on success, `END` otherwise.
- `route_after_coder` — `"coder"` to self-loop, `END` when all tasks done.

---

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
