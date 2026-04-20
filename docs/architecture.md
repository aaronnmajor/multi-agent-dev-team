# Architecture — Multi-Agent Dev Team

## Week 3: Complete Agent Team v3.0 (current)

Three-agent system with an iterative review loop between Coder and QA, plus both MCP and A2A protocol layers.

```
  START
    |
    v
+-------+      routing == "coder"     +---------+     routing == "qa"      +------+
|  pm   | -------------------------> |  coder  | -----------------------> |  qa  |
+-------+                             +---------+                          +------+
    |                                      ^                                   |
    | (routing == "error")                 |  routing == "coder"               |
    v                                      |  (retry OR next task)             |
   END                                     +-----------------------------------+
                                                                               |
                                                              routing == "done"
                                                                               |
                                                                               v
                                                                              END
```

### QA review loop (Reflexion-style)

1. Coder writes artifact to shared state and routes to QA.
2. QA evaluates the artifact against a rubric (correctness, safety, style, execution result).
3. On fail, if `retry_count < MAX_RETRIES_PER_TASK (2)`, QA routes back to Coder with structured feedback. Coder's next iteration includes the review's `issues` and `suggestions` in its instruction.
4. On pass (or retries exhausted), QA advances `current_task_index` and routes back to Coder for the next task, or END if all tasks are complete.
5. A best-effort artifact is preserved even when retries are exhausted — no code is discarded.

### Updated `ProjectState` (W3 additions in bold)

| Field | Type | Written by | Reducer |
|---|---|---|---|
| `user_requirement` | `str` | caller | none |
| `tech_spec` | `str` | PM | none |
| `tasks` | `list[CodingTask]` | PM | `operator.add` (append) |
| `artifacts` | `list[CodingArtifact]` | Coder | `operator.add` (append) |
| **`reviews`** | **`list[QAReview]`** | **QA** | **`operator.add` (append)** |
| `current_task_index` | `int` | QA (on pass/exhaustion) | none |
| **`retry_count`** | **`int`** | **QA** | **none** |
| `routing` | `str` | all three | none |
| `error` | `str` | PM | none |

### QA Agent (agents/qa_agent.py)

- `review_artifact(task, artifact)` — single LLM call returning a `QAReview` dict with `passed`, `issues`, `suggestions`, `summary`. Malformed JSON defaults to pass (to avoid blocking on transient parse failures, per the Reflexion stopping-condition guidance).
- `qa_node(state)` — LangGraph node that runs the review, updates `reviews` and `retry_count`, advances or retains `current_task_index`, and sets `routing`.

### MCP adapter (tools/mcp_adapter.py)

- `TOOL_REGISTRY` maps tool names to their callable handlers.
- `list_tools()` returns MCP-style descriptors (`name`, `description`, `input_schema`) by introspecting the LangChain `@tool` metadata.
- `call_tool(name, arguments)` dispatches by name, returns the string result. Equivalent to MCP's `tools/call` method.
- This is an in-process adapter rather than a full stdio subprocess server. It demonstrates the MCP contract at the interface level without the transport overhead. A full subprocess server can be added in Week 4 polish.

### A2A protocol (orchestration/a2a.py)

- `Message` dataclass enforces the five mandatory fields (`sender`, `receiver`, `intent`, `payload`, `correlation_id`) and validates sender/receiver/intent in `__post_init__`.
- `Broker` maintains a separate `asyncio.Queue` per agent; senders enqueue by receiver name. Decoupled — no direct references between agents.
- `AGENT_CAPABILITIES` advertises each agent's supported intents (sends/receives) — the in-process equivalent of A2A's agent card discovery.
- `validate_incoming(message, receiver, allowed_senders)` is the trust-boundary check; returns False for unknown intents or unauthorised senders so the caller can log and discard rather than raise.

In the Week 3 graph the protocol is demonstrated via these primitives; LangGraph shared state is still the primary channel between Coder and QA for simplicity. In Week 4 or 5 the review loop can be refactored to actually use the A2A broker if async peer communication is needed.

---

## Week 2: Multi-Agent System v2.0

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
