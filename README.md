# Multi-Agent Dev Team

**Capstone project for *Build Autonomous Multi-Agent Systems* (Saras AI Institute, Spring 2026).**

A LangGraph-based autonomous software development team. The system grows weekly:
Week 1 is a single Coder Agent; by Week 5 it is a PM + Coder + QA team with shared state, feedback loops, and production hardening.

## Current state (Week 2: Multi-Agent System v2.0)

A two-agent system built on LangGraph shared state:

- **PM Agent** — converts a raw user requirement into a Markdown tech spec and a JSON task list.
- **Coder Agent** — picks tasks off the shared state one at a time, writes code, executes it, and stores the artifact back in state. Self-loops until all tasks are complete.

The graph is `START → PM → (conditional) → Coder → (self-loop until done) → END`. All state lives in a typed `ProjectState` dict; the `tasks` and `artifacts` fields use `Annotated[list, add]` reducers so values accumulate across agent turns rather than overwrite.

### Week 1 still available

The single-agent Coder (`agents.CoderAgent`) is retained unchanged — it accepts a coding task directly and returns a validated `AgentOutput` with the four required fields (`code`, `explanation`, `plan`, `result`).

## Requirements

- Python 3.11+
- Helicone / OpenRouter API keys (provided in the course; see `.env.example`)

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/aaronnmajor/multi-agent-dev-team.git
cd multi-agent-dev-team

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Fill in HELICONE_BASE_URL, OPENROUTER_API_KEY, HELICONE_API_KEY
```

## Usage

### Run the full multi-agent graph (Week 2)

```bash
python -m orchestration.graph
# or with a custom requirement:
python -m orchestration.graph "Build a CLI that reverses the lines of a text file."
```

Runs the PM → Coder pipeline end-to-end: the PM produces a spec and task list, the coder works through the tasks one at a time, and the final state is printed (spec, tasks, artifacts).

### Run the single Coder Agent (Week 1)

```bash
python -m agents.coder_agent
```

Skips the PM step and runs the demo coding task directly. Prints `AgentOutput` as JSON.

### Run the smoke test

```bash
pytest tests/test_smoke.py -v
```

Skip the real-LLM smoke test if you only want the schema check:

```bash
pytest tests/ -v -m "not llm"
```

### Run in Docker

```bash
docker compose up --build
```

## Project structure

```
multi-agent-dev-team/
├── agents/                  # One module per agent
│   ├── coder_agent.py       # CoderAgent class (W1) + coder_node (W2)
│   └── pm_agent.py          # PM agent: requirement -> spec -> tasks (W2)
├── tools/                   # Shared tool definitions
│   ├── file_io.py           # read_file, write_file
│   └── code_executor.py     # exec_python (subprocess sandbox)
├── memory.py                # SlidingWindowBuffer + Chroma SemanticMemory
├── orchestration/           # LangGraph graph + state schemas
│   ├── graph.py             # Multi-agent graph + entry point
│   └── state.py             # ProjectState TypedDict, Pydantic AgentOutput
├── tests/                   # Smoke + unit tests
│   ├── test_smoke.py        # W1 end-to-end (LLM)
│   └── test_multi_agent.py  # W2 PM + routers + graph (mocked)
├── docs/
│   └── architecture.md
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Implementation decisions

### Framework: LangGraph

Picked LangGraph over AutoGen and CrewAI for three reasons:

1. **Course focus** — Week 2 builds a multi-agent graph in LangGraph; starting here avoids a rewrite.
2. **Inspectable control flow** — every edge and node is explicit, which matters when debugging a multi-step agent.
3. **Mature Python-native API** — `StateGraph` + `ToolNode` + `bind_tools` cover the common cases without framework magic.

### LLM: gpt-4.1-mini via Helicone

All LLM calls are routed through Helicone for usage tracking. The model is `gpt-4.1-mini` (same as the course labs). Swapping models is a one-line change in `agents/coder_agent.py`.

### Memory: two-tier (sliding window + Chroma)

- **Short-term:** `SlidingWindowBuffer` keeps the last 20 conversation turns.
- **Long-term:** `SemanticMemory` writes task outcomes to a Chroma collection, retrieved by cosine similarity on the current task string. The top-3 memories are injected into the system prompt.

In Week 1 this is primarily a loading point for Week 3's QA feedback loop — the PM and QA agents will read and write the same store.

### Structured output: Pydantic `AgentOutput`

The four required fields (`code`, `explanation`, `plan`, `result`) are defined as required on the Pydantic model; validation fails loudly if any is missing. Two extra fields (`iterations_used`, `stopped_early`) are included for observability.

### ReACT loop: native function-calling

Rather than parse a text ReACT format (Thought / Action / Observation), the agent uses the LLM's native tool-calling. LangGraph's `ToolNode` handles dispatch, which removes the brittle parser code and lets the model use parallel tool calls when appropriate.

### Loop guards

- `MAX_ITERATIONS = 10` — hard ceiling on LangGraph agent-node invocations.
- `subprocess.TimeoutExpired` — kills runaway scripts at 10s (configurable, max 30s).
- Tool output truncation at 2000 chars — prevents context overflow.

## Roadmap

| Week | Deliverable | Additions |
|---|---|---|
| 1 | Coder Agent v1.0 | Memory, tools, ReACT loop, Pydantic output — **current** |
| 2 | Multi-Agent v2.0 | PM agent, LangGraph handoff, shared state |
| 3 | Complete Team v3.0 | QA & Debugger agent, MCP / A2A protocols, feedback loop |
| 4 | Production v4.0 | Observability, cost tracking, error recovery, Docker deploy |
| 5 | Final Submission | End-to-end polish, full documentation, demo recording |

## Course

- **Course:** Build Autonomous Multi-Agent Systems (Saras AI Institute)
- **Instructor:** Anshuman Singh
- **Student:** Aaron Major (GitHub: [@aaronnmajor](https://github.com/aaronnmajor))
- **Term:** Spring 2026
