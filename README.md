# Multi-Agent Dev Team

**Capstone project for *Build Autonomous Multi-Agent Systems* (Saras AI Institute, Spring 2026).**

A LangGraph-based autonomous software development team. The system grows weekly:
Week 1 is a single Coder Agent; by Week 5 it is a PM + Coder + QA team with shared state, feedback loops, and production hardening.

## System diagram

```
                     +-------------------------------------+
                     |             ProjectState            |
                     |  (TypedDict shared across all nodes)|
                     +-------------------------------------+
                                       ^
   START ──> [ PM Agent ] ──tasks──> [ Coder Agent ] ──artifact──> [ QA Agent ]
                  |                       ^                              |
                  |                       |  retry with feedback         |
                  |                       +------- (fail, <2 retries) ---+
                  |                                                      |
                  |                                                      v
                  +───────── error ──> END <── done (all tasks reviewed) +

   Cross-cutting: structured JSON logs · run_id propagation · trace_span ·
   token usage / cost report · retry-with-backoff · circuit breaker ·
   pipeline timeout · SHA-256 keyed response cache · Chroma long-term memory
```

The PM emits a tech spec and task list, the Coder works tasks one at a time
(self-critiquing each output), and the QA agent reviews the artifact and
either advances or sends it back for up to two retries. Every LLM call is
attributed to one agent and one `run_id` so the cost report at
`docs/cost_reports/<run_id>.json` shows the per-agent token + USD breakdown.

## Current state (Week 5: Polished Submission v5.0)

Week 5 polish on top of Week 4:

- **Self-critique** in the Coder agent (`agents/coder_agent.py::self_reflect`) — Reflexion-style review pass before the artifact reaches QA.
- **Tiered models** — `config.AGENT_MODELS` defaults to `gpt-4o` for PM and `gpt-4o-mini` for Coder/QA, with per-agent env-var overrides for cost-optimisation experiments.
- **Cost report auto-write** — every pipeline run drops a JSON report at `docs/cost_reports/<run_id>.json` with per-agent token + USD totals.
- **Task-list cap** — PM agent runs a consolidation pass when initial decomposition exceeds 8 tasks, preventing overgrown task fan-out.
- **Persistent Chroma** — `docker compose up` starts a Chroma service with healthcheck and the agent waits for it via `depends_on: condition: service_healthy`. Outside Docker the in-process client is still used so tests stay fast.

## Previous milestone (Week 4: Production-Ready System v4.0)

Week 3's three-agent team hardened for production:

- **Observability** — JSON-structured logs via stdlib `logging` with per-agent loggers; `run_id` propagated through `ProjectState` so every log event in a pipeline invocation links back to the same trace; `trace_span` context manager emits `span_start` / `span_end` / `span_error` events with duration.
- **Error handling** — structured exception hierarchy (`TransientError`, `PermanentError`, `DegradableError`), `retry_with_backoff` decorator with full jitter, `CircuitBreaker` with CLOSED/OPEN/HALF-OPEN state machine, and pipeline-level `with_timeout` watchdog.
- **Cost optimisation** — `TokenUsage` dataclass + `CostTracker` aggregator with per-model pricing; tiered `AGENT_MODELS` config so the model for each agent can be tuned independently; simple `ResponseCache` keyed by SHA-256 hash of the full prompt (TTL-based).
- **Test suite** — 79 deterministic non-LLM tests plus one LLM-gated end-to-end smoke test; covers unit (format, token math, routers), integration (graph assembly), and resilience (retry/breaker state transitions).
- **Docker deployment** — Dockerfile runs as a non-root `agent` user, includes a healthcheck that imports the graph module, docker-compose mounts `workspace/` and `logs/` volumes.

### Week 3 still in place

The three-agent team (PM, Coder, QA) with iterative review loop, MCP adapter, and A2A protocol primitives is unchanged. Week 4 layers production concerns around it without altering the core orchestration.

---

## Previous milestone (Week 3: Complete Agent Team v3.0)

A three-agent system with an iterative review loop and both MCP and A2A protocol layers:

- **PM Agent** — converts a raw user requirement into a Markdown tech spec and a JSON task list.
- **Coder Agent** — picks tasks off the shared state one at a time, writes code, executes it, and stores the artifact back in state.
- **QA & Debugger Agent** — reviews each artifact against a rubric (correctness, safety, style, execution). On fail, sends the coder back for up to 2 retries with structured feedback. On pass (or retry exhaustion), advances to the next task.

**Graph:** `START → PM → Coder → QA → (retry | next task | END)`

**Protocol layers:**
- **MCP adapter** (`tools/mcp_adapter.py`) — tool registry and dispatch that mirrors the MCP contract. Demonstrates agent-to-tool protocol at the interface level (full stdio subprocess server deferred to Week 4 polish).
- **A2A module** (`orchestration/a2a.py`) — five-field `Message` dataclass, validating `Broker` with per-agent `asyncio.Queue`, capability advertisement, and trust-boundary intent validation. Demonstrates agent-to-agent protocol for Coder ↔ QA peer messaging.

### Week 1 still available

The single-agent Coder (`agents.CoderAgent`) is retained unchanged — it accepts a coding task directly and returns a validated `AgentOutput` with the four required fields (`code`, `explanation`, `plan`, `result`).

## Quickstart (clone-to-run in under 10 minutes)

### Prerequisites

- **Python 3.11+** (3.13 also tested)
- **API keys** — Helicone proxy + OpenRouter or direct OpenAI. The course provides them; otherwise sign up at [helicone.ai](https://helicone.ai) and [openrouter.ai](https://openrouter.ai).
- **Docker Desktop** *(optional)* — only needed for `docker compose up`. The Python entry point works without it.

### 1. Clone and install (≈2 min)

```bash
git clone https://github.com/aaronnmajor/multi-agent-dev-team.git
cd multi-agent-dev-team
pip install -r requirements.txt
```

### 2. Configure secrets (≈2 min)

```bash
cp .env.example .env
```

Open `.env` and fill in three required variables:

| Variable | What it is | Example |
|---|---|---|
| `HELICONE_BASE_URL` | Helicone proxy endpoint | `https://oai.helicone.ai/v1` |
| `OPENROUTER_API_KEY` | API key for the upstream model provider | `sk-or-v1-…` |
| `HELICONE_API_KEY` | Helicone usage-tracking key | `sk-helicone-…` |

Optional (off by default):

| Variable | Effect |
|---|---|
| `MODEL_PM`, `MODEL_CODER`, `MODEL_QA` | Per-agent model overrides (anything in `MODEL_PRICES`). |
| `PM_DEBATE_MODE=true` | Enables Innovation #2 (PM debate / consensus layer). |
| `CODER_SELF_REFLECTION=false` | Disables Coder self-critique pass (default: enabled). |
| `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` | Sends every node as a span to LangSmith. |
| `CHROMA_HOST`, `CHROMA_PORT` | Use a remote Chroma server (auto-set by docker-compose). |

### 3. Run the demo (≈30 sec)

```bash
python -m orchestration.graph
```

This runs the rubric demo task ("Build a Python module that implements a binary search tree with insert, search, and in-order traversal methods") through PM → Coder → QA. Final spec, tasks, artifacts, and reviews are printed; a per-run cost report lands in `docs/cost_reports/<run_id>.json`.

Provide your own requirement on the command line:

```bash
python -m orchestration.graph "Build a CLI that counts word frequency in a text file."
```

### 4. Verify with tests

```bash
# Mocked deterministic suite (fast, no API calls, no cost)
pytest -m "not llm" -q

# Full suite including the live LLM smoke test
pytest -q
```

Expected: **93 mocked tests pass** plus the live LLM smoke test (95 total) as of the Week 5 polish. The smoke test makes real API calls (≈1k tokens).

### 5. Run in Docker (≈3 min)

```bash
docker compose up --build
```

This starts a `chroma` service (with a healthcheck), waits for it, then runs the agent container. The agent uses a non-root user and its own healthcheck imports the compiled graph, so an unhealthy install fails fast. Cost reports written inside the container appear at `docs/cost_reports/` on your host.

## Demo recording

A 4-minute walkthrough covering the system architecture and an end-to-end run of the BST grading task is linked here:

> **▶ [Demo video](#)** *(link added at submission time)*

## Documentation

| Document | What's in it |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Canonical system diagram (Mermaid), per-agent responsibilities, framework justification, operational topology. |
| [`docs/REFLECTION.md`](docs/REFLECTION.md) | 700-word post-mortem: three design decisions, the hardest bug, what I'd do differently. |
| [`docs/test_results.md`](docs/test_results.md) | End-to-end run logs for the four grading tasks plus the Innovation #1 cost-optimisation experiment. |
| [`docs/cost_reports/`](docs/cost_reports/) | Per-run JSON cost reports auto-written by every pipeline invocation. |
| [`docs/api/`](docs/api/) | `pdoc`-generated HTML API reference for every public module. |
| [`docs/DEMO_RECORDING_GUIDE.md`](docs/DEMO_RECORDING_GUIDE.md) | Script outline for the submission demo video. |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'No model configured for agent ...'` | A custom `MODEL_*` env var was set to a name not in `MODEL_PRICES`. | Use one of the keys in `observability/cost.py::MODEL_PRICES`, or add a price entry for your model. |
| `chromadb.errors.ChromaError: Could not connect ...` | `CHROMA_HOST` is set but no Chroma server is reachable. | Either start `docker compose up chroma`, or unset `CHROMA_HOST` to use the in-process client. |
| Pipeline runs but `docs/cost_reports/<run_id>.json` is empty / missing | All LLM calls were served by the response cache, or the run hit an error before any LLM call. | Check the JSON logs for `cost_report_written` events; an empty report is still emitted as `{}`. |
| Smoke test (`pytest -m llm`) hangs | Live LLM call queued behind a rate limit. | The test has a 120-second pytest timeout; let it run, or skip with `-m "not llm"`. |
| `docker compose up` exits immediately on Windows with `linux/amd64` warning | Docker Desktop not started or running with the wrong engine. | Start Docker Desktop and re-run. |

## Run modes at a glance

```bash
# Full PM + Coder + QA pipeline (recommended)
python -m orchestration.graph "<your requirement>"

# Single-agent Week 1 Coder (no PM, no QA)
python -m agents.coder_agent

# Mocked test suite
pytest -m "not llm" -q

# Live smoke test (one real LLM call)
pytest tests/test_smoke.py -v

# Docker (Chroma + agent, healthchecks, persistent memory)
docker compose up --build
```

## Project structure

```
multi-agent-dev-team/
├── agents/                   # One module per agent
│   ├── coder_agent.py        # CoderAgent (W1) + coder_node (W2/W3) + self_reflect (W5)
│   ├── pm_agent.py           # PM: spec -> tasks (W2) + max-8-task consolidation (W5)
│   ├── pm_debate.py          # Debate / consensus PM phase (W5 Innovation #2)
│   └── qa_agent.py           # QA & Debugger: rubric review + feedback (W3)
├── tools/                    # Shared tool definitions
│   ├── file_io.py
│   ├── code_executor.py      # exec_python (subprocess sandbox)
│   └── mcp_adapter.py        # MCP tool registry + dispatch (W3)
├── orchestration/
│   ├── graph.py              # Multi-agent graph + cost-report write (W5)
│   ├── state.py              # ProjectState, AgentOutput, etc.
│   └── a2a.py                # A2A Message, Broker, capability advertisement (W3)
├── observability/            # W4 + W5
│   ├── logging.py            # JSON formatter + per-agent loggers
│   ├── tracing.py            # run_id + trace_span
│   └── cost.py               # TokenUsage + CostTracker + per-run registry (W5)
├── resilience/               # W4
│   ├── retry.py              # retry_with_backoff
│   ├── circuit_breaker.py    # CLOSED/OPEN/HALF-OPEN
│   └── timeout.py            # with_timeout watchdog
├── caching/                  # W4
│   └── response_cache.py     # SHA-256-keyed TTL cache
├── memory.py                 # Sliding window + Chroma (HttpClient under Docker, W5)
├── exceptions.py             # AgentError hierarchy (W4)
├── config.py                 # Tiered AGENT_MODELS with env-var overrides (W5)
├── tests/
│   ├── test_smoke.py         # W1 end-to-end (LLM)
│   ├── test_multi_agent.py   # W2 PM + routers (mocked)
│   ├── test_week3.py         # W3 QA + A2A + MCP adapter (mocked)
│   ├── test_week4.py         # W4 observability + resilience + cost + cache (mocked)
│   └── test_week5.py         # W5 self-reflect + debate + consolidation + cost write (mocked)
├── docs/
│   ├── ARCHITECTURE.md       # Canonical architecture (W5)
│   ├── REFLECTION.md         # Post-mortem (W5)
│   ├── test_results.md       # E2E demo runs + cost-optimisation data (W5)
│   ├── cost_reports/         # Per-run JSON cost reports (W5, auto-generated)
│   ├── api/                  # pdoc HTML reference (W5)
│   └── architecture.md       # Chronological week-by-week record
├── .env.example
├── Dockerfile                # Non-root user, healthcheck
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

### LLM: tiered models via Helicone

All LLM calls are routed through Helicone for usage tracking. As of Week 5
the defaults are tiered — `gpt-4o` for the PM (where ambiguity cascades) and
`gpt-4o-mini` for the Coder and QA (where inputs and outputs are already
structured). Override per-agent via `MODEL_PM`, `MODEL_CODER`, `MODEL_QA`
env vars. See `docs/test_results.md` for the measured cost-quality
trade-off.

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
| 1 | Coder Agent v1.0 | Memory, tools, ReACT loop, Pydantic output |
| 2 | Multi-Agent v2.0 | PM agent, LangGraph handoff, shared state |
| 3 | Complete Team v3.0 | QA & Debugger agent, MCP / A2A protocols, feedback loop |
| 4 | Production v4.0 | Observability, cost tracking, error recovery, Docker deploy |
| 5 | Polished Submission v5.0 | Self-reflect + max-8-task PM consolidation + tiered models + cost reports + persistent Chroma + PM debate / consensus + full docs + demo — **current** |

## Course

- **Course:** Build Autonomous Multi-Agent Systems (Saras AI Institute)
- **Instructor:** Anshuman Singh
- **Student:** Aaron Major (GitHub: [@aaronnmajor](https://github.com/aaronnmajor))
- **Term:** Spring 2026
