# Demo Recording Guide

A 4-minute walkthrough script for the Saras AI capstone submission. Aim
for 3–5 minutes total. Don't go over five.

## Setup before recording

1. **Pin a clean terminal** with the repo's working directory: `cd C:\GitHub\aaronnmajor\saras-multi-agent\multi-agent-dev-team`
2. **Check `.env` is filled in** (`cat .env` quickly to confirm — but do not show it on camera).
3. **Start Docker Desktop** if you plan to demo `docker compose up`.
4. **Clear old workspace files** if you want a clean run: `rm -rf workspace/`.

## Suggested outline

### 0:00 – 0:30 — Introduction

> "Hi, I'm Aaron Major. This is my final submission for *Build Autonomous
> Multi-Agent Systems* at Saras AI Institute. The project is an autonomous
> three-agent software development team — a PM, a Coder, and a QA — that
> takes a single requirement and produces verified Python code. It's
> built on LangGraph, runs through Helicone for cost tracking, and ships
> in Docker."

### 0:30 – 1:30 — Architecture walkthrough

Open `docs/ARCHITECTURE.md` and walk through:

- **The pipeline diagram** (Mermaid flowchart): START → PM → Coder → QA → END, with the QA-then-Coder retry loop.
- **The three agent roles** at a glance.
- **Cross-cutting concerns** table (logging, tracing, cost, retry, breaker, timeout, cache, memory, MCP, A2A).

> "The key design choice: every list field in the shared state uses an
> `add` reducer, which is what makes the QA-then-Coder retry loop safe.
> Without it, each iteration would clobber the previous artifact."

### 1:30 – 3:00 — Live end-to-end run

Run the BST grading task:

```bash
python -m orchestration.graph "Build a Python module that implements a binary search tree with insert, search, and in-order traversal methods, with full test coverage."
```

While it runs (~3 min), narrate:

- "The PM is producing a tech spec — five sections, exactly the format the rubric calls for."
- "Now the Coder is working through tasks one at a time. You can see the JSON-structured logs with `run_id` propagating through every span."
- "And there's the self-reflection step firing — the Coder is critiquing its own code before the QA agent sees it."
- "Now QA is reviewing the artifact against a rubric — correctness, safety, style, execution."

Once it finishes, show the cost report:

```bash
cat docs/cost_reports/<run_id>.json
```

> "Two cents and ~80,000 tokens to ship a working binary search tree
> module with full test coverage."

### 3:00 – 3:45 — Innovations

Open `docs/test_results.md` and show the cost-comparison table:

> "Innovation #1 — same task, two model tiers. All-on-`gpt-4o` costs
> 26 cents. Tuned with the PM on `gpt-4o` and Coder/QA on `gpt-4o-mini`
> costs 2.7 cents. Same first-pass success rate. That's a 90 % cost
> reduction with no quality hit."

Show `agents/pm_debate.py`:

> "Innovation #2 — when ambiguous requirements come in, you can flip
> `PM_DEBATE_MODE=true` and the PM phase runs three calls: a simplicity
> advocate, a completeness advocate, and a synthesiser that arbitrates
> between them and emits a single spec with inline justification notes."

### 3:45 – 4:00 — Wrap

> "Tests pass at 95/95 across mocked and live suites. `docker compose
> up` brings up Chroma plus the agent with health checks. Full
> documentation is in `docs/`. Thanks for grading."

## What to capture in the recording

- ✅ The architecture diagram (`docs/ARCHITECTURE.md` rendered)
- ✅ A live end-to-end pipeline run with logs visible
- ✅ The final cost report JSON
- ✅ Both innovations demonstrated (cost comparison table + `pm_debate.py`)
- ✅ Test suite green (`pytest -m "not llm" -q`)

## Optional: Docker + LangSmith

If you want to demo Docker and LangSmith:

```bash
# Start Chroma + agent with healthchecks (~20 sec to settle)
docker compose up --build

# In a separate terminal, point a browser at https://smith.langchain.com
# and show the run as a single tree of spans
```

Skip if it adds time pressure — the local Python run is sufficient for the rubric.

## Posting

Upload as a YouTube *Unlisted* video or Loom. Paste the URL into the
README's `Demo recording` section, replacing the `(#)` placeholder.

---

*Once the demo is recorded and linked, you're ready for the LMS submission.*
