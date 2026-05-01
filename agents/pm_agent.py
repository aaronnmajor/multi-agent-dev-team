"""
Product Manager Agent — Week 2.

Takes a raw user requirement and produces:
  1. A markdown technical specification.
  2. A list of coding tasks (CodingTask dicts) for the Coder Agent.

Writes both to the shared ProjectState and signals the next routing step.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agents.pm_debate import is_enabled as _debate_enabled
from agents.pm_debate import propose_spec_with_debate
from config import get_model
from observability import get_logger, record_usage_from_response, trace_span
from orchestration.state import CodingTask, ProjectState

load_dotenv(override=True)

MODEL = get_model("pm")
_log = get_logger("pm")

# Hard ceiling on the task list. If the PM emits more than this, a
# consolidation pass is run to merge closely-related items. Prevents the
# pipeline from devolving into a long fan-out for sprawling specs.
MAX_TASKS_PER_REQUIREMENT = 8


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("HELICONE_BASE_URL"),
        default_headers={"Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}"},
    )


SPEC_PROMPT = """You are a Product Manager. Convert the user's raw requirement into a technical specification.

Produce a Markdown document with exactly these five sections:

# Overview
A one-paragraph summary of what is being built and why.

# Functional Requirements
Bullet list of concrete behaviors the system must support.

# Non-Functional Requirements
Bullet list of performance, quality, and safety requirements.

# File Structure
A code block showing the files/directories to create (relative paths only).

# Constraints and Assumptions
Bullet list of constraints and assumptions you are making.

Keep the spec focused and implementable — this is for a single-developer project, not enterprise scope.
"""


TASKS_PROMPT = """You are a Product Manager decomposing a technical specification into coding tasks.

Output a JSON array (and nothing else — no markdown fences, no commentary) of coding task objects.
Each object must have exactly these fields:

{
  "task_id": "T001",
  "title": "short title",
  "description": "what the coder must implement",
  "acceptance_criteria": "how to verify it is done",
  "status": "pending",
  "file": "relative filename the coder should create"
}

Task IDs are sequential (T001, T002, ...). Tasks should be ordered so that each task only depends on prior tasks.
Aim for 2-5 tasks unless the spec is genuinely larger. Never produce more than 8 tasks.
"""


CONSOLIDATE_PROMPT = """You are a Product Manager consolidating an overgrown task list.

The previous decomposition produced too many tasks. Merge closely-related items so the final list has 8 or fewer tasks while preserving every acceptance criterion. Combine adjacent tasks that touch the same file or that a single coder turn could complete together.

Output a JSON array of coding task objects with the SAME schema as before:

{
  "task_id": "T001",
  "title": "short title",
  "description": "what the coder must implement",
  "acceptance_criteria": "how to verify it is done",
  "status": "pending",
  "file": "relative filename the coder should create"
}

Renumber task_ids sequentially from T001. Output ONLY the JSON array — no fences, no commentary.
"""


def build_tech_spec(
    requirement: str,
    client: OpenAI | None = None,
    run_id: str = "",
) -> str:
    """Convert a raw requirement into a Markdown tech spec."""
    client = client or _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SPEC_PROMPT},
            {"role": "user", "content": f"Requirement:\n\n{requirement}"},
        ],
    )
    record_usage_from_response(run_id, "pm", MODEL, response)
    return response.choices[0].message.content or ""


def _parse_task_list(raw: str) -> list[CodingTask]:
    """Parse a JSON array of CodingTask dicts, tolerating markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    tasks: list[CodingTask] = []
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        tasks.append({
            "task_id":             item.get("task_id", f"T{i:03d}"),
            "title":               item.get("title", ""),
            "description":         item.get("description", ""),
            "acceptance_criteria": item.get("acceptance_criteria", ""),
            "status":              item.get("status", "pending"),
            "file":                item.get("file", ""),
        })
    return tasks


def decompose_into_tasks(
    tech_spec: str,
    client: OpenAI | None = None,
    run_id: str = "",
) -> list[CodingTask]:
    """Convert a tech spec into a list of coding tasks."""
    client = client or _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TASKS_PROMPT},
            {"role": "user", "content": f"Tech spec:\n\n{tech_spec}"},
        ],
    )
    record_usage_from_response(run_id, "pm", MODEL, response)
    raw = response.choices[0].message.content or ""
    return _parse_task_list(raw)


def consolidate_tasks(
    tasks: list[CodingTask],
    client: OpenAI | None = None,
    run_id: str = "",
) -> list[CodingTask]:
    """Compress an overgrown task list down to MAX_TASKS_PER_REQUIREMENT items.

    If the PM's first decomposition exceeds the cap, this runs a second LLM
    pass with a consolidation prompt. The original list is returned unchanged
    when the LLM fails to produce a valid (and shorter) list, so a malformed
    consolidation never silently drops tasks.
    """
    client = client or _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CONSOLIDATE_PROMPT},
            {"role": "user", "content": f"Original task list (JSON):\n{json.dumps(tasks, indent=2)}"},
        ],
    )
    record_usage_from_response(run_id, "pm", MODEL, response)
    raw = response.choices[0].message.content or ""
    consolidated = _parse_task_list(raw)
    if not consolidated or len(consolidated) >= len(tasks):
        return tasks
    return consolidated


def pm_node(state: ProjectState) -> dict[str, Any]:
    """LangGraph node: run the PM agent on the current state."""
    requirement = state.get("user_requirement", "").strip()
    run_id = state.get("run_id", "")

    if not requirement:
        _log.error("pm_missing_requirement", run_id=run_id)
        return {"routing": "error", "error": "Missing user_requirement in state"}

    with trace_span("pm", "pm_node", run_id):
        client = _client()
        if _debate_enabled():
            with trace_span("pm", "spec_debate", run_id):
                debate = propose_spec_with_debate(
                    requirement, model=MODEL, client=client, run_id=run_id,
                )
            spec = debate["spec"]
        else:
            with trace_span("pm", "build_tech_spec", run_id):
                spec = build_tech_spec(requirement, client=client, run_id=run_id)
        with trace_span("pm", "decompose_into_tasks", run_id):
            tasks = decompose_into_tasks(spec, client=client, run_id=run_id)
        if len(tasks) > MAX_TASKS_PER_REQUIREMENT:
            with trace_span("pm", "consolidate_tasks", run_id, original_count=len(tasks)):
                tasks = consolidate_tasks(tasks, client=client, run_id=run_id)
            _log.info("pm_consolidated", run_id=run_id, final_count=len(tasks))

    if not tasks:
        _log.error("pm_zero_tasks", run_id=run_id)
        return {
            "tech_spec": spec,
            "routing": "error",
            "error": "PM agent produced zero coding tasks from the spec",
        }

    _log.info("pm_complete", run_id=run_id, task_count=len(tasks))
    return {
        "tech_spec": spec,
        "tasks": tasks,
        "current_task_index": 0,
        "routing": "coder",
    }
