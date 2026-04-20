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

from config import get_model
from observability import get_logger, trace_span
from orchestration.state import CodingTask, ProjectState

load_dotenv(override=True)

MODEL = get_model("pm")
_log = get_logger("pm")


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
Aim for 2-5 tasks unless the spec is genuinely larger.
"""


def build_tech_spec(requirement: str, client: OpenAI | None = None) -> str:
    """Convert a raw requirement into a Markdown tech spec."""
    client = client or _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SPEC_PROMPT},
            {"role": "user", "content": f"Requirement:\n\n{requirement}"},
        ],
    )
    return response.choices[0].message.content or ""


def decompose_into_tasks(tech_spec: str, client: OpenAI | None = None) -> list[CodingTask]:
    """Convert a tech spec into a list of coding tasks."""
    client = client or _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TASKS_PROMPT},
            {"role": "user", "content": f"Tech spec:\n\n{tech_spec}"},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    # Strip accidental markdown fences.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(raw)
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
    except json.JSONDecodeError:
        return []


def pm_node(state: ProjectState) -> dict[str, Any]:
    """LangGraph node: run the PM agent on the current state."""
    requirement = state.get("user_requirement", "").strip()
    run_id = state.get("run_id", "")

    if not requirement:
        _log.error("pm_missing_requirement", run_id=run_id)
        return {"routing": "error", "error": "Missing user_requirement in state"}

    with trace_span("pm", "pm_node", run_id):
        client = _client()
        with trace_span("pm", "build_tech_spec", run_id):
            spec = build_tech_spec(requirement, client=client)
        with trace_span("pm", "decompose_into_tasks", run_id):
            tasks = decompose_into_tasks(spec, client=client)

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
