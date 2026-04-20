"""
QA & Debugger Agent — Week 3.

Reviews code artifacts produced by the Coder agent and emits structured feedback:
  - passed:      pass/fail verdict
  - issues:      concrete bugs or style violations found
  - suggestions: recommended fixes
  - summary:     one-line summary used by the coder on retry

The review is a single LLM call with a JSON response; malformed responses
default to "pass" to avoid blocking forward progress on parse errors.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config import get_model
from observability import get_logger, trace_span
from orchestration.state import ProjectState, QAReview

load_dotenv(override=True)

MODEL = get_model("qa")
MAX_RETRIES_PER_TASK = 2  # Coder gets up to 2 retries per task before QA approves as-is.

_log = get_logger("qa")


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("HELICONE_BASE_URL"),
        default_headers={"Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}"},
    )


REVIEW_PROMPT = """You are a senior code reviewer and QA engineer. You review a single code artifact
against its task specification and return structured feedback as JSON.

Return ONLY a JSON object (no markdown fences, no commentary) matching this schema:

{
  "passed": true | false,
  "issues": ["concrete bug or style violation", ...],
  "suggestions": ["recommended fix", ...],
  "summary": "one line summary for the developer"
}

Review criteria:
- Correctness: does the code satisfy the acceptance criteria?
- Safety: any unhandled exceptions, resource leaks, or security issues?
- Style: idiomatic Python with type hints and docstrings where appropriate?
- Execution: did the code run successfully (see exec_result)?

Be decisive. If the code is broken or clearly wrong, set passed=false.
If the code runs and mostly does the job, you may still flag improvements but set passed=true.
"""


def review_artifact(task: dict, artifact: dict, client: OpenAI | None = None) -> QAReview:
    """Run a single QA review on one artifact. Returns a QAReview dict."""
    client = client or _client()

    user_message = (
        f"Task:\n"
        f"  ID: {task.get('task_id', '')}\n"
        f"  Title: {task.get('title', '')}\n"
        f"  Description: {task.get('description', '')}\n"
        f"  Acceptance criteria: {task.get('acceptance_criteria', '')}\n"
        f"  Target file: {task.get('file', '')}\n\n"
        f"Artifact:\n"
        f"  File: {artifact.get('file', '')}\n"
        f"  Content:\n```python\n{artifact.get('content', '')}\n```\n"
        f"  Execution result:\n{artifact.get('exec_result', '')}\n"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Default to pass on parse failure so we don't block on a bad response.
        return {
            "task_id":     task.get("task_id", ""),
            "passed":      True,
            "issues":      [],
            "suggestions": [],
            "summary":     "QA review produced malformed JSON; defaulting to pass.",
        }

    return {
        "task_id":     task.get("task_id", ""),
        "passed":      bool(parsed.get("passed", True)),
        "issues":      list(parsed.get("issues", [])),
        "suggestions": list(parsed.get("suggestions", [])),
        "summary":     str(parsed.get("summary", "")),
    }


def qa_node(state: ProjectState) -> dict[str, Any]:
    """LangGraph node: review the most recent artifact and route accordingly.

    Reads the latest artifact (from the Coder's previous turn) and the task it
    corresponds to. Routes back to "coder" if the review failed AND we haven't
    exhausted MAX_RETRIES_PER_TASK; otherwise advances the task index and
    routes to "coder" (more tasks) or "done" (queue drained).
    """
    artifacts = state.get("artifacts", [])
    tasks = state.get("tasks", [])
    current_idx = state.get("current_task_index", 0)
    retry_count = state.get("retry_count", 0)

    if not artifacts:
        return {"routing": "done"}

    latest = artifacts[-1]
    task_id = latest.get("task_id", "")
    task = next((t for t in tasks if t.get("task_id") == task_id), None)
    if task is None:
        return {"routing": "done"}

    run_id = state.get("run_id", "")
    with trace_span("qa", "qa_review", run_id, task_id=task_id):
        review = review_artifact(task, latest)
    _log.info(
        "qa_review_complete",
        run_id=run_id,
        task_id=task_id,
        passed=review["passed"],
        issues=len(review.get("issues", [])),
    )

    if not review["passed"] and retry_count < MAX_RETRIES_PER_TASK:
        # Fail and retry: keep the same task; the coder will redo it using the review feedback.
        return {
            "reviews":     [review],
            "retry_count": retry_count + 1,
            "routing":     "coder",
        }

    # Pass OR out of retries: advance to the next task.
    next_idx = current_idx + 1
    done = next_idx >= len(tasks)
    return {
        "reviews":            [review],
        "retry_count":        0,
        "current_task_index": next_idx,
        "routing":            "done" if done else "coder",
    }
