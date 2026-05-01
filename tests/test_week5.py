"""
Tests for Week 5 polish:
  - PM consolidation pass when task list exceeds the cap
  - PM debate / consensus layer (advocate-A + advocate-B + synthesiser)
  - Coder self-reflection (revision applied, critique appended)
  - CostTracker write_report flushes JSON keyed by run_id
All LLM calls are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.pm_agent import (
    MAX_TASKS_PER_REQUIREMENT,
    consolidate_tasks,
    pm_node,
)
from agents.pm_debate import propose_spec_with_debate
from observability.cost import (
    _TRACKERS,
    TokenUsage,
    record_usage_from_response,
    tracker_for,
    write_report,
)
from orchestration.state import ProjectState

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_completion(content: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    if prompt_tokens or completion_tokens:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.prompt_tokens_details = None
        completion.usage = usage
    else:
        completion.usage = None
    return completion


def _mock_client(responses: list) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_completion(*r) if isinstance(r, tuple) else _mock_completion(r)
        for r in responses
    ]
    return client


# ─────────────────────────────────────────────────────────────────────────────
# PM consolidation
# ─────────────────────────────────────────────────────────────────────────────

class TestConsolidateTasks:
    NINE_TASKS = [
        {"task_id": f"T{i:03d}", "title": f"t{i}", "description": f"d{i}",
         "acceptance_criteria": "ac", "status": "pending", "file": f"f{i}.py"}
        for i in range(1, 10)
    ]

    CONSOLIDATED_JSON = (
        '[{"task_id": "T001", "title": "merged-1", "description": "d", '
        '"acceptance_criteria": "ac", "status": "pending", "file": "a.py"},'
        ' {"task_id": "T002", "title": "merged-2", "description": "d", '
        '"acceptance_criteria": "ac", "status": "pending", "file": "b.py"}]'
    )

    def test_returns_shorter_list_when_llm_complies(self):
        client = _mock_client([self.CONSOLIDATED_JSON])
        out = consolidate_tasks(self.NINE_TASKS, client=client)
        assert len(out) == 2
        assert out[0]["title"] == "merged-1"

    def test_falls_back_to_original_on_invalid_json(self):
        client = _mock_client(["not json"])
        out = consolidate_tasks(self.NINE_TASKS, client=client)
        assert out == self.NINE_TASKS  # unchanged

    def test_falls_back_when_consolidation_is_not_actually_shorter(self):
        # LLM "consolidates" to the same length — should reject.
        same_len = json.dumps(self.NINE_TASKS)
        client = _mock_client([same_len])
        out = consolidate_tasks(self.NINE_TASKS, client=client)
        assert out == self.NINE_TASKS

    def test_pm_node_invokes_consolidation_on_overgrown_list(self):
        many_tasks_json = json.dumps([
            {"task_id": f"T{i:03d}", "title": f"t{i}", "description": "d",
             "acceptance_criteria": "ac", "status": "pending", "file": f"f{i}.py"}
            for i in range(1, MAX_TASKS_PER_REQUIREMENT + 3)
        ])
        responses = [
            "# Overview\nspec",  # build_tech_spec
            many_tasks_json,     # decompose_into_tasks
            self.CONSOLIDATED_JSON,  # consolidate_tasks
        ]
        with patch("agents.pm_agent._client", return_value=_mock_client(responses)):
            state: ProjectState = {
                "user_requirement": "build something big",
                "tech_spec": "", "tasks": [], "artifacts": [], "reviews": [],
                "current_task_index": 0, "retry_count": 0,
                "routing": "", "error": "", "run_id": "test-run",
            }
            result = pm_node(state)
        assert len(result["tasks"]) <= MAX_TASKS_PER_REQUIREMENT
        assert result["routing"] == "coder"


# ─────────────────────────────────────────────────────────────────────────────
# PM debate / consensus
# ─────────────────────────────────────────────────────────────────────────────

class TestPmDebate:
    DRAFT_A = "# Overview\nLean spec\n"
    DRAFT_B = "# Overview\nThorough spec\n"
    FINAL  = "# Overview\nSynthesised spec\n## Synthesis notes\n- chose A's overview\n"

    def test_three_calls_produce_synthesised_spec(self):
        client = _mock_client([self.DRAFT_A, self.DRAFT_B, self.FINAL])
        result = propose_spec_with_debate(
            "build a widget", model="gpt-4o", client=client, run_id="r1",
        )
        assert "Synthesised" in result["spec"]
        assert result["draft_a"] == self.DRAFT_A.strip()
        assert result["draft_b"] == self.DRAFT_B.strip()
        assert client.chat.completions.create.call_count == 3

    def test_falls_back_to_surviving_draft_when_one_is_empty(self):
        client = _mock_client(["", self.DRAFT_B])  # only B succeeds
        result = propose_spec_with_debate(
            "build a widget", model="gpt-4o", client=client, run_id="r1",
        )
        assert result["spec"] == self.DRAFT_B.strip()
        # Synthesiser was skipped → only 2 calls.
        assert client.chat.completions.create.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Cost tracker registry + write_report
# ─────────────────────────────────────────────────────────────────────────────

class TestCostTrackerRegistry:
    def setup_method(self):
        _TRACKERS.clear()

    def test_tracker_for_returns_same_instance(self):
        a = tracker_for("run-1")
        b = tracker_for("run-1")
        assert a is b

    def test_record_usage_extracts_openai_usage(self):
        response = _mock_completion("ok", prompt_tokens=100, completion_tokens=50)
        record_usage_from_response("run-2", "pm", "gpt-4o-mini", response)
        report = tracker_for("run-2").report()
        assert report["by_agent"]["pm"]["prompt"] == 100
        assert report["by_agent"]["pm"]["completion"] == 50

    def test_record_usage_noop_on_empty_run_id(self):
        response = _mock_completion("ok", prompt_tokens=100, completion_tokens=50)
        record_usage_from_response("", "pm", "gpt-4o-mini", response)
        # No tracker should have been created.
        assert "" not in _TRACKERS

    def test_write_report_creates_file_and_clears_registry(self, tmp_path: Path):
        tracker_for("run-3").record(
            TokenUsage(agent="pm", model="gpt-4o-mini", prompt=10, completion=5)
        )
        path = write_report("run-3", tmp_path)
        assert path is not None
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "run-3"
        assert data["total_tokens"] == 15
        # Tracker is removed from the registry after writing.
        assert "run-3" not in _TRACKERS

    def test_write_report_returns_none_when_no_tracker(self, tmp_path: Path):
        assert write_report("never-existed", tmp_path) is None


# ─────────────────────────────────────────────────────────────────────────────
# Coder self-reflection
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfReflect:
    def test_no_revision_path(self):
        from agents.coder_agent import self_reflect
        client = _mock_client([
            '{"needs_revision": false, "critique": "looks good", "revised_code": null}'
        ])
        out = self_reflect("task", "print(1)", "1", client=client)
        assert out["needs_revision"] is False
        assert out["revised_code"] is None
        assert out["critique"] == "looks good"

    def test_revision_path_returns_revised_code(self):
        from agents.coder_agent import self_reflect
        client = _mock_client([
            '{"needs_revision": true, "critique": "needs docstring", '
            '"revised_code": "def f():\\n    \\"\\"\\"d.\\"\\"\\"\\n    return 1"}'
        ])
        out = self_reflect("task", "def f(): return 1", "", client=client)
        assert out["needs_revision"] is True
        assert "docstring" in out["critique"]
        assert "def f()" in out["revised_code"]

    def test_malformed_json_degrades_to_no_revision(self):
        from agents.coder_agent import self_reflect
        client = _mock_client(["not json at all"])
        out = self_reflect("task", "print(1)", "1", client=client)
        assert out["needs_revision"] is False
        assert out["revised_code"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Workspace verification (post-pipeline pytest run)
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyWorkspace:
    def test_no_tests_status_when_workspace_empty(self, tmp_path: Path):
        from orchestration.verify import verify_workspace
        result = verify_workspace(tmp_path)
        assert result["status"] == "no_tests"
        assert result["passed"] == 0
        assert result["failed"] == 0

    def test_pass_status_when_all_tests_pass(self, tmp_path: Path):
        from orchestration.verify import verify_workspace
        (tmp_path / "test_passes.py").write_text(
            "def test_one():\n    assert 1 + 1 == 2\n",
            encoding="utf-8",
        )
        result = verify_workspace(tmp_path)
        assert result["status"] == "pass"
        assert result["passed"] == 1
        assert result["failed"] == 0

    def test_fail_status_when_any_test_fails(self, tmp_path: Path):
        from orchestration.verify import verify_workspace
        (tmp_path / "test_fails.py").write_text(
            "def test_bad():\n    assert False\n",
            encoding="utf-8",
        )
        result = verify_workspace(tmp_path)
        assert result["status"] == "fail"
        assert result["failed"] == 1

    def test_format_verification_renders_status(self):
        from orchestration.verify import format_verification
        line = format_verification({
            "status": "pass", "passed": 3, "failed": 0, "errors": 0, "skipped": 0,
        })
        assert "PASS" in line
        assert "passed=3" in line


# ─────────────────────────────────────────────────────────────────────────────
# LangSmith tracing wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestLangSmithWiring:
    def test_configure_langsmith_returns_false_without_key(self, monkeypatch):
        from observability.tracing import configure_langsmith
        for var in (
            "LANGCHAIN_API_KEY",
            "LANGSMITH_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_PROJECT",
        ):
            monkeypatch.delenv(var, raising=False)
        assert configure_langsmith() is False

    def test_configure_langsmith_promotes_langsmith_aliases(self, monkeypatch):
        from observability.tracing import configure_langsmith
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test")
        monkeypatch.setenv("LANGSMITH_PROJECT", "demo")
        assert configure_langsmith() is True
        import os
        assert os.environ["LANGCHAIN_API_KEY"] == "ls__test"
        assert os.environ["LANGCHAIN_PROJECT"] == "demo"

    def test_traceable_decorator_is_callable_when_langsmith_absent(self):
        from observability.tracing import traceable

        @traceable(name="x")
        def f(state: dict) -> dict:
            return {"ok": True}

        assert f({}) == {"ok": True}
