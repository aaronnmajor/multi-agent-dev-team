"""
Tests for the Week 2 multi-agent system (PM + Coder).

Most tests mock the LLM so they're fast and free. One integration test
runs the full graph with real API calls; skip it with `pytest -m "not llm"`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.pm_agent import build_tech_spec, decompose_into_tasks, pm_node
from orchestration.graph import build_graph, route_after_coder, route_after_pm
from orchestration.state import CodingTask, ProjectState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_completion(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _mock_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = [_mock_completion(r) for r in responses]
    return client


# ─────────────────────────────────────────────────────────────────────────────
# build_tech_spec
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTechSpec:
    def test_returns_string(self):
        client = _mock_client(["# Overview\nTest spec\n"])
        result = build_tech_spec("do a thing", client=client)
        assert isinstance(result, str)
        assert "Overview" in result

    def test_includes_requirement_in_prompt(self):
        client = _mock_client(["spec"])
        build_tech_spec("build a widget", client=client)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert any("build a widget" in m["content"] for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# decompose_into_tasks
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeIntoTasks:
    VALID_JSON = '''[
      {"task_id": "T001", "title": "First", "description": "Do first",
       "acceptance_criteria": "works", "status": "pending", "file": "a.py"},
      {"task_id": "T002", "title": "Second", "description": "Do second",
       "acceptance_criteria": "also works", "status": "pending", "file": "b.py"}
    ]'''

    def test_parses_valid_json(self):
        client = _mock_client([self.VALID_JSON])
        tasks = decompose_into_tasks("spec", client=client)
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == "T001"
        assert tasks[1]["file"] == "b.py"

    def test_strips_markdown_fences(self):
        wrapped = "```json\n" + self.VALID_JSON + "\n```"
        client = _mock_client([wrapped])
        tasks = decompose_into_tasks("spec", client=client)
        assert len(tasks) == 2

    def test_returns_empty_on_invalid_json(self):
        client = _mock_client(["not json at all"])
        tasks = decompose_into_tasks("spec", client=client)
        assert tasks == []

    def test_returns_empty_on_non_list_json(self):
        client = _mock_client(['{"not": "a list"}'])
        tasks = decompose_into_tasks("spec", client=client)
        assert tasks == []

    def test_fills_missing_fields_with_defaults(self):
        client = _mock_client(['[{"title": "partial"}]'])
        tasks = decompose_into_tasks("spec", client=client)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "T001"
        assert tasks[0]["status"] == "pending"


# ─────────────────────────────────────────────────────────────────────────────
# pm_node
# ─────────────────────────────────────────────────────────────────────────────

class TestPmNode:
    def test_empty_requirement_sets_error_routing(self):
        state: ProjectState = {
            "user_requirement": "",
            "tech_spec": "", "tasks": [], "artifacts": [],
            "current_task_index": 0, "routing": "", "error": "",
        }
        result = pm_node(state)
        assert result["routing"] == "error"
        assert "user_requirement" in result["error"]

    def test_happy_path_routes_to_coder(self):
        fake_responses = [
            "# Overview\nTest spec content",
            '[{"task_id": "T001", "title": "t1", "description": "d1", '
            '"acceptance_criteria": "ac1", "status": "pending", "file": "t1.py"}]',
        ]
        with patch("agents.pm_agent._client", return_value=_mock_client(fake_responses)):
            state: ProjectState = {
                "user_requirement": "build a widget",
                "tech_spec": "", "tasks": [], "artifacts": [],
                "current_task_index": 0, "routing": "", "error": "",
            }
            result = pm_node(state)
        assert result["routing"] == "coder"
        assert "Test spec" in result["tech_spec"]
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["file"] == "t1.py"
        assert result["current_task_index"] == 0

    def test_zero_tasks_sets_error_routing(self):
        fake_responses = ["# Overview\nSpec", "[]"]
        with patch("agents.pm_agent._client", return_value=_mock_client(fake_responses)):
            state: ProjectState = {
                "user_requirement": "build a widget",
                "tech_spec": "", "tasks": [], "artifacts": [],
                "current_task_index": 0, "routing": "", "error": "",
            }
            result = pm_node(state)
        assert result["routing"] == "error"
        assert "zero" in result["error"].lower() or "no" in result["error"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterPm:
    def test_coder_routing_goes_to_coder(self):
        state = {"routing": "coder"}
        assert route_after_pm(state) == "coder"  # type: ignore[arg-type]

    def test_error_routing_goes_to_end(self):
        from langgraph.graph import END
        state = {"routing": "error"}
        assert route_after_pm(state) == END  # type: ignore[arg-type]

    def test_missing_routing_goes_to_end(self):
        from langgraph.graph import END
        state = {}
        assert route_after_pm(state) == END  # type: ignore[arg-type]


class TestRouteAfterCoder:
    """In the Week 3 graph, after the coder produces an artifact we route to QA,
    not back to the coder. The QA node is what decides whether to loop or advance."""

    def test_qa_routing_goes_to_qa(self):
        state = {"routing": "qa"}
        assert route_after_coder(state) == "qa"  # type: ignore[arg-type]

    def test_done_routing_goes_to_end(self):
        from langgraph.graph import END
        state = {"routing": "done"}
        assert route_after_coder(state) == END  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildGraph:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "pm" in node_names
        assert "coder" in node_names
