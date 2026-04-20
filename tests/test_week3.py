"""
Tests for Week 3: QA agent, A2A protocol primitives, MCP adapter, graph routing.
All LLM calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.qa_agent import MAX_RETRIES_PER_TASK, qa_node, review_artifact
from orchestration.a2a import (
    AGENT_CAPABILITIES,
    Broker,
    Message,
    validate_incoming,
)
from orchestration.graph import (
    build_graph,
    route_after_coder,
    route_after_pm,
    route_after_qa,
)
from orchestration.state import ProjectState
from tools.mcp_adapter import TOOL_REGISTRY, call_tool, list_tools


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


def _base_state(**overrides) -> ProjectState:
    state: ProjectState = {
        "user_requirement":   "",
        "tech_spec":          "",
        "tasks":              [],
        "artifacts":          [],
        "reviews":            [],
        "current_task_index": 0,
        "retry_count":        0,
        "routing":            "",
        "error":              "",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ─────────────────────────────────────────────────────────────────────────────
# review_artifact
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewArtifact:
    TASK = {
        "task_id": "T001", "title": "t1", "description": "d1",
        "acceptance_criteria": "ac1", "status": "pending", "file": "t1.py",
    }
    ARTIFACT = {"task_id": "T001", "file": "t1.py", "content": "print('hi')", "exec_result": "hi"}

    def test_parses_passed_review(self):
        client = _mock_client(['{"passed": true, "issues": [], "suggestions": [], "summary": "looks good"}'])
        review = review_artifact(self.TASK, self.ARTIFACT, client=client)
        assert review["passed"] is True
        assert review["task_id"] == "T001"
        assert review["summary"] == "looks good"

    def test_parses_failed_review_with_issues(self):
        client = _mock_client([
            '{"passed": false, "issues": ["missing docstring"], "suggestions": ["add docstring"], '
            '"summary": "needs work"}'
        ])
        review = review_artifact(self.TASK, self.ARTIFACT, client=client)
        assert review["passed"] is False
        assert "missing docstring" in review["issues"]
        assert "add docstring" in review["suggestions"]

    def test_strips_markdown_fences(self):
        wrapped = '```json\n{"passed": true, "issues": [], "suggestions": [], "summary": "ok"}\n```'
        client = _mock_client([wrapped])
        review = review_artifact(self.TASK, self.ARTIFACT, client=client)
        assert review["passed"] is True

    def test_malformed_json_defaults_to_pass(self):
        client = _mock_client(["not valid json"])
        review = review_artifact(self.TASK, self.ARTIFACT, client=client)
        assert review["passed"] is True
        assert "malformed" in review["summary"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# qa_node routing
# ─────────────────────────────────────────────────────────────────────────────

class TestQaNode:
    TASKS = [
        {"task_id": "T001", "title": "t1", "description": "d1",
         "acceptance_criteria": "ac1", "status": "pending", "file": "t1.py"},
        {"task_id": "T002", "title": "t2", "description": "d2",
         "acceptance_criteria": "ac2", "status": "pending", "file": "t2.py"},
    ]

    def _patched(self, review, state):
        with patch("agents.qa_agent.review_artifact", return_value=review):
            return qa_node(state)

    def test_no_artifacts_routes_done(self):
        result = qa_node(_base_state(tasks=self.TASKS))
        assert result["routing"] == "done"

    def test_pass_advances_to_next_task(self):
        state = _base_state(
            tasks=self.TASKS,
            artifacts=[{"task_id": "T001", "file": "t1.py", "content": "", "exec_result": ""}],
            current_task_index=0,
        )
        review = {"task_id": "T001", "passed": True, "issues": [], "suggestions": [], "summary": "ok"}
        result = self._patched(review, state)
        assert result["routing"] == "coder"
        assert result["current_task_index"] == 1
        assert result["retry_count"] == 0

    def test_pass_on_last_task_routes_done(self):
        state = _base_state(
            tasks=self.TASKS,
            artifacts=[{"task_id": "T002", "file": "t2.py", "content": "", "exec_result": ""}],
            current_task_index=1,
        )
        review = {"task_id": "T002", "passed": True, "issues": [], "suggestions": [], "summary": "ok"}
        result = self._patched(review, state)
        assert result["routing"] == "done"
        assert result["current_task_index"] == 2

    def test_fail_with_retries_available_routes_coder_same_task(self):
        state = _base_state(
            tasks=self.TASKS,
            artifacts=[{"task_id": "T001", "file": "t1.py", "content": "", "exec_result": ""}],
            current_task_index=0,
            retry_count=0,
        )
        review = {"task_id": "T001", "passed": False, "issues": ["bug"], "suggestions": [], "summary": "fix"}
        result = self._patched(review, state)
        assert result["routing"] == "coder"
        assert result["retry_count"] == 1
        assert "current_task_index" not in result  # same task

    def test_fail_at_retry_limit_advances(self):
        state = _base_state(
            tasks=self.TASKS,
            artifacts=[{"task_id": "T001", "file": "t1.py", "content": "", "exec_result": ""}],
            current_task_index=0,
            retry_count=MAX_RETRIES_PER_TASK,
        )
        review = {"task_id": "T001", "passed": False, "issues": ["bug"], "suggestions": [], "summary": "fix"}
        result = self._patched(review, state)
        assert result["routing"] == "coder"
        assert result["current_task_index"] == 1
        assert result["retry_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Graph routers
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterCoder:
    def test_qa_routing_goes_to_qa(self):
        assert route_after_coder({"routing": "qa"}) == "qa"  # type: ignore[arg-type]

    def test_done_routing_goes_to_end(self):
        from langgraph.graph import END
        assert route_after_coder({"routing": "done"}) == END  # type: ignore[arg-type]


class TestRouteAfterQa:
    def test_coder_routing_loops_back(self):
        assert route_after_qa({"routing": "coder"}) == "coder"  # type: ignore[arg-type]

    def test_done_routing_goes_to_end(self):
        from langgraph.graph import END
        assert route_after_qa({"routing": "done"}) == END  # type: ignore[arg-type]


class TestBuildGraph:
    def test_graph_has_all_three_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "pm" in node_names
        assert "coder" in node_names
        assert "qa" in node_names


# ─────────────────────────────────────────────────────────────────────────────
# A2A protocol
# ─────────────────────────────────────────────────────────────────────────────

class TestA2AMessage:
    def test_valid_message_constructs(self):
        m = Message(sender="coder", receiver="qa", intent="review_request",
                    payload={"filename": "x.py"})
        assert m.correlation_id  # auto-generated UUID

    def test_correlation_id_is_preserved(self):
        m = Message(sender="coder", receiver="qa", intent="review_request",
                    payload={}, correlation_id="fixed-id")
        assert m.correlation_id == "fixed-id"

    def test_rejects_unknown_sender(self):
        with pytest.raises(ValueError, match="Unknown sender"):
            Message(sender="hacker", receiver="qa", intent="review_request", payload={})

    def test_rejects_unknown_intent(self):
        with pytest.raises(ValueError, match="Unknown intent"):
            Message(sender="coder", receiver="qa", intent="steal_data", payload={})

    def test_rejects_self_addressed_message(self):
        with pytest.raises(ValueError, match="must differ"):
            Message(sender="coder", receiver="coder", intent="review_request", payload={})


class TestA2ABroker:
    @pytest.mark.asyncio
    async def test_send_and_receive_roundtrip(self):
        broker = Broker()
        msg = Message(sender="coder", receiver="qa", intent="review_request",
                      payload={"filename": "x.py"})
        await broker.send(msg)
        received = await broker.receive("qa")
        assert received.sender == "coder"
        assert received.payload["filename"] == "x.py"
        assert received.correlation_id == msg.correlation_id

    def test_pending_count_starts_zero(self):
        broker = Broker()
        assert broker.pending("qa") == 0


class TestA2AValidation:
    def test_valid_intent_and_sender_passes(self):
        msg = Message(sender="coder", receiver="qa", intent="review_request", payload={})
        assert validate_incoming(msg, receiver="qa", allowed_senders={"coder"})

    def test_wrong_receiver_fails(self):
        msg = Message(sender="coder", receiver="qa", intent="review_request", payload={})
        assert not validate_incoming(msg, receiver="pm", allowed_senders={"coder"})

    def test_unauthorised_sender_fails(self):
        msg = Message(sender="coder", receiver="qa", intent="review_request", payload={})
        assert not validate_incoming(msg, receiver="qa", allowed_senders={"pm"})


class TestA2ACapabilities:
    def test_coder_advertises_review_request(self):
        assert "review_request" in AGENT_CAPABILITIES["coder"]["sends"]

    def test_qa_advertises_approved(self):
        assert "approved" in AGENT_CAPABILITIES["qa"]["sends"]

    def test_qa_receives_review_request(self):
        assert "review_request" in AGENT_CAPABILITIES["qa"]["receives"]


# ─────────────────────────────────────────────────────────────────────────────
# MCP adapter
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPAdapter:
    def test_registry_has_three_tools(self):
        assert set(TOOL_REGISTRY.keys()) == {"read_file", "write_file", "exec_python"}

    def test_list_tools_returns_descriptors(self):
        descriptors = list_tools()
        names = [d["name"] for d in descriptors]
        assert "read_file" in names
        assert "write_file" in names
        assert "exec_python" in names
        for d in descriptors:
            assert "description" in d
            assert "input_schema" in d

    def test_call_tool_unknown_returns_error_string(self):
        result = call_tool("no_such_tool", {})
        assert isinstance(result, str)
        assert "unknown tool" in result.lower()

    def test_call_tool_write_and_read_roundtrip(self, tmp_path, monkeypatch):
        # Point WORKSPACE at a tmp_path so this test is isolated from the repo.
        from pathlib import Path
        import tools.file_io as fio
        monkeypatch.setattr(fio, "WORKSPACE", Path(tmp_path))
        write_result = call_tool("write_file", {"path": "greet.py", "content": "print('hi')\n"})
        assert "written successfully" in write_result.lower()
        read_result = call_tool("read_file", {"path": "greet.py"})
        assert "print('hi')" in read_result
