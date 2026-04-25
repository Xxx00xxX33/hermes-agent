"""Dedicated parent-safe context-isolation sentinel tests.

These tests intentionally place a unique raw sentinel into child responses,
session-event payloads, handle metadata, searchable text, and artifact bodies.
Parent-visible surfaces must expose only summaries, size metadata, handles,
locators, and access-policy contracts -- never the sentinel itself.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tools.delegate_tool import delegate_task
from tools.retrieval_resolve_tool import retrieval_resolve
from tools.session_search_tool import session_search

RAW_CONTEXT_SENTINEL = "RAW_CONTEXT_ISOLATION_SENTINEL_7f0e6d2c9b"

BANNED_PARENT_KEYS = {
    "body",
    "content",
    "message",
    "preview",
    "raw_body",
    "raw_output",
    "searchable_text",
    "snippet",
    "stderr",
    "stdout",
    "text",
    "transcript",
}


def _flatten(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode(raw: str) -> dict:
    return json.loads(raw)


def _assert_no_sentinel(value) -> None:
    assert RAW_CONTEXT_SENTINEL not in _flatten(value)


def _assert_no_banned_parent_keys(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            assert lowered not in BANNED_PARENT_KEYS
            _assert_no_banned_parent_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_banned_parent_keys(item)


@pytest.fixture()
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _make_parent(session_db=None):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.session_id = "sid-parent"
    parent._session_db = session_db
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _record_context_event(
    db: SessionDB,
    *,
    session_id: str = "sid-current",
    event_id: str = "evt-context-isolation",
    handle_id: str = "rh-context-isolation",
    artifact_path: str = "/tmp/context-isolation-artifact.txt",
    searchable_text: str = "contextisolationmarker safe breadcrumb",
) -> None:
    db.create_session(session_id=session_id, source="cli")
    db.record_session_event(
        session_id,
        "delegation_child_response",
        summary="safe context-isolation summary",
        payload={
            "artifact_path": artifact_path,
            "artifact_kind": "delegation_child_final_response",
            "artifact_access_policy": "delegate_only",
            "parent_access": "metadata_only",
            "raw_chars": len(RAW_CONTEXT_SENTINEL),
            "raw_output": RAW_CONTEXT_SENTINEL,
            "content": RAW_CONTEXT_SENTINEL,
            "message": RAW_CONTEXT_SENTINEL,
            "nested": {
                "safe": "kept",
                "stderr": RAW_CONTEXT_SENTINEL,
            },
        },
        retrieval_handles=[
            {
                "handle_id": handle_id,
                "source_type": "session_event",
                "source_id": event_id,
                "locator": {
                    "artifact_path": artifact_path,
                    "preview": RAW_CONTEXT_SENTINEL,
                },
                "metadata": {
                    "artifact_kind": "delegation_child_final_response",
                    "artifact_access_policy": "delegate_only",
                    "parent_access": "metadata_only",
                    "raw_chars": len(RAW_CONTEXT_SENTINEL),
                    "raw_body": RAW_CONTEXT_SENTINEL,
                    "safe_note": "safe metadata",
                    "content_access_policy": "delegate_only",
                },
            }
        ],
        source_type="delegation_child",
        source_id="sid-child",
        event_id=event_id,
        searchable_text=f"{searchable_text} {RAW_CONTEXT_SENTINEL}",
        created_at=1710000000.0,
    )


def test_delegate_oversized_child_response_is_parent_safe_artifact_only():
    parent = _make_parent(session_db=MagicMock())
    parent._session_db.record_session_event.return_value = "evt-recorded"
    oversized = RAW_CONTEXT_SENTINEL * 400

    with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
        os.environ,
        {"HERMES_HOME": str(Path(tmpdir) / ".hermes")},
    ):
        with patch("run_agent.AIAgent") as MockAgent:
            child = MagicMock()
            child.model = "claude-sonnet-4-6"
            child.session_id = "sid-child"
            child.session_prompt_tokens = 0
            child.session_completion_tokens = 0
            child.run_conversation.return_value = {
                "final_response": oversized,
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [],
            }
            MockAgent.return_value = child

            result = _decode(delegate_task(
                goal="Inspect high-volume child output",
                parent_agent=parent,
            ))

        entry = result["results"][0]
        _assert_no_sentinel(result)
        _assert_no_banned_parent_keys(result)

        assert entry["status"] == "completed"
        assert entry["summary_truncated"] is True
        assert entry["summary_chars"] == len(oversized)
        assert entry["summary_omitted_chars"] == len(oversized)
        assert "omitted from parent context" in entry["summary"]

        handle = entry["summary_retrieval_handle"]
        assert handle["metadata"]["artifact_kind"] == "delegation_child_final_response"
        assert handle["metadata"]["artifact_access_policy"] == "delegate_only"
        assert handle["metadata"]["parent_access"] == "metadata_only"
        assert handle["metadata"]["raw_chars"] == len(oversized)

        artifact_path = Path(handle["locator"]["artifact_path"])
        assert artifact_path.exists()
        assert artifact_path.read_text(encoding="utf-8") == oversized

        parent._session_db.record_session_event.assert_called_once()
        _, kwargs = parent._session_db.record_session_event.call_args
        assert RAW_CONTEXT_SENTINEL not in kwargs["searchable_text"]
        assert kwargs["payload"]["artifact_access_policy"] == "delegate_only"
        assert kwargs["payload"]["parent_access"] == "metadata_only"
        _assert_no_sentinel({
            "summary": kwargs["summary"],
            "payload": kwargs["payload"],
            "retrieval_handles": kwargs["retrieval_handles"],
            "source_type": kwargs["source_type"],
            "source_id": kwargs["source_id"],
            "event_id": kwargs["event_id"],
            "searchable_text": kwargs["searchable_text"],
        })


def test_delegate_child_exception_omits_raw_error_from_parent_context():
    parent = _make_parent(session_db=MagicMock())
    raw_error = f"raw stderr traceback {RAW_CONTEXT_SENTINEL}"

    with patch("run_agent.AIAgent") as MockAgent:
        child = MagicMock()
        child.model = "claude-sonnet-4-6"
        child.session_id = "sid-child"
        child.run_conversation.side_effect = RuntimeError(raw_error)
        MockAgent.return_value = child

        result = _decode(delegate_task(
            goal="Trigger child exception with raw stderr",
            parent_agent=parent,
        ))

    _assert_no_sentinel(result)
    _assert_no_banned_parent_keys(result)

    entry = result["results"][0]
    assert entry["status"] == "error"
    assert entry["summary"] is None
    assert entry["error_type"] == "RuntimeError"
    assert entry["error_chars"] == len(raw_error)
    assert entry["error_bytes"] == len(raw_error.encode("utf-8"))
    assert "raw error details omitted" in entry["error"]


def test_delegate_failed_child_error_string_omits_raw_error_from_parent_context():
    parent = _make_parent(session_db=MagicMock())
    raw_error = f"tool stderr body {RAW_CONTEXT_SENTINEL}"

    with patch("run_agent.AIAgent") as MockAgent:
        child = MagicMock()
        child.model = "claude-sonnet-4-6"
        child.session_id = "sid-child"
        child.session_prompt_tokens = 0
        child.session_completion_tokens = 0
        child.run_conversation.return_value = {
            "final_response": "",
            "completed": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
            "error": raw_error,
        }
        MockAgent.return_value = child

        result = _decode(delegate_task(
            goal="Trigger child failed result with raw tool output",
            parent_agent=parent,
        ))

    _assert_no_sentinel(result)
    _assert_no_banned_parent_keys(result)

    entry = result["results"][0]
    assert entry["status"] == "failed"
    assert entry["summary"] == ""
    assert entry["error_type"] == "child_error"
    assert entry["error_chars"] == len(raw_error)
    assert entry["error_bytes"] == len(raw_error.encode("utf-8"))
    assert "raw error details omitted" in entry["error"]


def test_delegate_progress_display_surfaces_omit_raw_child_previews():
    parent = _make_parent(session_db=MagicMock())
    spinner = MagicMock()
    parent._delegate_spinner = spinner
    progress_events = []

    def parent_progress(*args, **kwargs):
        progress_events.append({"args": args, "kwargs": kwargs})

    parent.tool_progress_callback = parent_progress
    raw_preview = f"tool args stderr {RAW_CONTEXT_SENTINEL}"
    raw_thinking = f"reasoning over raw output {RAW_CONTEXT_SENTINEL}"
    raw_completion = f"completion preview {RAW_CONTEXT_SENTINEL}"

    with patch("run_agent.AIAgent") as MockAgent:
        def build_child(*args, **kwargs):
            child = MagicMock()
            child.model = "claude-sonnet-4-6"
            child.session_id = "sid-child"
            child.session_prompt_tokens = 0
            child.session_completion_tokens = 0
            child.tool_progress_callback = kwargs.get("tool_progress_callback")
            child.thinking_callback = kwargs.get("thinking_callback")

            def run_conversation(**_run_kwargs):
                callback = child.tool_progress_callback
                assert callback is not None
                callback(
                    "subagent.start",
                    preview=raw_preview,
                    message=raw_preview,
                    args={"cmd": raw_preview},
                )
                callback(
                    "tool.started",
                    tool_name="terminal",
                    preview=raw_preview,
                    args={"cmd": raw_preview},
                )
                if child.thinking_callback:
                    child.thinking_callback(raw_thinking)
                callback(
                    "subagent.complete",
                    preview=raw_completion,
                    summary=raw_completion,
                    error=f"raw error {RAW_CONTEXT_SENTINEL}",
                    message=f"raw message {RAW_CONTEXT_SENTINEL}",
                    details={"stderr": f"raw stderr {RAW_CONTEXT_SENTINEL}"},
                    status="completed",
                )
                return {
                    "final_response": "safe child summary",
                    "completed": True,
                    "interrupted": False,
                    "api_calls": 1,
                    "messages": [],
                }

            child.run_conversation.side_effect = run_conversation
            return child

        MockAgent.side_effect = build_child

        result = _decode(delegate_task(
            goal="Trigger raw child progress preview",
            parent_agent=parent,
        ))

    _assert_no_sentinel(result)
    _assert_no_sentinel(progress_events)
    _assert_no_sentinel([
        {"args": call.args, "kwargs": call.kwargs}
        for call in spinner.print_above.call_args_list
    ])

    assert result["results"][0]["status"] == "completed"
    assert progress_events or spinner.print_above.call_args_list


def test_retrieval_resolve_returns_metadata_only_for_delegate_only_artifact(session_db, tmp_path):
    artifact = tmp_path / "raw-child-response.txt"
    artifact.write_text(RAW_CONTEXT_SENTINEL, encoding="utf-8")
    _record_context_event(session_db, artifact_path=str(artifact))

    result = _decode(retrieval_resolve(
        event_id="evt-context-isolation",
        db=session_db,
        current_session_id="sid-current",
    ))

    assert result["success"] is True
    _assert_no_sentinel(result)
    _assert_no_banned_parent_keys(result)

    assert result["payload_metadata"]["artifact_access_policy"] == "delegate_only"
    assert result["payload_metadata"]["parent_access"] == "metadata_only"
    assert result["payload_metadata"]["raw_chars"] == len(RAW_CONTEXT_SENTINEL)
    assert result["payload_metadata"]["nested"]["safe"] == "kept"
    assert result["matched_handle"]["metadata"]["artifact_access_policy"] == "delegate_only"
    assert result["matched_handle"]["metadata"]["parent_access"] == "metadata_only"
    assert "content_access_policy" not in _flatten(result)

    artifact_entry = result["artifacts"][0]
    assert artifact_entry["artifact_path"] == str(artifact)
    assert artifact_entry["artifact_access_policy"] == "delegate_only"
    assert artifact_entry["parent_access"] == "metadata_only"
    assert artifact_entry["inspection_route"] == "delegate_task"

    inspection = result["artifact_inspection"]
    assert inspection["artifact_access_policy"] == "delegate_only"
    assert inspection["parent_access"] == "metadata_only"
    assert inspection["recommended_tool"] == "delegate_task"
    assert inspection["recommended_toolsets"] == ["file"]


def test_session_search_event_results_are_parent_safe(session_db, tmp_path):
    artifact = tmp_path / "raw-session-event-artifact.txt"
    artifact.write_text(RAW_CONTEXT_SENTINEL, encoding="utf-8")
    _record_context_event(
        session_db,
        artifact_path=str(artifact),
        searchable_text="contextisolationmarker searchable safe breadcrumb",
    )

    result = _decode(session_search(
        query="contextisolationmarker",
        limit=3,
        db=session_db,
        current_session_id="sid-unrelated-current",
    ))

    assert result["success"] is True
    assert result["event_count"] >= 1
    _assert_no_sentinel(result["event_results"])
    _assert_no_banned_parent_keys(result["event_results"])

    event = result["event_results"][0]
    assert event["kind"] == "session_event"
    assert event["event_id"] == "evt-context-isolation"
    assert event["summary"] == "safe context-isolation summary"
    assert event["payload"]["artifact_access_policy"] == "delegate_only"
    assert event["payload"]["parent_access"] == "metadata_only"
    assert event["payload"]["raw_chars"] == len(RAW_CONTEXT_SENTINEL)
    assert event["payload"]["nested"]["safe"] == "kept"
    assert event["retrieval_handles"][0]["metadata"]["artifact_access_policy"] == "delegate_only"
    assert event["retrieval_handles"][0]["metadata"]["parent_access"] == "metadata_only"
    assert "content_access_policy" not in _flatten(event)
