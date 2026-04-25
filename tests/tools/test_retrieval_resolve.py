"""Tests for metadata-only retrieval_resolve tool."""

import json

import pytest

from hermes_state import SessionDB
from tools.retrieval_resolve_tool import retrieval_resolve

RAW_SENTINEL = "RETRIEVAL_RESOLVE_RAW_SENTINEL"
BANNED_KEYS = {
    "searchable_text",
    "snippet",
    "snippets",
    "preview",
    "previews",
    "stdout",
    "stderr",
    "output",
    "body",
    "content",
    "text",
    "message",
    "raw_output",
    "raw_body",
}


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


def _record_event(db, session_id="sid-current", event_id="evt-current", handle_id="rh-current", artifact_path="/tmp/artifact.txt"):
    db.create_session(session_id=session_id, source="cli")
    return db.record_session_event(
        session_id,
        "tool_result",
        summary="bounded public summary",
        payload={
            "tool": "terminal",
            "status": "stored",
            "artifact_path": artifact_path,
            "raw_chars": 2048,
            "artifact_kind": "delegation_child_final_response",
            "artifact_access_policy": "delegate_only",
            "parent_access": "metadata_only",
            "raw_output": RAW_SENTINEL,
            "content": RAW_SENTINEL,
            "nested": {"safe": "kept", "stderr": RAW_SENTINEL},
        },
        retrieval_handles=[
            {
                "handle_id": handle_id,
                "source_type": "session_event",
                "source_id": event_id,
                "locator": {
                    "artifact_path": artifact_path,
                    "sandbox_artifact_path": "/sandbox/artifact.txt",
                    "preview": RAW_SENTINEL,
                    "url": "https://example.test/file?token=secret&ok=yes",
                },
                "metadata": {
                    "tool": "terminal",
                    "raw_chars": 2048,
                    "artifact_kind": "delegation_child_final_response",
                    "artifact_access_policy": "delegate_only",
                    "parent_access": "metadata_only",
                    "raw_body": RAW_SENTINEL,
                    "safe_note": "safe metadata",
                },
            }
        ],
        source_type="tool_call",
        source_id="call-1",
        event_id=event_id,
        searchable_text=f"{RAW_SENTINEL} indexed-only raw text",
        created_at=1710000000.0,
    )


def _decode(result):
    return json.loads(result)


def _flatten(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _assert_no_banned_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in BANNED_KEYS
            assert key.lower() not in BANNED_KEYS
            _assert_no_banned_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_banned_keys(item)


class TestRetrievalResolve:
    def test_requires_event_or_handle_id(self, db):
        result = _decode(retrieval_resolve(db=db, current_session_id="sid-current"))
        assert result["success"] is False
        assert "event_id" in result["error"]

    def test_missing_session_scope_fails_closed_for_event_and_handle(self, db):
        _record_event(db)

        event_result = _decode(retrieval_resolve(event_id="evt-current", db=db))
        handle_result = _decode(retrieval_resolve(handle_id="rh-current", db=db))
        blank_result = _decode(retrieval_resolve(
            event_id="evt-current",
            db=db,
            current_session_id="  ",
        ))

        assert event_result["success"] is False
        assert handle_result["success"] is False
        assert blank_result["success"] is False
        assert "session id" in event_result["error"].lower()
        assert "session id" in handle_result["error"].lower()
        assert "session id" in blank_result["error"].lower()
        for result in (event_result, handle_result, blank_result):
            flattened = _flatten(result)
            assert "evt-current" not in flattened
            assert "rh-current" not in flattened

    def test_resolves_by_event_id_without_raw_leakage(self, db, tmp_path):
        artifact = tmp_path / "artifact.txt"
        artifact.write_text(RAW_SENTINEL)
        _record_event(db, artifact_path=str(artifact))

        result = _decode(retrieval_resolve(event_id="evt-current", db=db, current_session_id="sid-current"))

        assert result["success"] is True
        assert result["event"]["event_id"] == "evt-current"
        assert result["event"]["event_type"] == "tool_result"
        assert result["retrieval_handles"][0]["handle_id"] == "rh-current"
        assert result["payload_metadata"]["raw_chars"] == 2048
        assert result["payload_metadata"]["artifact_kind"] == "delegation_child_final_response"
        assert result["payload_metadata"]["artifact_access_policy"] == "delegate_only"
        assert result["payload_metadata"]["parent_access"] == "metadata_only"
        assert result["payload_metadata"]["nested"]["safe"] == "kept"
        assert result["raw_size"]["raw_chars"] == 2048
        assert result["artifacts"][0]["artifact_access_policy"] == "delegate_only"
        assert result["artifacts"][0]["parent_access"] == "metadata_only"
        assert result["artifacts"][0]["inspection_route"] == "delegate_task"
        assert result["artifact_inspection"]["recommended_tool"] == "delegate_task"
        assert result["artifact_inspection"]["recommended_toolsets"] == ["file"]
        assert result["artifact_inspection"]["parent_access"] == "metadata_only"
        assert str(artifact) in _flatten(result)
        flattened = _flatten(result)
        assert RAW_SENTINEL not in flattened
        assert "secret" not in flattened
        assert "token=secret" not in flattened
        _assert_no_banned_keys(result)

    def test_resolves_by_handle_id(self, db):
        _record_event(db)

        result = _decode(retrieval_resolve(handle_id="rh-current", db=db, current_session_id="sid-current"))

        assert result["success"] is True
        assert result["event"]["event_id"] == "evt-current"
        assert result["matched_handle"]["handle_id"] == "rh-current"
        assert result["matched_handle"]["metadata"]["safe_note"] == "safe metadata"
        assert result["matched_handle"]["metadata"]["artifact_access_policy"] == "delegate_only"
        assert result["matched_handle"]["metadata"]["parent_access"] == "metadata_only"

    def test_resolves_when_event_and_handle_match(self, db):
        _record_event(db)

        result = _decode(retrieval_resolve(
            event_id="evt-current",
            handle_id="rh-current",
            db=db,
            current_session_id="sid-current",
        ))

        assert result["success"] is True
        assert result["event"]["event_id"] == "evt-current"
        assert result["matched_handle"]["handle_id"] == "rh-current"

    def test_conflict_fails_closed_without_partial_metadata(self, db):
        _record_event(db, event_id="evt-current", handle_id="rh-current")
        _record_event(db, event_id="evt-other", handle_id="rh-other", artifact_path="/tmp/other.txt")

        result = _decode(retrieval_resolve(
            event_id="evt-current",
            handle_id="rh-other",
            db=db,
            current_session_id="sid-current",
        ))

        assert result["success"] is False
        assert "retrieval metadata" in result["error"]
        flattened = _flatten(result)
        assert "evt-current" not in flattened
        assert "rh-other" not in flattened

    def test_missing_event_or_handle_fails_closed(self, db):
        _record_event(db)

        missing_event = _decode(retrieval_resolve(
            event_id="evt-missing",
            db=db,
            current_session_id="sid-current",
        ))
        missing_handle = _decode(retrieval_resolve(
            handle_id="rh-missing",
            db=db,
            current_session_id="sid-current",
        ))

        assert missing_event["success"] is False
        assert missing_handle["success"] is False
        assert "retrieval metadata" in missing_event["error"]
        assert "retrieval metadata" in missing_handle["error"]

    def test_other_session_event_and_handle_are_not_resolved(self, db):
        _record_event(db, session_id="sid-other", event_id="evt-other", handle_id="rh-other")

        event_result = _decode(retrieval_resolve(
            event_id="evt-other",
            db=db,
            current_session_id="sid-current",
        ))
        handle_result = _decode(retrieval_resolve(
            handle_id="rh-other",
            db=db,
            current_session_id="sid-current",
        ))

        assert event_result["success"] is False
        assert handle_result["success"] is False
        assert "evt-other" not in _flatten(event_result)
        assert "rh-other" not in _flatten(handle_result)

    def test_ambiguous_handle_fails_closed(self, db):
        _record_event(db, event_id="evt-one", handle_id="rh-duplicate")
        _record_event(db, event_id="evt-two", handle_id="rh-duplicate", artifact_path="/tmp/two.txt")

        result = _decode(retrieval_resolve(
            handle_id="rh-duplicate",
            db=db,
            current_session_id="sid-current",
        ))

        assert result["success"] is False
        assert "retrieval metadata" in result["error"]
        assert "evt-one" not in _flatten(result)
        assert "evt-two" not in _flatten(result)
