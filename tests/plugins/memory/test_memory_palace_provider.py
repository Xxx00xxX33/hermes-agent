import json

import pytest

from plugins.memory.memory_palace import (
    MemoryPalaceMemoryProvider,
    _MemoryPalaceClient,
    _build_session_summary,
    _extract_turn_events,
)


class FakeClient:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.search_calls = []
        self.append_calls = []
        self.search_results = []

    def search(self, query, *, max_results, mode, path_prefix=""):
        self.search_calls.append({
            "query": query,
            "max_results": max_results,
            "mode": mode,
            "path_prefix": path_prefix,
        })
        return self.search_results

    def append_or_create(self, uri, text, *, priority, disclosure=""):
        self.append_calls.append({
            "uri": uri,
            "text": text,
            "priority": priority,
            "disclosure": disclosure,
        })
        return True


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("MEMORY_PALACE_URL", "http://memory-palace.local")
    monkeypatch.setenv("MEMORY_PALACE_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.memory_palace._MemoryPalaceClient", FakeClient)
    p = MemoryPalaceMemoryProvider()
    p.initialize("session-1", hermes_home="/tmp/hermes", platform="cli")
    return p


def test_is_available_false_without_env(monkeypatch):
    monkeypatch.delenv("MEMORY_PALACE_URL", raising=False)
    monkeypatch.delenv("MEMORY_PALACE_API_KEY", raising=False)
    p = MemoryPalaceMemoryProvider()
    assert p.is_available() is False


def test_get_config_schema_contains_url_and_api_key():
    p = MemoryPalaceMemoryProvider()
    schema = p.get_config_schema()
    assert [field["key"] for field in schema] == ["base_url", "api_key"]
    assert schema[0]["env_var"] == "MEMORY_PALACE_URL"
    assert schema[1]["env_var"] == "MEMORY_PALACE_API_KEY"
    assert schema[1]["secret"] is True


def test_prefetch_formats_recall_block(provider):
    provider._client.search_results = [
        {"snippet": "User prefers concise, evidence-backed explanations."},
        {"snippet": "Prefer a Hermes MemoryProvider bridge over MCP-only."},
    ]
    result = provider.prefetch("how should I integrate memory?")
    assert "## Memory Palace Recall" in result
    assert "User prefers concise" in result
    assert provider._client.search_calls[0]["path_prefix"] == "external_agents"


def test_extract_turn_events_preference_strategy_and_lesson():
    events = _extract_turn_events(
        "我希望能完美契合 Hermes 的自我学习特性。",
        "The best way is to prefer a native MemoryProvider bridge over MCP-only. Do not put evo in the per-turn hot path.",
        session_id="session-1",
    )
    kinds = [event["event_type"] for event in events]
    assert "preference" in kinds
    assert "strategy" in kinds
    assert "failure_lesson" in kinds


def test_sync_turn_writes_events_and_preference_rollup(provider):
    provider.sync_turn(
        "I prefer designs that preserve Hermes-native learning semantics.",
        "The best way is to use a layered memory bridge.",
        session_id="session-1",
    )
    provider._sync_thread.join(timeout=1)
    uris = [call["uri"] for call in provider._client.append_calls]
    assert "notes://external_agents/hermes_events" in uris
    assert "core://external_agents/hermes_user_preference_rollup" in uris


def test_on_session_end_writes_summary(provider):
    messages = [
        {"role": "user", "content": "I prefer designs that preserve Hermes-native learning semantics."},
        {"role": "assistant", "content": "The best way is to use a native bridge. Do not put evo in the per-turn hot path."},
    ]
    provider.on_session_end(messages)
    uris = [call["uri"] for call in provider._client.append_calls]
    assert "notes://external_agents/hermes_sessions" in uris
    session_writes = [call for call in provider._client.append_calls if call["uri"] == "notes://external_agents/hermes_sessions"]
    assert "Preferences confirmed" in session_writes[-1]["text"]
    assert "Failure lessons" in session_writes[-1]["text"]


def test_build_session_summary_returns_empty_for_no_user_or_assistant_messages():
    assert _build_session_summary([], session_id="session-1") == ""


def test_provider_has_no_explicit_tools(provider):
    assert provider.get_tool_schemas() == []


def test_shutdown_clears_threads(provider):
    provider.sync_turn(
        "I prefer concise docs.",
        "The best way is to keep the bridge small.",
        session_id="session-1",
    )
    provider.shutdown()
    assert provider._sync_thread is None
    assert provider._summary_thread is None


def test_client_create_bootstraps_parent_chain(monkeypatch):
    client = _MemoryPalaceClient("http://memory-palace.local", "test-key", 5.0)
    created_uris = set()
    post_payloads = []

    def fake_read(uri: str):
        return "existing" if uri in created_uris else None

    class DummyResponse:
        status_code = 200
        text = '{"created": true}'

        def json(self):
            return {"created": True}

    class DummySession:
        def post(self, url, headers, data, timeout):
            payload = json.loads(data)
            full_path = f"{payload['parent_path']}/{payload['title']}" if payload["parent_path"] else payload["title"]
            created_uris.add(f"{payload['domain']}://{full_path}")
            post_payloads.append(payload)
            return DummyResponse()

    monkeypatch.setattr(client, "read", fake_read)
    client._session = DummySession()

    assert client.create(
        "notes://external_agents/hermes_events",
        "hello",
        priority=5,
        disclosure="When recalling Hermes durable learning events",
    )
    assert [payload["title"] for payload in post_payloads] == ["external_agents", "hermes_events"]
    assert post_payloads[1]["parent_path"] == "external_agents"
