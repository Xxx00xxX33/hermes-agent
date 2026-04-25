"""Regression tests for gateway /model support of config.yaml custom_providers."""

from types import SimpleNamespace

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = _DummyLock()
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_handle_model_command_lists_saved_custom_provider(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openai-codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                },
                "providers": {},
                "custom_providers": [
                    {
                        "name": "Local (127.0.0.1:4141)",
                        "base_url": "http://127.0.0.1:4141/v1",
                        "model": "rotator-openrouter-coding",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    result = await _make_runner()._handle_model_command(_make_event())

    assert result is not None
    assert "Local (127.0.0.1:4141)" in result
    assert "custom:local-(127.0.0.1:4141)" in result
    assert "rotator-openrouter-coding" in result


@pytest.mark.asyncio
async def test_handle_model_command_display_honors_cached_agent_config_context_length(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    custom_base_url = "https://gpt.lucienfc.eu.org/v1"
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "custom",
                    "base_url": custom_base_url,
                    "context_length": 200000,
                },
                "providers": {},
                "custom_providers": [],
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run
    import agent.model_metadata as model_metadata

    captured = {}

    def fake_get_model_context_length(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return kwargs.get("config_context_length") or 128000

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_metadata, "get_model_context_length", fake_get_model_context_length)

    runner = _make_runner()
    session_key = runner._session_key_for_source(_make_event("/model gpt-5.5").source)
    cached_agent = SimpleNamespace(
        _config_context_length=200000,
        switch_model=lambda **kwargs: None,
    )
    runner._agent_cache[session_key] = (cached_agent, "sig")

    result = await runner._handle_model_command(_make_event("/model gpt-5.5"))

    assert result is not None
    assert "Context: 200,000 tokens" in result
    assert captured["model"] == "gpt-5.5"
    assert captured["kwargs"]["provider"] == "custom"
    assert captured["kwargs"]["config_context_length"] == 200000
