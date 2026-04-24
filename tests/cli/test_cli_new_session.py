"""Regression tests for CLI fresh-session commands."""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import timedelta
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from tools.todo_tool import TodoStore


class _FakeCompressor:
    """Minimal stand-in for ContextCompressor."""

    def __init__(self):
        self.last_prompt_tokens = 500
        self.last_completion_tokens = 200
        self.last_total_tokens = 700
        self.compression_count = 3
        self._context_probed = True


class _FakeAgent:
    def __init__(
        self,
        session_id: str,
        session_start,
        *,
        model: str = "anthropic/claude-opus-4.6",
        requested_provider: str = "auto",
        provider: str = "openrouter",
        base_url: str | None = "https://openrouter.ai/api/v1",
        api_mode: str = "chat_completions",
        api_key: str | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
    ):
        self.session_id = session_id
        self.session_start = session_start
        self.model = model
        self.requested_provider = requested_provider
        self.provider = provider
        self.base_url = base_url
        self.api_mode = api_mode
        self.api_key = api_key
        self.acp_command = acp_command
        self.acp_args = list(acp_args or [])
        self._last_flushed_db_idx = 7
        self._todo_store = TodoStore()
        self._todo_store.write(
            [{"id": "t1", "content": "unfinished task", "status": "in_progress"}]
        )
        self.flush_memories = MagicMock()
        self.commit_memory_session = MagicMock()
        self._invalidate_system_prompt = MagicMock()
        self.switch_model = MagicMock(side_effect=self._apply_switch_model)

        # Token counters (non-zero to verify reset)
        self.session_total_tokens = 1000
        self.session_input_tokens = 600
        self.session_output_tokens = 400
        self.session_prompt_tokens = 550
        self.session_completion_tokens = 350
        self.session_cache_read_tokens = 100
        self.session_cache_write_tokens = 50
        self.session_reasoning_tokens = 80
        self.session_api_calls = 5
        self.session_estimated_cost_usd = 0.42
        self.session_cost_status = "estimated"
        self.session_cost_source = "openrouter"
        self.context_compressor = _FakeCompressor()

    def _apply_switch_model(
        self,
        *,
        new_model,
        new_provider,
        api_key,
        base_url,
        api_mode,
        requested_provider=None,
        acp_command=None,
        acp_args=None,
    ):
        self.model = new_model
        self.requested_provider = requested_provider or new_provider
        self.provider = new_provider
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = api_mode
        self.acp_command = acp_command
        self.acp_args = list(acp_args or [])

    def reset_session_state(self):
        """Mirror the real AIAgent.reset_session_state()."""
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.last_prompt_tokens = 0
            self.context_compressor.last_completion_tokens = 0
            self.context_compressor.last_total_tokens = 0
            self.context_compressor.compression_count = 0
            self.context_compressor._context_probed = False


def _make_cli(env_overrides=None, config_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        _clean_config.update(config_overrides)
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI(**kwargs)


def _session_model_config(session_row):
    raw = session_row.get("model_config")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return {}


def _prepare_cli_with_active_session(tmp_path, *, env_overrides=None, config_overrides=None):
    cli = _make_cli(env_overrides=env_overrides, config_overrides=config_overrides)
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(session_id=cli.session_id, source="cli", model=cli.model)

    cli.agent = _FakeAgent(
        cli.session_id,
        cli.session_start,
        model=cli.model,
        requested_provider=cli.requested_provider,
        provider=cli.provider,
        base_url=cli.base_url,
        api_mode=cli.api_mode,
        api_key=cli.api_key,
        acp_command=cli.acp_command,
        acp_args=list(cli.acp_args or []),
    )
    cli.conversation_history = [{"role": "user", "content": "hello"}]
    cli._active_agent_route_signature = cli._agent_route_signature(
        model=cli.agent.model,
        provider=cli.agent.provider,
        base_url=cli.agent.base_url,
        api_mode=cli.agent.api_mode,
        acp_command=cli.agent.acp_command,
        acp_args=list(cli.agent.acp_args or []),
    )

    old_session_start = cli.session_start - timedelta(seconds=1)
    cli.session_start = old_session_start
    cli.agent.session_start = old_session_start
    return cli


def test_new_command_creates_real_fresh_session_and_resets_agent_state(tmp_path):
    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id
    old_session_start = cli.session_start

    cli.process_command("/new")

    assert cli.session_id != old_session_id

    old_session = cli._session_db.get_session(old_session_id)
    assert old_session is not None
    assert old_session["end_reason"] == "new_session"

    new_session = cli._session_db.get_session(cli.session_id)
    assert new_session is not None

    cli._session_db.append_message(cli.session_id, role="user", content="next turn")

    assert cli.agent.session_id == cli.session_id
    assert cli.agent._last_flushed_db_idx == 0
    assert cli.agent._todo_store.read() == []
    assert cli.session_start > old_session_start
    assert cli.agent.session_start == cli.session_start
    cli.agent.flush_memories.assert_called_once_with([{"role": "user", "content": "hello"}])
    cli.agent._invalidate_system_prompt.assert_called_once()


def test_reset_command_is_alias_for_new_session(tmp_path):
    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id

    cli.process_command("/reset")

    assert cli.session_id != old_session_id
    assert cli._session_db.get_session(old_session_id)["end_reason"] == "new_session"
    assert cli._session_db.get_session(cli.session_id) is not None


def test_clear_command_starts_new_session_before_redrawing(tmp_path):
    cli = _prepare_cli_with_active_session(tmp_path)
    cli.console = MagicMock()
    cli.show_banner = MagicMock()

    old_session_id = cli.session_id
    cli.process_command("/clear")

    assert cli.session_id != old_session_id
    assert cli._session_db.get_session(old_session_id)["end_reason"] == "new_session"
    assert cli._session_db.get_session(cli.session_id) is not None
    cli.console.clear.assert_called_once()
    cli.show_banner.assert_called_once()
    assert cli.conversation_history == []


def test_new_session_resets_token_counters(tmp_path):
    """Regression test for #2099: /new must zero all token counters."""
    cli = _prepare_cli_with_active_session(tmp_path)

    # Verify counters are non-zero before reset
    agent = cli.agent
    assert agent.session_total_tokens > 0
    assert agent.session_api_calls > 0
    assert agent.context_compressor.compression_count > 0

    cli.process_command("/new")

    # All agent token counters must be zero
    assert agent.session_total_tokens == 0
    assert agent.session_input_tokens == 0
    assert agent.session_output_tokens == 0
    assert agent.session_prompt_tokens == 0
    assert agent.session_completion_tokens == 0
    assert agent.session_cache_read_tokens == 0
    assert agent.session_cache_write_tokens == 0
    assert agent.session_reasoning_tokens == 0
    assert agent.session_api_calls == 0
    assert agent.session_estimated_cost_usd == 0.0
    assert agent.session_cost_status == "unknown"
    assert agent.session_cost_source == "none"

    # Context compressor counters must also be zero
    comp = agent.context_compressor
    assert comp.last_prompt_tokens == 0
    assert comp.last_completion_tokens == 0
    assert comp.last_total_tokens == 0
    assert comp.compression_count == 0
    assert comp._context_probed is False



def test_new_session_resets_provider_to_startup_default_not_previous_session_override(tmp_path):
    cli = _prepare_cli_with_active_session(
        tmp_path,
        config_overrides={
            "model": {
                "default": "gpt-5.4",
                "base_url": "https://gpt.lucienfc.eu.org/v1",
                "provider": "custom",
            },
            "display": {"compact": False, "tool_progress": "all"},
            "agent": {},
            "terminal": {"env_type": "local"},
        },
    )
    old_session_id = cli.session_id

    cli.model = "octopus-model"
    cli.requested_provider = "custom:octopus"
    cli.provider = "custom:octopus"
    cli.base_url = "https://octopus.lucienfc.eu.org/v1"
    cli.api_mode = "chat_completions"
    cli.api_key = None
    cli._explicit_api_key = None
    cli._explicit_base_url = "https://octopus.lucienfc.eu.org/v1"
    cli._active_agent_route_signature = cli._agent_route_signature(
        model=cli.model,
        provider=cli.provider,
        base_url=cli.base_url,
        api_mode=cli.api_mode,
        acp_command=cli.acp_command,
        acp_args=list(cli.acp_args or []),
    )
    cli.agent._apply_switch_model(
        new_model=cli.model,
        new_provider=cli.provider,
        api_key=cli.api_key,
        base_url=cli.base_url,
        api_mode=cli.api_mode,
        requested_provider=cli.requested_provider,
        acp_command=cli.acp_command,
        acp_args=list(cli.acp_args or []),
    )

    cli.process_command("/new")

    assert cli.session_id != old_session_id
    assert cli.model == "gpt-5.4"
    assert cli.requested_provider == "custom"
    assert cli.provider == "custom"
    assert cli.base_url == "https://gpt.lucienfc.eu.org/v1"
    assert cli.agent.model == "gpt-5.4"
    assert cli.agent.requested_provider == "custom"
    assert cli.agent.provider == "custom"
    assert cli.agent.base_url == "https://gpt.lucienfc.eu.org/v1"

    new_session = cli._session_db.get_session(cli.session_id)
    model_config = _session_model_config(new_session)
    assert model_config["model"] == "gpt-5.4"
    assert model_config["requested_provider"] == "custom"
    assert model_config["provider"] == "custom"
    assert model_config["base_url"] == "https://gpt.lucienfc.eu.org/v1"


def test_resume_restores_session_specific_provider_and_model(tmp_path, monkeypatch):
    cli = _prepare_cli_with_active_session(
        tmp_path,
        config_overrides={
            "model": {
                "default": "gpt-5.4",
                "base_url": "https://gpt.lucienfc.eu.org/v1",
                "provider": "custom",
            },
            "display": {"compact": False, "tool_progress": "all"},
            "agent": {},
            "terminal": {"env_type": "local"},
        },
    )

    target_id = "20260418_000000_abc123"
    cli._session_db.create_session(
        session_id=target_id,
        source="cli",
        model="octopus-model",
        model_config={
            "model": "octopus-model",
            "requested_provider": "custom:octopus",
            "provider": "custom:octopus",
            "base_url": "https://octopus.lucienfc.eu.org/v1",
            "api_mode": "chat_completions",
        },
    )
    cli._session_db.append_message(target_id, role="user", content="resume me")
    monkeypatch.setattr("hermes_cli.main._resolve_session_by_name_or_id", lambda _target: target_id)

    cli.process_command(f"/resume {target_id}")

    assert cli.session_id == target_id
    assert cli.model == "octopus-model"
    assert cli.requested_provider == "custom:octopus"
    assert cli.provider == "custom:octopus"
    assert cli.base_url == "https://octopus.lucienfc.eu.org/v1"
    assert cli.agent.session_id == target_id
    assert cli.agent.model == "octopus-model"
    assert cli.agent.requested_provider == "custom:octopus"
    assert cli.agent.provider == "custom:octopus"
    assert cli.agent.base_url == "https://octopus.lucienfc.eu.org/v1"
    cli.agent.switch_model.assert_called()
