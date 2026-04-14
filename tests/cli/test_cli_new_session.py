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
    def __init__(self, session_id: str, session_start):
        self.session_id = session_id
        self.session_start = session_start
        self.model = "anthropic/claude-opus-4.6"
        self.requested_provider = "auto"
        self.provider = "auto"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_mode = "chat_completions"
        self.api_key = "test-key"
        self.acp_command = None
        self.acp_args = []
        self._last_flushed_db_idx = 7
        self._todo_store = TodoStore()
        self._todo_store.write(
            [{"id": "t1", "content": "unfinished task", "status": "in_progress"}]
        )
        self.flush_memories = MagicMock()
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

    def _apply_switch_model(self, **kwargs):
        self.model = kwargs.get("new_model", self.model)
        self.provider = kwargs.get("new_provider", self.provider)
        self.requested_provider = kwargs.get("requested_provider", self.requested_provider)
        self.base_url = kwargs.get("base_url", self.base_url)
        self.api_mode = kwargs.get("api_mode", self.api_mode)
        self.api_key = kwargs.get("api_key", self.api_key)
        self.acp_command = kwargs.get("acp_command", self.acp_command)
        self.acp_args = list(kwargs.get("acp_args") or self.acp_args)

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
        "dotenv": MagicMock(),
        "fire": MagicMock(),
        "run_agent": MagicMock(AIAgent=MagicMock()),
        "tools.browser_tool": MagicMock(_emergency_cleanup_all_sessions=MagicMock()),
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


def _prepare_cli_with_active_session(tmp_path):
    cli = _make_cli()
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(session_id=cli.session_id, source="cli", model=cli.model)

    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "hello"}]

    old_session_start = cli.session_start - timedelta(seconds=1)
    cli.session_start = old_session_start
    cli.agent.session_start = old_session_start
    return cli


def _session_reasoning_config(session_row):
    model_config = json.loads(session_row["model_config"] or "{}")
    return model_config.get("reasoning_config")


def _session_model_config(session_row):
    return json.loads(session_row["model_config"] or "{}")


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


def test_new_session_uses_default_reasoning_effort_not_previous_session_override(tmp_path):
    cli = _make_cli(config_overrides={"agent": {"reasoning_effort": "high"}})
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
        model_config={"reasoning_config": cli.reasoning_config},
    )
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "hello"}]

    cli.process_command("/reasoning low")

    first_session_id = cli.session_id
    first_session = cli._session_db.get_session(first_session_id)
    assert _session_reasoning_config(first_session) == {"enabled": True, "effort": "low"}

    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.process_command("/new")

    assert cli.session_id != first_session_id
    assert cli.reasoning_config == {"enabled": True, "effort": "high"}

    new_session = cli._session_db.get_session(cli.session_id)
    assert new_session is not None
    assert _session_reasoning_config(new_session) == {"enabled": True, "effort": "high"}


def test_resume_restores_session_specific_reasoning_effort(tmp_path):
    cli = _make_cli(config_overrides={"agent": {"reasoning_effort": "high"}})
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
        model_config={"reasoning_config": cli.reasoning_config},
    )
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "hello"}]

    cli.process_command("/reasoning low")

    original_session_id = cli.session_id
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.process_command("/new")
    second_session_id = cli.session_id

    assert cli.reasoning_config == {"enabled": True, "effort": "high"}

    with patch.dict(
        sys.modules,
        {"hermes_cli.main": MagicMock(_resolve_session_by_name_or_id=lambda target: target)},
    ):
        cli.process_command(f"/resume {original_session_id}")

    assert cli.session_id == original_session_id
    assert cli.reasoning_config == {"enabled": True, "effort": "low"}
    assert cli._session_db.get_session(second_session_id)["end_reason"] == "resumed_other"
    assert cli._session_db.get_session(original_session_id)["end_reason"] is None


def test_new_session_resets_provider_to_startup_default_not_previous_session_override(tmp_path):
    cli = _make_cli(
        config_overrides={
            "model": {
                "default": "anthropic/claude-opus-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
            }
        }
    )
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
        model_config=cli._session_routing_model_config(),
    )
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "hello"}]

    cli.model = "gpt-5.4"
    cli.requested_provider = "openai-codex"
    cli.provider = "openai-codex"
    cli.base_url = "https://chatgpt.com/backend-api/codex"
    cli.api_mode = "codex_responses"
    cli._explicit_base_url = cli.base_url
    cli.api_key = "test-codex-key"
    cli._explicit_api_key = cli.api_key
    cli._persist_session_routing_config()

    cli.process_command("/new")

    assert cli.model == "anthropic/claude-opus-4.6"
    assert cli.requested_provider == "openrouter"
    assert cli.provider == "openrouter"
    assert cli.base_url == "https://openrouter.ai/api/v1"
    assert cli.api_mode == "chat_completions"
    assert cli._explicit_base_url is None
    assert cli.agent is not None
    cli.agent.switch_model.assert_called()
    assert cli.agent.provider == "openrouter"

    new_session = cli._session_db.get_session(cli.session_id)
    assert new_session is not None
    model_config = _session_model_config(new_session)
    assert model_config["requested_provider"] == "openrouter"
    assert model_config["provider"] == "openrouter"
    assert model_config["base_url"] == "https://openrouter.ai/api/v1"


def test_resume_restores_session_specific_provider_and_model(tmp_path):
    cli = _make_cli(
        config_overrides={
            "model": {
                "default": "anthropic/claude-opus-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
            }
        }
    )
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
        model_config=cli._session_routing_model_config(),
    )
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "hello"}]

    cli.model = "gpt-5.4"
    cli.requested_provider = "openai-codex"
    cli.provider = "openai-codex"
    cli.base_url = "https://chatgpt.com/backend-api/codex"
    cli.api_mode = "codex_responses"
    cli._explicit_base_url = cli.base_url
    cli.api_key = "test-codex-key"
    cli._explicit_api_key = cli.api_key
    cli._persist_session_routing_config()

    original_session_id = cli.session_id
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.process_command("/new")
    assert cli.requested_provider == "openrouter"

    with patch.dict(
        sys.modules,
        {"hermes_cli.main": MagicMock(_resolve_session_by_name_or_id=lambda target: target)},
    ):
        cli.process_command(f"/resume {original_session_id}")

    assert cli.session_id == original_session_id
    assert cli.model == "gpt-5.4"
    assert cli.requested_provider == "openai-codex"
    assert cli.provider == "openai-codex"
    assert cli.base_url == "https://chatgpt.com/backend-api/codex"
    assert cli.api_mode == "codex_responses"
    assert cli.agent is not None
    cli.agent.switch_model.assert_called()
    assert cli.agent.provider == "openai-codex"


def test_single_query_main_closes_session_before_exit_summary():
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
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
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
        "dotenv": MagicMock(),
        "fire": MagicMock(),
        "run_agent": MagicMock(AIAgent=MagicMock()),
        "tools.browser_tool": MagicMock(_emergency_cleanup_all_sessions=MagicMock()),
    }

    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        fake_cli = MagicMock()
        fake_cli.console = MagicMock()
        fake_cli.chat = MagicMock(return_value="done")
        fake_cli._close_session_in_db = MagicMock()
        fake_cli._print_exit_summary = MagicMock()
        fake_cli.show_banner = MagicMock()

        with patch.object(_cli_mod, "HermesCLI", return_value=fake_cli),              patch.object(_cli_mod, "_run_cleanup", return_value=None),              patch.object(_cli_mod.atexit, "register", return_value=None),              patch("hermes_cli.tools_config._get_platform_tools", return_value=[]),              patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            _cli_mod.main(query="hello")

    fake_cli.chat.assert_called_once_with("hello", images=None)
    fake_cli._close_session_in_db.assert_called_once_with("cli_close")
    fake_cli._print_exit_summary.assert_called_once()
