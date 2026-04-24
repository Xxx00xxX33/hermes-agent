from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from cli import HermesCLI


def _make_cli(model: str = "anthropic/claude-sonnet-4-20250514"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = model
    cli_obj.provider = "anthropic" if model.startswith("anthropic/") else None
    cli_obj.requested_provider = cli_obj.provider
    cli_obj._provider_source = ""
    cli_obj.session_start = datetime.now() - timedelta(minutes=14, seconds=32)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    cli_obj.reasoning_config = None
    cli_obj._status_bar_visible = False
    cli_obj._background_tasks = {}
    cli_obj._background_task_counter = 0
    cli_obj._subagent_active_tasks = {}
    cli_obj._subagent_task_counter = 0
    cli_obj._pending_edit_snapshots = {}
    cli_obj._inline_diffs_enabled = False
    cli_obj._voice_mode = False
    cli_obj._emit_persistent_tool_progress_line = lambda *args, **kwargs: None
    cli_obj._pending_title = None
    cli_obj._session_db = None
    cli_obj.session_id = "session-1"
    cli_obj._tmux_pane_title = ""
    cli_obj._tmux_session_task_title = ""
    cli_obj._tmux_task_title_locked = False
    return cli_obj


def _attach_agent(
    cli_obj,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    api_calls: int,
    context_tokens: int,
    context_length: int,
    compressions: int = 0,
):
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider="anthropic" if cli_obj.model.startswith("anthropic/") else None,
        base_url="",
        session_input_tokens=prompt_tokens,
        session_output_tokens=completion_tokens,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_prompt_tokens=prompt_tokens,
        session_completion_tokens=completion_tokens,
        session_total_tokens=total_tokens,
        session_api_calls=api_calls,
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=context_tokens,
            context_length=context_length,
            compression_count=compressions,
        ),
    )
    return cli_obj


def _attach_todo_items(cli_obj, *items):
    items = [
        {"id": f"t{i}", "content": content, "status": status}
        for i, (content, status) in enumerate(items, start=1)
    ]
    if cli_obj.agent is None:
        cli_obj.agent = SimpleNamespace()
    cli_obj.agent._todo_store = SimpleNamespace(read=lambda: [item.copy() for item in items])
    return cli_obj


class TestStatusbarFastToolRefreshRuntimePatch:
    def test_status_bar_places_subtask_progress_at_primary_line_end(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(model="DeepSeek-R1"),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=12_450,
                context_length=200_000,
            ),
            ("build", "completed"),
            ("deploy", "in_progress"),
            ("docs", "pending"),
        )
        cli_obj._subagent_task_counter = 1
        cli_obj._subagent_active_tasks = {"sa-1": 1}
        cli_obj._status_bar_visible = True
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            primary_fragments = cli_obj._get_status_bar_fragments()
            secondary_fragments = cli_obj._get_status_bar_secondary_fragments()

        primary_text = "".join(text for _, text in primary_fragments)
        secondary_text = "".join(text for _, text in secondary_fragments)
        assert "子任务：1/1" in primary_text
        assert primary_text.rstrip().endswith("子任务：1/1")
        assert "(2/3)" in secondary_text
        assert "子任务：" not in secondary_text

    def test_on_tool_progress_delegate_start_updates_subtask_counter_immediately(self):
        cli_obj = _make_cli()
        cli_obj._invalidate = MagicMock()
        cli_obj._subagent_active_tasks = {}
        cli_obj._subagent_task_counter = 0

        cli_obj._on_tool_progress(
            "tool.started",
            "delegate_task",
            "delegating",
            {"goal": "inspect"},
            tool_call_id="call-1",
        )

        assert cli_obj._subagent_active_tasks["call-1"] == 1
        assert cli_obj._subagent_task_counter == 1
        assert call(min_interval=0.0) in cli_obj._invalidate.call_args_list

    def test_on_tool_progress_delegate_completion_removes_active_task_immediately(self):
        cli_obj = _make_cli()
        cli_obj._invalidate = MagicMock()
        cli_obj._subagent_active_tasks = {"call-1": 1}
        cli_obj._subagent_task_counter = 1
        cli_obj._tool_start_time = 123.0

        cli_obj._on_tool_progress(
            "tool.completed",
            "delegate_task",
            None,
            {},
            tool_call_id="call-1",
        )

        assert cli_obj._subagent_active_tasks == {}
        assert cli_obj._tool_start_time == 0.0
        cli_obj._invalidate.assert_called_once_with(min_interval=0.0)

    def test_chat_resets_stale_subtask_tracking_before_new_turn(self):
        cli_obj = _make_cli()
        cli_obj._subagent_active_tasks = {"old-call": 1}
        cli_obj._subagent_task_counter = 4
        cli_obj._ensure_runtime_credentials = lambda: False

        result = cli_obj.chat("continue")

        assert result is None
        assert cli_obj._subagent_active_tasks == {}
        assert cli_obj._subagent_task_counter == 0
