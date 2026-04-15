from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI, _ACCENT, _RST


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
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
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
        session_input_tokens=input_tokens if input_tokens is not None else prompt_tokens,
        session_output_tokens=output_tokens if output_tokens is not None else completion_tokens,
        session_cache_read_tokens=cache_read_tokens,
        session_cache_write_tokens=cache_write_tokens,
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


def _attach_todos(cli_obj, *statuses: str):
    return _attach_todo_items(
        cli_obj,
        *[(f"task {i}", status) for i, status in enumerate(statuses, start=1)],
    )


def _attach_todo_items(cli_obj, *items):
    items = [
        {"id": f"t{i}", "content": content, "status": status}
        for i, (content, status) in enumerate(items, start=1)
    ]
    if cli_obj.agent is None:
        cli_obj.agent = SimpleNamespace()
    cli_obj.agent._todo_store = SimpleNamespace(read=lambda: [item.copy() for item in items])
    return cli_obj


def _attach_background_tasks(cli_obj, *, total: int, active: int):
    cli_obj._background_task_counter = total
    cli_obj._background_tasks = {f"bg-{i}": object() for i in range(active)}
    return cli_obj


class TestCLIStatusBar:
    def test_context_style_thresholds(self):
        cli_obj = _make_cli()

        assert cli_obj._status_bar_context_style(None) == "class:status-bar-dim"
        assert cli_obj._status_bar_context_style(10) == "class:status-bar-good"
        assert cli_obj._status_bar_context_style(50) == "class:status-bar-warn"
        assert cli_obj._status_bar_context_style(81) == "class:status-bar-bad"
        assert cli_obj._status_bar_context_style(95) == "class:status-bar-critical"

    def test_build_status_bar_text_for_wide_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "claude-sonnet-4-20250514" in text
        assert "anthropic" in text
        assert "default" in text
        assert "12.4K/200K" in text
        assert "6%" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" not in text

    def test_input_height_counts_wide_characters_using_cell_width(self):
        cli_obj = _make_cli()

        class _Doc:
            lines = ["你" * 10]

        class _Buffer:
            document = _Doc()

        input_area = SimpleNamespace(buffer=_Buffer())

        def _input_height():
            try:
                from prompt_toolkit.application import get_app
                from prompt_toolkit.utils import get_cwidth

                doc = input_area.buffer.document
                prompt_width = max(2, get_cwidth(cli_obj._get_tui_prompt_text()))
                try:
                    available_width = get_app().output.get_size().columns - prompt_width
                except Exception:
                    import shutil
                    available_width = shutil.get_terminal_size((80, 24)).columns - prompt_width
                if available_width < 10:
                    available_width = 40
                visual_lines = 0
                for line in doc.lines:
                    line_width = get_cwidth(line)
                    if line_width <= 0:
                        visual_lines += 1
                    else:
                        visual_lines += max(1, -(-line_width // available_width))
                return min(max(visual_lines, 1), 8)
            except Exception:
                return 1

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=14)
        with patch.object(HermesCLI, "_get_tui_prompt_text", return_value="❯ "), \
             patch("prompt_toolkit.application.get_app", return_value=mock_app):
            assert _input_height() == 2

    def test_input_height_uses_prompt_toolkit_width_over_shutil(self):
        cli_obj = _make_cli()

        class _Doc:
            lines = ["你" * 10]

        class _Buffer:
            document = _Doc()

        input_area = SimpleNamespace(buffer=_Buffer())

        def _input_height():
            try:
                from prompt_toolkit.application import get_app
                from prompt_toolkit.utils import get_cwidth

                doc = input_area.buffer.document
                prompt_width = max(2, get_cwidth(cli_obj._get_tui_prompt_text()))
                try:
                    available_width = get_app().output.get_size().columns - prompt_width
                except Exception:
                    import shutil
                    available_width = shutil.get_terminal_size((80, 24)).columns - prompt_width
                if available_width < 10:
                    available_width = 40
                visual_lines = 0
                for line in doc.lines:
                    line_width = get_cwidth(line)
                    if line_width <= 0:
                        visual_lines += 1
                    else:
                        visual_lines += max(1, -(-line_width // available_width))
                return min(max(visual_lines, 1), 8)
            except Exception:
                return 1

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=14)
        with patch.object(HermesCLI, "_get_tui_prompt_text", return_value="❯ "), \
             patch("prompt_toolkit.application.get_app", return_value=mock_app), \
             patch("shutil.get_terminal_size") as mock_shutil:
            assert _input_height() == 2
        mock_shutil.assert_not_called()

    def test_build_status_bar_text_no_cost_in_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=5000,
            total_tokens=15000,
            api_calls=7,
            context_tokens=50000,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)
        assert "$" not in text  # cost is never shown in status bar

    def test_build_status_bar_text_collapses_for_narrow_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=2400,
            total_tokens=12400,
            api_calls=7,
            context_tokens=12400,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=60)

        assert "⚕" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" not in text
        assert "200K" not in text

    def test_build_status_bar_text_handles_missing_agent(self):
        cli_obj = _make_cli()

        text = cli_obj._build_status_bar_text(width=100)

        assert "⚕" in text
        assert "claude-sonnet-4-20250514" in text

    def test_status_bar_primary_line_shows_provider_and_reasoning_next_to_model(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )
        cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_fragments()

        text = "".join(text for _, text in fragments)
        assert "claude-sonnet-4-20250514" in text
        assert "anthropic" in text
        assert "high" in text
        assert "elapsed 15m" not in text
        assert "session" not in text

    def test_status_bar_primary_line_uses_named_custom_provider_and_task_progress(self):
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
        cli_obj.provider = "custom"
        cli_obj.requested_provider = "lucienfc-gpt"
        cli_obj._provider_source = "custom_provider:lucienfc-gpt"
        cli_obj.agent.provider = "custom"
        cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_fragments()

        text = "".join(text for _, text in fragments)
        assert "DeepSeek-R1" in text
        assert "lucienfc-gpt" in text
        assert "custom" not in text
        assert "(2/3)" in text

    def test_status_bar_primary_line_shows_background_task_counts_from_cli_tracking(self):
        cli_obj = _attach_background_tasks(
            _attach_agent(
                _make_cli(model="DeepSeek-R1"),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=12_450,
                context_length=200_000,
            ),
            total=5,
            active=3,
        )
        cli_obj.provider = "custom"
        cli_obj.requested_provider = "lucienfc-gpt"
        cli_obj._provider_source = "custom_provider:lucienfc-gpt"
        cli_obj.agent.provider = "custom"
        cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_fragments()

        text = "".join(text for _, text in fragments)
        assert "DeepSeek-R1" in text
        assert "lucienfc-gpt" in text
        assert "后台任务：2/5" in text

    def test_status_bar_background_task_label_hidden_when_no_background_tasks_started(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )
        cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_fragments()

        text = "".join(text for _, text in fragments)
        assert "后台任务：" not in text

    @pytest.mark.parametrize("provider_name", ["lucienfc-gpt", "other-custom-provider"])
    def test_status_bar_primary_line_uses_pool_backed_custom_provider_name(self, provider_name):
        cli_obj = _attach_agent(
            _make_cli(model="gpt-5.4"),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )
        cli_obj.provider = "custom"
        cli_obj.requested_provider = "custom"
        cli_obj._provider_source = f"pool:custom:{provider_name}"
        cli_obj.agent.provider = "custom"
        cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_fragments()

        text = "".join(text for _, text in fragments)
        assert "gpt-5.4" in text
        assert provider_name in text
        assert " custom " not in text

    @pytest.mark.parametrize(
        ("base_url", "inferred_source", "expected_name"),
        [
            ("https://gpt.lucienfc.eu.org/v1", "custom:lucienfc-gpt", "lucienfc-gpt"),
            (
                "https://other-provider.example/v1",
                "custom:other-custom-provider",
                "other-custom-provider",
            ),
        ],
    )
    def test_status_bar_provider_label_infers_named_custom_provider_from_base_url_before_first_turn(
        self,
        base_url,
        inferred_source,
        expected_name,
    ):
        cli_obj = _make_cli(model="gpt-5.4")
        cli_obj.provider = "custom"
        cli_obj.requested_provider = "custom"
        cli_obj.base_url = base_url
        cli_obj._provider_source = None

        with patch(
            "agent.credential_pool.get_custom_provider_pool_key",
            return_value=inferred_source,
        ) as mock_lookup:
            assert cli_obj._get_status_bar_provider_label() == expected_name

        mock_lookup.assert_called_once_with(base_url)

    def test_minimal_tui_chrome_threshold(self):
        cli_obj = _make_cli()

        assert cli_obj._use_minimal_tui_chrome(width=63) is True
        assert cli_obj._use_minimal_tui_chrome(width=64) is False

    def test_bottom_input_rule_hides_on_narrow_terminals(self):
        cli_obj = _make_cli()

        assert cli_obj._tui_input_rule_height("top", width=50) == 1
        assert cli_obj._tui_input_rule_height("bottom", width=50) == 0
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 1

    def test_agent_spacer_reclaimed_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._agent_running = True

        assert cli_obj._agent_spacer_height(width=50) == 0
        assert cli_obj._agent_spacer_height(width=90) == 1
        cli_obj._agent_running = False
        assert cli_obj._agent_spacer_height(width=90) == 0

    def test_spinner_line_hidden_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "thinking"

        assert cli_obj._spinner_widget_height(width=50) == 0
        assert cli_obj._spinner_widget_height(width=90) == 1
        cli_obj._spinner_text = ""
        assert cli_obj._spinner_widget_height(width=90) == 0

    def test_spinner_line_visible_while_agent_working_before_first_spinner_text(self):
        cli_obj = _make_cli()
        cli_obj._agent_running = True

        assert cli_obj._spinner_widget_height(width=50) == 0
        assert cli_obj._spinner_widget_height(width=90) == 1

    def test_agent_working_frame_pulses_deterministically(self):
        cli_obj = _make_cli()

        assert cli_obj._agent_working_frame(now=0.00) == "·"
        assert cli_obj._agent_working_frame(now=0.13) == "•"
        assert cli_obj._agent_working_frame(now=0.26) == "●"
        assert cli_obj._agent_working_frame(now=0.39) == "•"

    def test_on_tool_progress_styles_todo_transcript_with_accent_prefix(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = ""
        cli_obj._tool_start_time = 0.0
        cli_obj._voice_mode = False
        cli_obj._invalidate = MagicMock()
        cli_obj._stream_box_opened = False
        cli_obj._reasoning_box_opened = False

        with patch("cli._cprint") as mock_cprint:
            cli_obj._on_tool_progress(
                "tool.started",
                "todo",
                "planning 3 task(s)",
                {"todos": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]},
            )

        assert mock_cprint.call_count == 1
        assert mock_cprint.call_args.args[0] == f"  ┊ {_ACCENT}📋 planning 3 task(s){_RST}"
        assert cli_obj._spinner_text.endswith("planning 3 task(s)")
        cli_obj._invalidate.assert_called_once()

    @pytest.mark.parametrize(
        ("function_name", "preview", "function_args"),
        [
            ("memory", '+user: \"prefers dark mode\"', {"action": "add", "target": "user"}),
            ("session_search", 'recall: \"tmux hang\"', {"query": "tmux hang"}),
        ],
    )
    def test_on_tool_progress_persists_transcript_for_other_fast_tools(self, function_name, preview, function_args):
        cli_obj = _make_cli()
        cli_obj._spinner_text = ""
        cli_obj._tool_start_time = 0.0
        cli_obj._voice_mode = False
        cli_obj._invalidate = MagicMock()
        cli_obj._stream_box_opened = False
        cli_obj._reasoning_box_opened = False

        with patch("cli._cprint") as mock_cprint:
            cli_obj._on_tool_progress("tool.started", function_name, preview, function_args)

        assert mock_cprint.call_count == 1
        assert preview in mock_cprint.call_args.args[0]
        assert str(_ACCENT) not in mock_cprint.call_args.args[0]
        assert cli_obj._spinner_text.endswith(preview)
        cli_obj._invalidate.assert_called_once()

    @pytest.mark.parametrize(
        ("function_name", "preview", "function_args"),
        [
            ("terminal", "$ pwd", {"command": "pwd"}),
            ("read_file", "/etc/hosts", {"path": "/etc/hosts"}),
        ],
    )
    def test_on_tool_progress_keeps_regular_tools_footer_only(self, function_name, preview, function_args):
        cli_obj = _make_cli()
        cli_obj._spinner_text = ""
        cli_obj._tool_start_time = 0.0
        cli_obj._voice_mode = False
        cli_obj._invalidate = MagicMock()

        with patch("cli._cprint") as mock_cprint:
            cli_obj._on_tool_progress("tool.started", function_name, preview, function_args)

        mock_cprint.assert_not_called()
        assert cli_obj._spinner_text.endswith(preview)
        cli_obj._invalidate.assert_called_once()

    def test_on_tool_progress_flushes_open_stream_before_persisting_fast_tool(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = ""
        cli_obj._tool_start_time = 0.0
        cli_obj._voice_mode = False
        cli_obj._invalidate = MagicMock()
        cli_obj._stream_box_opened = True
        cli_obj._stream_started = True
        cli_obj._reasoning_box_opened = False
        cli_obj._flush_stream = MagicMock()
        cli_obj._reset_stream_state = MagicMock()

        with patch("cli._cprint") as mock_cprint:
            cli_obj._on_tool_progress("tool.started", "todo", "planning 1 task(s)", {"todos": [{"id": "t1"}]})

        cli_obj._flush_stream.assert_called_once()
        cli_obj._reset_stream_state.assert_called_once()
        assert mock_cprint.call_count == 1

    def test_voice_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = True
        cli_obj._voice_continuous = True

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status", " 🎤 Ctrl+B ")]

    def test_voice_recording_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = True
        cli_obj._voice_processing = False

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status-recording", " ● REC ")]

    def test_status_bar_secondary_line_shows_context_details_on_wide_terminal(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=12_450,
                context_length=200_000,
                compressions=1,
            ),
            ("build", "completed"),
            ("deploy", "in_progress"),
            ("docs", "pending"),
        )
        cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_secondary_fragments()

        text = "".join(text for _, text in fragments)
        assert "[x] build" not in text
        assert "[>] deploy" in text
        assert "[ ] docs" not in text
        assert "todo" not in text
        assert "high" not in text
        assert "12.4K/200K" in text
        assert "6%" in text
        assert "cmp 1" in text
        assert "[" in text and "]" in text

    def test_status_bar_secondary_line_compacts_on_narrow_terminal(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=10_230,
                context_length=200_000,
            ),
            ("b", "completed"),
            ("d", "in_progress"),
            ("doc", "pending"),
        )
        cli_obj.reasoning_config = {"enabled": True, "effort": "medium"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=70)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_secondary_fragments()

        text = "".join(text for _, text in fragments)
        assert "[x] b" not in text
        assert "[>] d" in text
        assert "[ ] doc" not in text
        assert "med" not in text
        assert "10.2K/200K" not in text

    def test_status_bar_secondary_line_prioritizes_full_active_task_over_context_details(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=12_450,
                context_length=200_000,
                compressions=1,
            ),
            ("Load relevant Hermes skill/instructions for local CLI status bar maintenance", "in_progress"),
            ("Inspect Hermes status bar rendering code", "pending"),
        )
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=90)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_secondary_fragments()

        text = "".join(text for _, text in fragments)
        assert "[>] Load relevant Hermes skill/instructions for local CLI status bar maintenance" in text
        assert "12.4K/200K" not in text
        assert "[█" not in text

    def test_status_bar_secondary_line_uses_available_width_for_long_task_on_narrow_terminal(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=12_450,
                context_length=200_000,
            ),
            ("Load relevant Hermes skill/instructions for local CLI status bar maintenance", "in_progress"),
        )
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=70)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_secondary_fragments()

        text = "".join(text for _, text in fragments)
        assert "skill/instructions" in text

    def test_status_bar_primary_line_keeps_progress_visible_with_octopus_provider_on_medium_width(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(model="claude-sonnet-4-20250514"),
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
        cli_obj.provider = "custom"
        cli_obj.requested_provider = "custom"
        cli_obj._provider_source = "custom_provider:octopus-gpt"
        cli_obj.agent.provider = "custom"
        cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=50)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_fragments()

        text = "".join(text for _, text in fragments)
        assert "claude-sonnet-4-20250514" in text
        assert "(2/3)" in text

    def test_status_bar_secondary_line_shows_task_name_instead_of_todo_summary_when_no_active_task(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=12_450,
                context_length=200_000,
            ),
            ("build", "completed"),
            ("deploy", "pending"),
            ("docs", "pending"),
        )
        cli_obj._tmux_session_task_title = "调查状态栏任务进度显示问题"
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=71)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_secondary_fragments()

        text = "".join(text for _, text in fragments)
        assert "调查状态栏任务进度显示问题" in text
        assert "todo 1/3" not in text
        assert "td 1/3" not in text
        assert "active --" not in text

    def test_status_bar_secondary_line_falls_back_to_specific_todo_name_when_no_saved_title(self):
        cli_obj = _attach_todo_items(
            _attach_agent(
                _make_cli(),
                prompt_tokens=10_230,
                completion_tokens=2_220,
                total_tokens=12_450,
                api_calls=7,
                context_tokens=12_450,
                context_length=200_000,
            ),
            ("build", "completed"),
            ("deploy", "pending"),
            ("docs", "pending"),
        )
        cli_obj._status_bar_visible = True

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=71)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            fragments = cli_obj._get_status_bar_secondary_fragments()

        text = "".join(text for _, text in fragments)
        assert "[ ] deploy" in text
        assert "Hermes" not in text
        assert "todo 1/3" not in text
        assert "td 1/3" not in text
        assert "active --" not in text


class TestCLITmuxTaskTitle:
    def test_summarize_prompt_task_filters_domains_and_keeps_core_theme(self):
        cli_obj = _make_cli()

        title = cli_obj._summarize_prompt_task(
            "tmux在屏幕右下角的标题永远是lucienfc.con，我希望可以实时显示当前任务的标题，"
            "注意不是todo的任务进度，而是总结prompt下达的任务"
        )

        assert title == "实时显示当前任务的标题 / 总结prompt下达的任务"
        assert "lucienfc.con" not in title

    def test_get_prompt_task_title_skips_modifier_only_followups(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = [
            {
                "role": "user",
                "content": "tmux在屏幕右下角的标题永远是lucienfc.con，我希望可以实时显示当前任务的标题，注意不是todo的任务进度，而是总结prompt下达的任务",
            },
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "另外不需要显示时间和日期"},
        ]

        assert cli_obj._get_prompt_task_title() == "实时显示当前任务的标题 / 总结prompt下达的任务"

    def test_summarize_prompt_task_prefers_requested_change_over_complaint_text(self):
        cli_obj = _make_cli()

        title = cli_obj._summarize_prompt_task(
            "做一个修改，在总结标题时，不应该显示网址，因为会占用太多空间。"
            "另外，目前显示的标题“另外不需要显示时间和日期”没有意义，对总结这个session当前做的任务主题没有帮助"
        )

        assert title == "在总结标题时 / 不应该显示网址"

    def test_get_prompt_task_title_summarizes_multiline_prompt(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = []

        title = cli_obj._get_prompt_task_title("修复 tmux 右下角标题\n不要显示时间和日期\n实时显示当前任务")

        assert title == "修复 tmux 右下角标题 / 实时显示当前任务"

    def test_summarize_prompt_task_strips_observation_prefixes_and_keeps_short_task_name(self):
        cli_obj = _make_cli()

        title = cli_obj._summarize_prompt_task(
            "做个修改：我注意到现在右下角标题有点长，希望右下角只保留任务名，不要这些口头前缀"
        )

        assert title == "右下角只保留任务名"

    def test_summarize_prompt_task_prefers_cleanup_request_over_background_complaint(self):
        cli_obj = _make_cli()

        title = cli_obj._summarize_prompt_task(
            "我注意到似乎更新这个标题会在后台多开一个hermes实例，我希望有一个资源回收机制，避免被僵尸后台浪费系统资源"
        )

        assert title == "资源回收机制"

    def test_summarize_prompt_task_rejects_context_compaction_summary_prefix(self):
        cli_obj = _make_cli()

        title = cli_obj._summarize_prompt_task(
            "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted to save context space. "
            "The summary below describes work that was already completed, and the current session state may still reflect that work."
        )

        assert title == ""

    def test_summarize_prompt_task_rejects_python_mock_repr(self):
        cli_obj = _make_cli()

        title = cli_obj._summarize_prompt_task(
            "<MagicMock name='_session_title()' id='133736162728528'>"
        )

        assert title == ""

    def test_get_saved_tmux_task_title_skips_compaction_history_and_prefers_session_title(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = [
            {
                "role": "user",
                "content": "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted to save context space. "
                "The summary below describes work that was already completed.",
            }
        ]
        cli_obj._session_db = MagicMock()
        cli_obj._session_db.get_session_title.return_value = "检查hermes4实例上下文压缩 #5"

        assert cli_obj._get_saved_tmux_task_title() == "检查hermes4实例上下文压缩 #5"

    def test_get_prompt_task_title_locks_to_first_prompt_per_session(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = []
        cli_obj._session_db = MagicMock()
        cli_obj._session_db.get_session_title.return_value = None

        first_title = cli_obj._get_prompt_task_title("tmux在右下角实时显示当前任务标题，不要显示时间和日期")
        cli_obj.conversation_history.append({"role": "user", "content": "tmux在右下角实时显示当前任务标题，不要显示时间和日期"})
        cli_obj._session_db.get_session_title.return_value = first_title

        followup_title = cli_obj._get_prompt_task_title("另外加一个资源回收机制，避免僵尸后台")

        assert first_title == "tmux在右下角实时显示当前任务标题"
        assert followup_title == first_title
        cli_obj._session_db.set_session_title.assert_called_once_with("session-1", first_title)

    def test_sync_tmux_pane_title_updates_current_pane_once_per_session(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = []

        with patch.dict("os.environ", {"TMUX_PANE": "%9"}, clear=False), patch("subprocess.run") as mock_run:
            title = cli_obj._sync_tmux_pane_title("tmux在右下角实时显示当前任务标题，不要显示时间和日期")
            cli_obj.conversation_history.append({"role": "user", "content": "tmux在右下角实时显示当前任务标题，不要显示时间和日期"})
            followup = cli_obj._sync_tmux_pane_title("另外加一个资源回收机制，避免僵尸后台")

        assert title == "tmux在右下角实时显示当前任务标题"
        assert followup == title
        # Startup can sync the title once, then the first user prompt locks it.
        # We only require that follow-up prompts do not change the title.
        assert mock_run.call_count >= 1
        args, kwargs = mock_run.call_args
        assert args[0] == ["tmux", "select-pane", "-t", "%9", "-T", title]
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 1.0

    def test_sync_tmux_pane_title_does_not_clobber_existing_title_with_fallback(self):
        """Regression: restarting Hermes inside the same tmux pane should not
        overwrite an existing meaningful title with the fallback string.

        This covers the startup path where Hermes calls _sync_tmux_pane_title()
        before any user prompt is sent.
        """
        cli_obj = _make_cli()
        cli_obj.conversation_history = []
        cli_obj._tmux_pane_title = "已有标题"

        with patch.dict("os.environ", {"TMUX_PANE": "%9"}, clear=False), patch("subprocess.run") as mock_run:
            title = cli_obj._sync_tmux_pane_title(task_source=None, fallback="Hermes", force=False)

        assert title == "已有标题"
        assert mock_run.call_count == 0

    def test_prime_tmux_pane_title_cache_preserves_visible_title_without_locking_new_session(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = []

        with patch.dict(
            "os.environ",
            {"TMUX": "/tmp/tmux-1000/default,1234,0", "TMUX_PANE": "%9"},
            clear=False,
        ), patch("subprocess.check_output", return_value="旧任务标题\n"):
            primed = cli_obj._prime_tmux_pane_title_cache(fallback="Hermes")

        assert primed == "旧任务标题"
        assert cli_obj._tmux_pane_title == "旧任务标题"
        assert cli_obj._tmux_session_task_title == ""
        assert cli_obj._tmux_task_title_locked is False

        with patch.dict("os.environ", {"TMUX_PANE": "%9"}, clear=False), patch("subprocess.run") as mock_run:
            title = cli_obj._sync_tmux_pane_title("新的任务标题")

        assert title == "新的任务标题"
        assert cli_obj._tmux_session_task_title == "新的任务标题"
        assert cli_obj._tmux_task_title_locked is True
        args, kwargs = mock_run.call_args
        assert args[0] == ["tmux", "select-pane", "-t", "%9", "-T", "新的任务标题"]
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 1.0

    def test_tmux_title_enforces_30_chinese_char_limit_by_width(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = []

        # 31 Chinese characters (CJK width≈62 columns) should be truncated.
        raw_title = "一" * 31
        with patch.dict("os.environ", {"TMUX_PANE": "%9"}, clear=False), patch("subprocess.run") as mock_run:
            title = cli_obj._sync_tmux_pane_title(raw_title)

        assert title != raw_title
        assert cli_obj._title_display_width(title) <= 60
        assert mock_run.call_count == 1

    def test_tmux_title_preserves_two_clauses_when_truncating(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = []

        long_two_clause = (
            "修复tmux右下角标题总结显示策略" * 3
            + " / "
            + "限制长度30个中文字符以内并保持信息完整" * 2
        )
        with patch.dict("os.environ", {"TMUX_PANE": "%9"}, clear=False), patch("subprocess.run"):
            title = cli_obj._sync_tmux_pane_title(long_two_clause)

        assert " / " in title
        assert cli_obj._title_display_width(title) <= 60


class TestCLIUsageReport:
    def test_show_usage_includes_estimated_cost(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=1,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Model:" in output
        assert "Cost status:" in output
        assert "Cost source:" in output
        assert "Total cost:" in output
        assert "$" in output
        assert "0.064" in output
        assert "Session duration:" in output
        assert "Compressions:" in output

    def test_show_usage_marks_unknown_pricing(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(model="local/my-custom-model"),
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            api_calls=1,
            context_tokens=1_000,
            context_length=32_000,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Total cost:" in output
        assert "n/a" in output
        assert "Pricing unknown for local/my-custom-model" in output

    def test_zero_priced_provider_models_stay_unknown(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(model="glm-5"),
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            api_calls=1,
            context_tokens=1_000,
            context_length=32_000,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Total cost:" in output
        assert "n/a" in output
        assert "Pricing unknown for glm-5" in output


class TestStatusBarWidthSource:
    """Ensure status bar fragments don't overflow the terminal width."""

    def _make_wide_cli(self):
        from datetime import datetime, timedelta
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=100_000,
            completion_tokens=5_000,
            total_tokens=105_000,
            api_calls=20,
            context_tokens=100_000,
            context_length=200_000,
        )
        cli_obj._status_bar_visible = True
        return cli_obj

    def test_fragments_fit_within_announced_width(self):
        """Total fragment text length must not exceed the width used to build them."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        for width in (40, 52, 76, 80, 120, 200):
            mock_app = MagicMock()
            mock_app.output.get_size.return_value = MagicMock(columns=width)

            with patch("prompt_toolkit.application.get_app", return_value=mock_app):
                frags = cli_obj._get_status_bar_fragments()

            total_text = "".join(text for _, text in frags)
            display_width = cli_obj._status_bar_display_width(total_text)
            assert display_width <= width + 4, (  # +4 for minor padding chars
                f"At width={width}, fragment total {display_width} cells overflows "
                f"({total_text!r})"
            )

    def test_fragments_use_pt_width_over_shutil(self):
        """When prompt_toolkit reports a width, shutil.get_terminal_size must not be used."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app) as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            cli_obj._get_status_bar_fragments()

        mock_shutil.assert_not_called()

    def test_fragments_fall_back_to_shutil_when_no_app(self):
        """Outside a TUI context (no running app), shutil must be used as fallback."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app", side_effect=Exception("no app")), \
             patch("shutil.get_terminal_size", return_value=MagicMock(columns=100)) as mock_shutil:
            frags = cli_obj._get_status_bar_fragments()

        mock_shutil.assert_called()
        assert len(frags) > 0

    def test_build_status_bar_text_uses_pt_width(self):
        """_build_status_bar_text() must also prefer prompt_toolkit width."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=80)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app), \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text()  # no explicit width

        mock_shutil.assert_not_called()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_explicit_width_skips_pt_lookup(self):
        """An explicit width= argument must bypass both PT and shutil lookups."""
        from unittest.mock import patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app") as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text(width=100)

        mock_get_app.assert_not_called()
        mock_shutil.assert_not_called()
        assert len(text) > 0
