import os
import sys
from unittest.mock import call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_cprint_sends_multiline_text_to_prompt_toolkit_line_by_line_without_extra_end_newlines():
    from cli import _cprint

    text = (
        "- 如果你愿意，下一步最有价值的是：\n"
        "  1. 我帮你审一遍这个 `user_profile`，区分：\n"
        "     - 应保留为‘真正长期偏好’\n"
        "     - 应迁移为‘项目/工作流记忆’\n"
        "  2. 或者我直接给你一版‘更干净的 user_profile 精简草案’\n"
        "  推荐：2。"
    )

    with patch("cli._pt_print") as mock_print, patch("cli._PT_ANSI", side_effect=lambda s: s):
        _cprint(text)

    assert mock_print.call_args_list == [
        call("- 如果你愿意，下一步最有价值的是：\n", end=""),
        call("  1. 我帮你审一遍这个 `user_profile`，区分：\n", end=""),
        call("     - 应保留为‘真正长期偏好’\n", end=""),
        call("     - 应迁移为‘项目/工作流记忆’\n", end=""),
        call("  2. 或者我直接给你一版‘更干净的 user_profile 精简草案’\n", end=""),
        call("  推荐：2。", end=""),
    ]


def test_cprint_preserves_blank_lines_when_splitting_multiline_text():
    from cli import _cprint

    with patch("cli._pt_print") as mock_print, patch("cli._PT_ANSI", side_effect=lambda s: s):
        _cprint("第一行\n\n第三行")

    assert mock_print.call_args_list == [
        call("第一行\n", end=""),
        call("\n", end=""),
        call("第三行", end=""),
    ]


def test_cprint_strips_prompt_toolkit_unsafe_private_mode_ansi_before_rendering():
    from cli import _cprint

    rich_like_output = "╭─ Hermes ╮\n\x1b[?7h\x1b[31m第一行\x1b[0m\n\x1b[?7h第二行\x1b[0m\n╰──╯"

    with patch("cli._pt_print") as mock_print, patch("cli._PT_ANSI", side_effect=lambda s: s):
        _cprint(rich_like_output)

    assert mock_print.call_args_list == [
        call("╭─ Hermes ╮\n", end=""),
        call("\x1b[31m第一行\x1b[0m\n", end=""),
        call("第二行\x1b[0m\n", end=""),
        call("╰──╯", end=""),
    ]


def test_chatconsole_print_preserves_rich_newline_semantics_for_single_calls():
    from cli import ChatConsole

    console = ChatConsole()

    with patch("cli._cprint") as mock_cprint:
        console.print("/help line")

    mock_cprint.assert_called_once()
    rendered = mock_cprint.call_args.args[0]
    assert "/help line" in rendered
    assert rendered.endswith("\n")


def test_cprint_routes_live_tui_output_to_active_cli_transcript_before_stdout_fallback():
    import cli as cli_mod

    class _FakeCLI:
        def __init__(self):
            self._app = object()
            self.calls = []

        def _append_tui_transcript(self, text):
            self.calls.append(text)
            return True

    fake_cli = _FakeCLI()
    previous = cli_mod._active_cli_ref
    cli_mod._active_cli_ref = fake_cli
    try:
        with patch("cli._pt_print") as mock_print:
            cli_mod._cprint("hello from tui\n")
    finally:
        cli_mod._active_cli_ref = previous

    assert fake_cli.calls == ["hello from tui\n"]
    mock_print.assert_not_called()


def test_cprint_line_adds_one_newline_for_line_oriented_tui_notices():
    from cli import _cprint_line

    with patch("cli._cprint") as mock_cprint:
        _cprint_line("notice")
        _cprint_line("already newline\n")
        _cprint_line()

    assert mock_cprint.call_args_list == [
        call("notice\n"),
        call("already newline\n"),
        call("\n"),
    ]
