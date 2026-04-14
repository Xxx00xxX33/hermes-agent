"""Tests for terminal/file tool availability in local dev environments."""

import importlib

from model_tools import get_tool_definitions

terminal_tool_module = importlib.import_module("tools.terminal_tool")


class TestTerminalRequirements:
    def test_local_backend_requirements(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        assert terminal_tool_module.check_terminal_requirements() is True

    def test_terminal_and_file_tools_resolve_for_local_backend(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )
        tools = get_tool_definitions(enabled_toolsets=["terminal", "file"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}
        assert "terminal" in names
        assert {"read_file", "write_file", "patch", "search_files"}.issubset(names)

    def test_terminal_tool_definition_exposes_background_guidance_for_local_backend(self, monkeypatch):
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "local"},
        )

        tools = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
        terminal_def = next(tool["function"] for tool in tools if tool["function"]["name"] == "terminal")
        combined = (
            terminal_def["description"]
            + "\n"
            + terminal_def["parameters"]["properties"]["background"]["description"]
        ).lower()

        assert "independent" in combined
        assert "notify_on_complete=true" in combined
        assert "short commands" in combined
        assert "dependency-blocking" in combined
        assert "needed right away" in combined
        assert any(term in combined for term in ("ram-constrained", "memory-heavy", "resource-heavy"))

    def test_terminal_and_execute_code_tools_resolve_for_managed_modal(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_ENABLE_NOUS_MANAGED_TOOLS", "1")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        monkeypatch.setattr(
            terminal_tool_module,
            "_get_env_config",
            lambda: {"env_type": "modal", "modal_mode": "managed"},
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "is_managed_tool_gateway_ready",
            lambda _vendor: True,
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "ensure_minisweagent_on_path",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
        )

        tools = get_tool_definitions(enabled_toolsets=["terminal", "code_execution"], quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}

        assert "terminal" in names
        assert "execute_code" in names
