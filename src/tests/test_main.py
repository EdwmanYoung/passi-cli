"""Tests for passi.main CLI entry points."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from passi.main import main


def test_version_flag() -> None:
    """--version prints the application version."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "passi" in result.output


def test_chat_help() -> None:
    """passi chat --help shows usage."""
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--help"])
    assert result.exit_code == 0
    assert "interactive" in result.output.lower()


def test_ask_command_json_output() -> None:
    """passi ask delegates to run_print_mode_sync and exits with its code."""
    runner = CliRunner()

    with patch("passi.ui.print_mode.run_print_mode_sync", return_value=0) as mock_run:
        result = runner.invoke(main, ["ask", "hello", "--format", "json", "--domain", "genomics"])

    assert result.exit_code == 0
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == "hello"
    assert call_args[0][2] == "json"
    assert call_args[0][3] == "genomics"


def test_afk_command_sets_afk_mode() -> None:
    """passi afk enables afk_mode in config before running print mode."""
    runner = CliRunner()

    with patch("passi.ui.print_mode.run_print_mode_sync", return_value=0) as mock_run:
        result = runner.invoke(main, ["afk", "analyze"])

    assert result.exit_code == 0
    config_passed = mock_run.call_args[0][1]
    assert config_passed.afk_mode is True


def test_session_list_empty() -> None:
    """passi session list shows empty message when no sessions."""
    runner = CliRunner()

    with patch("passi.infra.session.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.list_sessions.return_value = []
        mock_mgr_cls.return_value = mock_mgr
        result = runner.invoke(main, ["session", "list"])

    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_session_load_found() -> None:
    """passi session load prints session info when found."""
    runner = CliRunner()

    class FakeMeta:
        session_id = "sess-123"
        domain = "genomics"
        message_count = 5

    with patch("passi.infra.session.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.load_session.return_value = FakeMeta()
        mock_mgr_cls.return_value = mock_mgr
        result = runner.invoke(main, ["session", "load", "sess-123"])

    assert result.exit_code == 0
    assert "sess-123" in result.output
    assert "5 messages" in result.output


def test_session_load_not_found() -> None:
    """passi session load exits with error when session missing."""
    runner = CliRunner()

    with patch("passi.infra.session.SessionManager") as mock_mgr_cls:
        mock_mgr = MagicMock()
        mock_mgr.load_session.side_effect = FileNotFoundError()
        mock_mgr_cls.return_value = mock_mgr
        result = runner.invoke(main, ["session", "load", "missing"])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_tool_list() -> None:
    """passi tool list prints available tools."""
    runner = CliRunner()
    result = runner.invoke(main, ["tool", "list"])
    assert result.exit_code == 0
    assert "Available tools" in result.output


def test_tool_run_read_file(tmp_path) -> None:
    """passi tool run executes a tool directly with JSON params."""
    runner = CliRunner()
    test_file = tmp_path / "hello.txt"
    test_file.write_text("world", encoding="utf-8")

    result = runner.invoke(main, [
        "tool", "run", "read_file",
        json.dumps({"path": str(test_file)}),
    ])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["success"] is True


def test_tool_run_invalid_json() -> None:
    """passi tool run exits with error for invalid JSON params."""
    runner = CliRunner()
    result = runner.invoke(main, ["tool", "run", "read_file", "not-json"])

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


def test_knowledge_methods() -> None:
    """passi knowledge methods lists transcriptomics methods."""
    runner = CliRunner()
    result = runner.invoke(main, ["knowledge", "methods", "--domain", "transcriptomics"])
    assert result.exit_code == 0
    assert "Methods for domain" in result.output


def test_knowledge_formats() -> None:
    """passi knowledge formats lists formats."""
    runner = CliRunner()
    result = runner.invoke(main, ["knowledge", "formats", "--domain", "genomics"])
    assert result.exit_code == 0
    assert "Formats for domain" in result.output


def test_server_missing_dependencies() -> None:
    """passi server exits cleanly when web API deps are missing."""
    runner = CliRunner()

    with patch("passi.api.server.create_app", side_effect=ImportError("No module named 'fastapi'")):
        result = runner.invoke(main, ["server"])

    assert result.exit_code == 1
    assert "Web API dependencies not installed" in result.output
