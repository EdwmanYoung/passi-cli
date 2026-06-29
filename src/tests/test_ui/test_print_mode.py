"""Tests for passi.ui.print_mode.

Covers the non-interactive single-query output path without requiring a real
LLM, Runtime session directory, or R environment.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from passi.soul.protocol import AgentMessage
from passi.ui.print_mode import (
    _print_json,
    _print_markdown,
    _print_text,
    _safe_print,
    run_print_mode,
    run_print_mode_sync,
)


class TestSafePrint:
    """Unicode-safe stdout printing."""

    def test_safe_print_outputs_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Normal text is printed unchanged."""
        _safe_print("hello")
        captured = capsys.readouterr()
        assert captured.out == "hello\n"

    def test_safe_print_fallback_on_unicode_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """When stdout raises UnicodeEncodeError, re-encode with replacement."""

        class BadStdout:
            encoding = "ascii"

            def write(self, text: str) -> int:
                if "\u4e16" in text:
                    raise UnicodeEncodeError("ascii", text, 0, 1, "bad char")
                return 0

            def flush(self) -> None:
                pass

        with patch.object(__import__("sys"), "stdout", BadStdout()):
            # Should not raise
            _safe_print("hello \u4e16 world")


class TestPrintText:
    """Plain-text response formatting."""

    def test_print_text_plain_string(self, capsys: pytest.CaptureFixture[str]) -> None:
        """String content is printed as-is."""
        response = AgentMessage(role="assistant", content="hello world")
        _print_text(response)
        assert "hello world" in capsys.readouterr().out

    def test_print_text_text_blocks(self, capsys: pytest.CaptureFixture[str]) -> None:
        """List of text blocks prints each text block."""
        response = AgentMessage(
            role="assistant",
            content=[{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}],
        )
        _print_text(response)
        out = capsys.readouterr().out
        assert "line 1" in out
        assert "line 2" in out

    def test_print_text_tool_use_block(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Tool_use blocks render as [Tool: name]."""
        response = AgentMessage(
            role="assistant",
            content=[{"type": "tool_use", "name": "read_file", "input": {}}],
        )
        _print_text(response)
        assert "[Tool: read_file]" in capsys.readouterr().out

    def test_print_text_pending_question(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Pending question metadata renders question, context, and options."""
        response = AgentMessage(
            role="assistant",
            content="",
            metadata={
                "pending_question": {
                    "question": "Which method?",
                    "context": "Choose a DE method",
                    "options": ["DESeq2", "edgeR"],
                }
            },
        )
        _print_text(response)
        out = capsys.readouterr().out
        assert "Agent needs your input" in out
        assert "Which method?" in out
        assert "Choose a DE method" in out
        assert "1. DESeq2" in out
        assert "2. edgeR" in out


class TestPrintJson:
    """JSON response formatting."""

    def test_print_json_outputs_valid_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Response is serialized to JSON with role, content, metadata."""
        response = AgentMessage(
            role="assistant",
            content="hello",
            metadata={"key": "value"},
        )
        _print_json(response)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["role"] == "assistant"
        assert data["content"] == "hello"
        assert data["metadata"] == {"key": "value"}


class TestPrintMarkdown:
    """Markdown response formatting."""

    def test_print_markdown_string(self, capsys: pytest.CaptureFixture[str]) -> None:
        """String content is printed."""
        response = AgentMessage(role="assistant", content="# Title")
        _print_markdown(response)
        assert "# Title" in capsys.readouterr().out

    def test_print_markdown_text_blocks(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Text blocks are printed with blank lines between them."""
        response = AgentMessage(
            role="assistant",
            content=[
                {"type": "text", "text": "paragraph 1"},
                {"type": "text", "text": "paragraph 2"},
            ],
        )
        _print_markdown(response)
        out = capsys.readouterr().out
        assert "paragraph 1" in out
        assert "paragraph 2" in out


class TestRunPrintMode:
    """End-to-end print mode orchestration."""

    @pytest.mark.asyncio
    async def test_run_print_mode_success(self) -> None:
        """Successful query returns exit code 0 and prints text output."""
        fake_response = AgentMessage(role="assistant", content="Done")

        with patch("passi.ui.print_mode.Runtime") as mock_runtime_cls:
            mock_runtime = MagicMock()
            mock_runtime_cls.return_value = mock_runtime
            mock_agent = AsyncMock()
            mock_agent.chat = AsyncMock(return_value=fake_response)
            mock_agent.initialize = AsyncMock()
            mock_agent.shutdown = AsyncMock()

            with patch("passi.ui.print_mode.PassiAgent", return_value=mock_agent):
                exit_code = await run_print_mode("analyze counts", output_format="text")

        assert exit_code == 0
        mock_runtime.session.create_session.assert_called_once_with(domain="multi-omics")
        mock_agent.initialize.assert_awaited_once()
        mock_agent.chat.assert_awaited_once_with("analyze counts")
        mock_agent.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_print_mode_error_returns_one(self) -> None:
        """Unhandled exception returns exit code 1 and still shuts down."""
        with patch("passi.ui.print_mode.Runtime") as mock_runtime_cls:
            mock_runtime = MagicMock()
            mock_runtime_cls.return_value = mock_runtime
            mock_agent = AsyncMock()
            mock_agent.chat = AsyncMock(side_effect=RuntimeError("boom"))
            mock_agent.initialize = AsyncMock()
            mock_agent.shutdown = AsyncMock()

            with patch("passi.ui.print_mode.PassiAgent", return_value=mock_agent):
                exit_code = await run_print_mode("analyze counts")

        assert exit_code == 1
        mock_agent.shutdown.assert_awaited_once()

    def test_run_print_mode_sync(self) -> None:
        """Synchronous wrapper delegates to async implementation."""
        with patch("passi.ui.print_mode.run_print_mode", new=AsyncMock(return_value=0)) as mock_run:
            result = run_print_mode_sync("query", output_format="json", domain="genomics")

        assert result == 0
        mock_run.assert_awaited_once_with("query", None, "json", "genomics")
