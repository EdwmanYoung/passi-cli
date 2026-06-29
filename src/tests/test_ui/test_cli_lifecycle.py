"""Lifecycle and rendering tests for PassiCLI.

Covers shutdown, repaint, resize monitor, and session-load error paths that
are awkward to exercise through the main TUI test suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from passi.config import PassiConfig
from passi.infra.runtime import Runtime
from passi.soul.passi_agent import PassiAgent
from passi.ui.cli import PassiCLI


def _make_cli(tmp_path: Path) -> tuple[PassiCLI, MagicMock, Runtime]:
    """Create a PassiCLI with mocked console and agent."""
    cfg = PassiConfig(
        anthropic={"api_key": "test-key", "model": "claude-sonnet-4-6"},
        default_provider="anthropic",
        session={"sessions_dir": tmp_path / "sessions"},
        output_dir=tmp_path / "output",
        debug=True,
    )
    runtime = Runtime(config=cfg)
    cli = PassiCLI(cfg)

    mock_console = MagicMock()
    cli.console = mock_console

    runtime.session.create_session(domain="test-domain")
    agent = PassiAgent(runtime)
    agent._initialized = True
    agent._mode = "chat"
    agent.shutdown = AsyncMock()
    cli.agent = agent
    cli.runtime = runtime
    cli._domain = "test-domain"

    return cli, mock_console, runtime


class TestPassiCLIShutdown:
    """Clean shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_resize_task_and_shuts_down_agent(self, tmp_path: Path) -> None:
        """_shutdown cancels the resize monitor and awaits agent.shutdown()."""
        cli, mock_console, _ = _make_cli(tmp_path)
        task = asyncio.create_task(asyncio.sleep(10))
        cli._resize_task = task

        await cli._shutdown()
        await asyncio.sleep(0)  # let the event loop process the cancellation

        assert task.cancelled()
        cli.agent.shutdown.assert_awaited_once()
        mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown_without_agent_prints_message(self, tmp_path: Path) -> None:
        """_shutdown works when no agent is attached."""
        cli, mock_console, _ = _make_cli(tmp_path)
        cli.agent = None

        await cli._shutdown()

        mock_console.print.assert_called_with("\n[dim]Session ended.[/dim]")


class TestPassiCLIRepaint:
    """Buffered content re-rendering."""

    def test_repaint_renders_all_buffer_entry_types(self, tmp_path: Path) -> None:
        """_repaint clears and re-renders each supported entry type."""
        cli, mock_console, _ = _make_cli(tmp_path)

        cli._render_buffer = [
            ("banner", ()),
            ("user", ("hello",)),
            ("agent", ("response",)),
            ("tool_call", ("read_file", '{"path": "x"}')),
            ("tool_result", ("done",)),
            ("system", ("msg",)),
            ("error", ("boom",)),
            ("status_bar", ("status",)),
        ]

        cli._repaint()

        mock_console.clear.assert_called_once()
        assert mock_console.print.call_count == len(cli._render_buffer)


class TestPassiCLISessionLoadErrors:
    """Error handling in _load_existing_session."""

    @pytest.mark.asyncio
    async def test_load_missing_session_starts_new_session(self, tmp_path: Path) -> None:
        """FileNotFoundError during load falls back to a new session."""
        cli, mock_console, runtime = _make_cli(tmp_path)
        cli._start_new_session = AsyncMock()

        await cli._load_existing_session("nonexistent-session")

        cli._start_new_session.assert_awaited_once()
        mock_console._print_error = MagicMock()

    @pytest.mark.asyncio
    async def test_load_session_generic_error_starts_new_session(self, tmp_path: Path) -> None:
        """Generic exception during load falls back to a new session."""
        cli, mock_console, runtime = _make_cli(tmp_path)
        runtime.session.load_session = MagicMock(side_effect=ValueError("corrupted"))
        cli._start_new_session = AsyncMock()

        await cli._load_existing_session("bad-session")

        cli._start_new_session.assert_awaited_once()
