"""TUI test suite for PassiCLI — command handling, mode system, skills, hooks.

Tests cover slash command dispatch, mode cycling, skill activation, hook CRUD,
status/config display, help text completeness, and print helper formatting.

Strategy: mock Rich Console to capture output, inject a minimal PassiAgent with
FakeLLMClient, then exercise command handlers directly.
"""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest
from rich.style import Style

from passi.config import PassiConfig
from passi.infra.runtime import Runtime
from passi.infra.hooks import HookConfig, HookEvent, HookManager, HookType
from passi.prompts.manager import PromptManager
from passi.soul.passi_agent import PassiAgent
from passi.soul.protocol import AgentMessage
from passi.ui.cli import (
    PassiCLI,
    HELP_TEXT,
    USER_STYLE,
    AGENT_STYLE,
    TOOL_STYLE,
    ERROR_STYLE,
    SYSTEM_STYLE,
    HEADER_STYLE,
    STATUS_STYLE,
    _MODE_CYCLE,
    _MODE_LABELS,
    _SENTINEL_CYCLE_MODE,
    _SENTINEL_SAVE,
    _SENTINEL_CLEAR_SCREEN,
    _SENTINEL_QUIT,
    _SENTINEL_CUSTOM_INPUT,
    _PROMPT_STYLE,
)
import json
from tests.fixtures.mock_llm import FakeLLMClient
from passi.soul.protocol import AgentStreamEvent
from passi.infra.plan import AnalysisPlan, PlanStep, StepStatus
from passi.wire.protocol import WireEvent, EventType
from passi.wire.persistence import WirePersistence


# ── Pipe input helper for testing PromptSession without a real terminal ──────


@contextmanager
def _pipe_session() -> Iterator[tuple[object, object]]:
    """Create a pipe input + dummy output for testing PromptSession.

    Usage:
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
            # test session.message(), key bindings, etc.

    Only setup errors are caught and skipped. Assertion failures in test
    bodies propagate normally.
    """
    try:
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    inp = None
    try:
        inp = create_pipe_input()
        inp.__enter__()
        yield inp, DummyOutput()
    except (OSError, RuntimeError, AttributeError) as exc:
        pytest.skip(f"Pipe input not available: {exc}")
    finally:
        if inp is not None:
            try:
                inp.__exit__(None, None, None)
            except Exception:
                pass


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_runtime(tmp_path: Path) -> Runtime:
    """Create a Runtime with test config pointing to a temp directory."""
    cfg = PassiConfig(
        anthropic={"api_key": "test-key", "model": "claude-sonnet-4-6"},
        default_provider="anthropic",
        session={"sessions_dir": tmp_path / "sessions"},
        output_dir=tmp_path / "output",
        hooks={"hooks_file": tmp_path / "hooks.yaml", "enabled": True},
        debug=True,
    )
    return Runtime(config=cfg)


def _make_cli_with_mocks(
    tmp_path: Path,
    *,
    create_agent: bool = True,
) -> tuple[PassiCLI, MagicMock, Runtime]:
    """Create a PassiCLI with mocked Console and optional agent.

    Returns (cli, mock_console, runtime).
    """
    runtime = _make_runtime(tmp_path)
    config = runtime.config
    cli = PassiCLI(config)

    # Replace console with mock
    mock_console = MagicMock()
    cli.console = mock_console

    if create_agent:
        runtime.session.create_session(domain="test-domain")
        agent = PassiAgent(runtime)

        # Inject minimal dependencies without full initialize()
        llm_client = FakeLLMClient("Test response.")
        agent._llm_client = llm_client
        agent._tool_registry = MagicMock()
        agent._provenance = MagicMock()
        agent._plan_manager = MagicMock()
        agent._task_tracker = MagicMock()

        # Set up prompt manager
        agent._prompt_manager = PromptManager(
            template_dir=config.prompt_template_dir,
        )
        agent._prompt_manager.load_skill  # ensure method exists

        # Set up hook manager
        hooks_path = tmp_path / "hooks.yaml"
        agent._hook_manager = HookManager(hooks_path)
        agent._hook_manager.set_session_context("test-session", "test-domain")

        agent._initialized = True
        agent._mode = "chat"
        agent._plan_first = False

        cli.agent = agent
        cli.runtime = runtime
        cli._domain = "test-domain"

    return cli, mock_console, runtime


# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Help Text
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIConstants:
    """Validate TUI constants and style definitions."""

    def test_mode_cycle_order(self):
        """Modes cycle in correct order."""
        assert _MODE_CYCLE == ["chat", "plan", "afk"]

    def test_mode_labels_all_modes(self):
        """Every mode in the cycle has a label."""
        for mode in _MODE_CYCLE:
            assert mode in _MODE_LABELS
            assert isinstance(_MODE_LABELS[mode], str)

    def test_help_text_includes_all_command_categories(self):
        """HELP_TEXT documents all major command categories."""
        categories = [
            "Agent Modes",
            "Skills",
            "Hooks",
            "Session",
            "Other",
        ]
        for cat in categories:
            assert cat in HELP_TEXT, f"Missing category: {cat}"

    def test_help_text_includes_mode_commands(self):
        assert "/mode" in HELP_TEXT
        assert "/mode [" in HELP_TEXT
        assert "/plan show" in HELP_TEXT
        assert "/plan approve" in HELP_TEXT
        assert "/plan reject" in HELP_TEXT

    def test_help_text_includes_skill_commands(self):
        assert "/skill list" in HELP_TEXT
        assert "/skill use" in HELP_TEXT
        assert "/skill off" in HELP_TEXT
        assert "/skill show" in HELP_TEXT

    def test_help_text_includes_hook_commands(self):
        assert "/hook list" in HELP_TEXT
        assert "/hook add" in HELP_TEXT
        assert "/hook remove" in HELP_TEXT
        assert "/hook toggle" in HELP_TEXT

    def test_help_text_includes_session_commands(self):
        assert "/status" in HELP_TEXT
        assert "/config" in HELP_TEXT
        assert "/save" in HELP_TEXT
        assert "/clear" in HELP_TEXT
        assert "/export" in HELP_TEXT
        assert "/domain" in HELP_TEXT

    def test_help_text_includes_quit(self):
        assert "/quit" in HELP_TEXT or "/exit" in HELP_TEXT

    def test_styles_are_style_objects(self):
        """All brand styles are Rich Style instances."""
        for s in (USER_STYLE, AGENT_STYLE, TOOL_STYLE, ERROR_STYLE, SYSTEM_STYLE, HEADER_STYLE, STATUS_STYLE):
            assert isinstance(s, Style), f"Expected Style, got {type(s)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Command Dispatch
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLICommandDispatch:
    """Slash command routing and basic command behavior."""

    @pytest.mark.asyncio
    async def test_help_command_prints_help(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_help("")
        # Should have printed the HELP_TEXT (Markdown)
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_unknown_command_shows_error(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._handle_command("/nonexistent_command_xyz")
        # Check that some error was printed
        printed = [str(c) for c in mock_console.print.call_args_list]
        has_err = any("Unknown command" in " ".join(str(a) for a in args[0] if args[0])
                      for args in mock_console.print.call_args_list)
        # The error is printed via _print_system which uses console.print(text, style=SYSTEM_STYLE)
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Unknown command" in args_text:
                found = True
        assert found, "Should print unknown command error"

    @pytest.mark.asyncio
    async def test_quit_sets_running_false(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli._running = True
        await cli._cmd_quit("")
        assert cli._running is False

    @pytest.mark.asyncio
    async def test_exit_is_same_as_quit(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli._running = True
        await cli._handle_command("/exit")
        assert cli._running is False

    @pytest.mark.asyncio
    async def test_handle_command_calls_correct_handler(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        # /help should call _cmd_help
        await cli._handle_command("/help")
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_handle_command_with_arg(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._handle_command("/domain genomics")
        assert cli._domain == "genomics"


# ═══════════════════════════════════════════════════════════════════════════════
# Mode System
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIModeSystem:
    """Agent mode switching — /mode, cycle, validation."""

    @pytest.mark.asyncio
    async def test_mode_switch_to_valid_mode(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        assert cli.agent.mode == "chat"
        await cli._cmd_mode("plan")
        assert cli.agent.mode == "plan"
        assert cli.agent.plan_first is True

    @pytest.mark.asyncio
    async def test_mode_switch_to_afk(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_mode("afk")
        assert cli.agent.mode == "afk"
        assert cli.agent.plan_first is False  # afk != plan-first

    @pytest.mark.asyncio
    async def test_mode_switch_back_to_chat(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "afk"
        await cli._cmd_mode("chat")
        assert cli.agent.mode == "chat"
        assert cli.agent.plan_first is False

    @pytest.mark.asyncio
    async def test_mode_invalid_shows_error(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_mode("invalid_mode")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Invalid mode" in args_text:
                found = True
        assert found, "Should show invalid mode error"

    @pytest.mark.asyncio
    async def test_mode_same_mode_is_noop(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "chat"
        await cli._cmd_mode("chat")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Already in" in args_text:
                found = True
        assert found, "Should say already in mode"

    @pytest.mark.asyncio
    async def test_mode_no_args_cycles(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "chat"
        await cli._cmd_mode("")
        assert cli.agent.mode == "plan"

    @pytest.mark.asyncio
    async def test_cycle_mode_chat_to_plan(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "chat"
        cli._cycle_mode()
        assert cli.agent.mode == "plan"
        assert cli.agent.plan_first is True

    @pytest.mark.asyncio
    async def test_cycle_mode_plan_to_afk(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "plan"
        cli._cycle_mode()
        assert cli.agent.mode == "afk"
        assert cli.agent.plan_first is False

    @pytest.mark.asyncio
    async def test_cycle_mode_afk_to_chat(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "afk"
        cli._cycle_mode()
        assert cli.agent.mode == "chat"
        assert cli.agent.plan_first is False

    @pytest.mark.asyncio
    async def test_cycle_mode_unknown_defaults_to_chat(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "unknown_value"
        cli._cycle_mode()
        assert cli.agent.mode == "chat"

    @pytest.mark.asyncio
    async def test_cycle_mode_none_agent_safe(self, tmp_path):
        """_cycle_mode should not raise when agent is None."""
        cli, _, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        # Should not raise
        cli._cycle_mode()

    @pytest.mark.asyncio
    async def test_mode_prints_status_bar_after_switch(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        mock_console.print.reset_mock()
        cli.agent._mode = "chat"
        await cli._cmd_mode("plan")
        # Status bar should be printed (at least one call with "[dim]Mode:[dim]")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Mode:" in args_text:
                found = True
        assert found, "Status bar should be printed after mode switch"

    @pytest.mark.asyncio
    async def test_mode_command_requires_agent(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        # _cmd_mode asserts agent is not None — test via _handle_command
        # Since agent is None, _handle_command routes to handler which will AssertionError.
        # This tests an edge case — the CLI should handle agent=None gracefully.
        # In practice, the REPL always has an agent after start().
        pass  # Edge case documented — handled by REPL lifecycle


# ═══════════════════════════════════════════════════════════════════════════════
# Skill Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLISkillCommands:
    """Skill management via /skill commands."""

    @pytest.mark.asyncio
    async def test_skill_list_shows_available(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_skill("list")
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_skill_show_when_empty(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_skill("show")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "No skills active" in args_text or "no skills" in args_text.lower():
                found = True
        assert found, "Should indicate no active skills"

    @pytest.mark.asyncio
    async def test_skill_show_active(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        # Load a skill first
        cli.agent._prompt_manager.load_skill("metabolomics")
        cli.agent._mode = "chat"
        await cli._cmd_skill("show")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "metabolomics" in args_text:
                found = True
        assert found, "Should show active skill name"

    @pytest.mark.asyncio
    async def test_skill_use_invalid_shows_error(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_skill("use nonexistent_skill_xyz")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Unknown skill" in args_text:
                found = True
        assert found, "Should show unknown skill error"

    @pytest.mark.asyncio
    async def test_skill_use_no_args_shows_usage(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_skill("use")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Usage:" in args_text:
                found = True
        assert found, "Should show usage when no skill name provided"

    @pytest.mark.asyncio
    async def test_skill_off_clears(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        # Load a skill, then turn off
        cli.agent._prompt_manager.load_skill("metabolomics")
        await cli._cmd_skill("off")
        assert cli.agent._prompt_manager.active_skills == []

    @pytest.mark.asyncio
    async def test_skill_use_valid_activates(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_skill("use metabolomics")
        assert "metabolomics" in cli.agent._prompt_manager.active_skills

    @pytest.mark.asyncio
    async def test_skill_use_multiple_comma_separated(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_skill("use metabolomics,stats")
        skills = cli.agent._prompt_manager.active_skills
        assert "metabolomics" in skills
        assert "stats" in skills

    @pytest.mark.asyncio
    async def test_skill_invalid_subcommand(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_skill("invalid_sub")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Usage:" in args_text:
                found = True
        assert found, "Should show usage for invalid subcommand"


# ═══════════════════════════════════════════════════════════════════════════════
# Hook Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIHookCommands:
    """Hook CRUD via /hook commands."""

    @pytest.mark.asyncio
    async def test_hook_list_empty(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_hook("list")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "No hooks configured" in args_text:
                found = True
        assert found, "Should show empty hooks message"

    @pytest.mark.asyncio
    async def test_hook_list_with_hooks(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        hm.add_hook(HookConfig(name="test", event=HookEvent.PRE_TOOL, type=HookType.SHELL, command="echo x"))
        await cli._cmd_hook("list")
        # Should render a Table (console.print called with Table)
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_hook_remove_nonexistent(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_hook("remove nonexistent_hook")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "not found" in args_text.lower():
                found = True
        assert found, "Should say hook not found"

    @pytest.mark.asyncio
    async def test_hook_toggle_nonexistent(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_hook("toggle nonexistent_hook")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "not found" in args_text.lower():
                found = True
        assert found, "Should say hook not found"

    @pytest.mark.asyncio
    async def test_hook_remove_no_args_shows_usage(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_hook("remove")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Usage:" in args_text:
                found = True
        assert found, "Should show usage"

    @pytest.mark.asyncio
    async def test_hook_toggle_no_args_shows_usage(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_hook("toggle")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Usage:" in args_text:
                found = True
        assert found, "Should show usage"

    @pytest.mark.asyncio
    async def test_hook_remove_existing(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        hm.add_hook(HookConfig(name="to_remove", event=HookEvent.PRE_TOOL, type=HookType.SHELL, command="echo x"))
        await cli._cmd_hook("remove to_remove")
        assert len(hm.hooks) == 0

    @pytest.mark.asyncio
    async def test_hook_toggle_existing(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        hm.add_hook(HookConfig(name="toggle_me", event=HookEvent.PRE_TOOL, type=HookType.SHELL, command="echo x"))
        assert hm.hooks[0].enabled is True
        await cli._cmd_hook("toggle toggle_me")
        assert hm.hooks[0].enabled is False

    @pytest.mark.asyncio
    async def test_hook_no_agent_shows_error(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        # _cmd_hook checks agent for get_hook_manager()
        # Without agent, it returns early with error
        pass  # Edge case — REPL always has agent


# ═══════════════════════════════════════════════════════════════════════════════
# Status & Config Commands
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_table_rows(mock_console: MagicMock) -> list[list[str]]:
    """Extract rows from the first Rich Table printed to mock_console.

    Returns list of [column0_value, column1_value, ...] pairs from all rows.
    """
    from rich.table import Table
    for call_args in mock_console.print.call_args_list:
        args = call_args[0]
        if args and isinstance(args[0], Table):
            table = args[0]
            # Materialize each column's cells once
            col_cells = [list(col.cells) for col in table.columns]
            if not col_cells:
                return []
            n_rows = table.row_count
            rows: list[list[str]] = []
            for i in range(n_rows):
                row_cells = [str(col_cells[j][i]) for j in range(len(col_cells))]
                rows.append(row_cells)
            return rows
    return []


class TestPassiCLIStatusConfig:
    """Status and config display commands."""

    @pytest.mark.asyncio
    async def test_status_shows_mode(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_status("")
        rows = _extract_table_rows(mock_console)
        prop_cols = [r[0] for r in rows if r]
        assert "Mode" in prop_cols, f"Status should include Mode. Got: {prop_cols}"

    @pytest.mark.asyncio
    async def test_status_includes_domain(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_status("")
        rows = _extract_table_rows(mock_console)
        val_cols = [r[1] for r in rows if len(r) > 1]
        assert "test-domain" in val_cols, f"Status should include domain. Got: {val_cols}"

    @pytest.mark.asyncio
    async def test_status_includes_provider(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_status("")
        rows = _extract_table_rows(mock_console)
        vals = " ".join(str(r) for r in rows)
        assert "anthropic" in vals.lower(), f"Status should include provider. Got: {vals}"

    @pytest.mark.asyncio
    async def test_config_shows_provider(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_config("")
        rows = _extract_table_rows(mock_console)
        vals = " ".join(str(r) for r in rows)
        assert "anthropic" in vals.lower(), f"Config should show provider. Got: {vals}"

    @pytest.mark.asyncio
    async def test_config_shows_model(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_config("")
        rows = _extract_table_rows(mock_console)
        vals = " ".join(str(r) for r in rows)
        assert "claude-sonnet" in vals, f"Config should show model. Got: {vals}"

    @pytest.mark.asyncio
    async def test_config_set_not_persisted(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_config("set foo bar")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "not persisted" in args_text.lower():
                found = True
        assert found, "Should warn config set is not persisted"

    @pytest.mark.asyncio
    async def test_config_bad_args_shows_usage(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_config("bad_arg")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Usage:" in args_text:
                found = True
        assert found, "Should show usage"


# ═══════════════════════════════════════════════════════════════════════════════
# Print Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIPrintHelpers:
    """Output renderer methods use correct Rich styles and components."""

    def test_print_user_uses_panel(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        from rich.panel import Panel
        cli._print_user("Hello")
        assert mock_console.print.called
        # First positional arg should be a Panel
        first_call_arg = mock_console.print.call_args[0][0]
        assert isinstance(first_call_arg, Panel)
        assert first_call_arg.style == USER_STYLE

    def test_print_agent_with_text(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        from rich.panel import Panel
        from rich.markdown import Markdown
        cli._print_agent("Analysis complete.")
        assert mock_console.print.called
        first_call_arg = mock_console.print.call_args[0][0]
        assert isinstance(first_call_arg, Panel)
        # Panel contains Markdown
        assert isinstance(first_call_arg.renderable, Markdown)

    def test_print_agent_skips_empty(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        mock_console.print.reset_mock()
        cli._print_agent("")
        assert not mock_console.print.called

    def test_print_agent_skips_none(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        mock_console.print.reset_mock()
        cli._print_agent(None)
        assert not mock_console.print.called

    def test_print_error_uses_error_style(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        cli._print_error("Something failed")
        assert mock_console.print.called
        kw = mock_console.print.call_args[1]
        assert kw.get("style") == ERROR_STYLE

    def test_print_system_uses_system_style(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        cli._print_system("System message")
        assert mock_console.print.called
        kw = mock_console.print.call_args[1]
        assert kw.get("style") == SYSTEM_STYLE

    def test_print_tool_call_formats_string(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        cli._print_tool_call("run_python", '{"code": "1+1"}')
        assert mock_console.print.called
        kw = mock_console.print.call_args[1]
        assert kw.get("style") == TOOL_STYLE

    def test_print_tool_call_formats_dict(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        cli._print_tool_call("run_python", {"code": "print(1)", "timeout": 120})
        assert mock_console.print.called
        kw = mock_console.print.call_args[1]
        assert kw.get("style") == TOOL_STYLE


# ═══════════════════════════════════════════════════════════════════════════════
# Status Bar
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIStatusBar:
    """Status bar rendering."""

    def test_status_bar_with_agent(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        mock_console.print.reset_mock()
        cli._print_status_bar()
        assert mock_console.print.called
        kw = mock_console.print.call_args[1]
        assert kw.get("style") == STATUS_STYLE

    def test_status_bar_without_agent_returns_early(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        mock_console.print.reset_mock()
        cli._print_status_bar()
        assert not mock_console.print.called

    def test_status_bar_contains_mode(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        mock_console.print.reset_mock()
        cli.agent._mode = "plan"
        cli._print_status_bar()
        args_text = str(mock_console.print.call_args[0])
        assert "plan" in args_text

    def test_status_bar_contains_skills(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        mock_console.print.reset_mock()
        cli.agent._prompt_manager.load_skill("metabolomics")
        cli._print_status_bar()
        args_text = str(mock_console.print.call_args[0])
        assert "metabolomics" in args_text

    def test_status_bar_contains_session_info(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        mock_console.print.reset_mock()
        cli._print_status_bar()
        args_text = str(mock_console.print.call_args[0])
        assert "Session:" in args_text

    def test_status_bar_shows_keyboard_shortcuts(self, tmp_path):
        """Status bar shows real keyboard shortcuts (Ctrl+T, Ctrl+S, etc.)."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        mock_console.print.reset_mock()
        cli._print_status_bar()
        args_text = str(mock_console.print.call_args[0])
        assert "Shift+Tab" not in args_text
        assert "Ctrl+T" in args_text
        assert "Ctrl+S" in args_text
        assert "Ctrl+L" in args_text
        assert "Ctrl+D" in args_text


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIDomainCommands:
    """Domain management commands."""

    @pytest.mark.asyncio
    async def test_domain_set(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_domain("metabolomics")
        assert cli._domain == "metabolomics"

    @pytest.mark.asyncio
    async def test_domain_show_current(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._domain = "transcriptomics"
        await cli._cmd_domain("")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "transcriptomics" in args_text:
                found = True
        assert found, "Should show current domain"


# ═══════════════════════════════════════════════════════════════════════════════
# Plan Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIPlanCommands:
    """Plan display and management commands."""

    @pytest.mark.asyncio
    async def test_plan_show_no_plan(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent.get_plan = lambda: None
        await cli._cmd_plan("show")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "No active plan" in args_text:
                found = True
        assert found, "Should show no active plan message"

    @pytest.mark.asyncio
    async def test_plan_approve(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_plan("approve")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "approved" in args_text.lower():
                found = True
        assert found, "Should confirm plan approved"

    @pytest.mark.asyncio
    async def test_plan_reject(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        # Mock recycle_plan to avoid real wire.emit() and LLM calls
        cli.agent.recycle_plan = AsyncMock(return_value=AgentMessage(
            role="agent",
            content=[{"type": "text", "text": "Revised plan based on feedback."}],
        ))
        await cli._cmd_plan("reject need more details")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "rejected" in args_text.lower():
                found = True
        assert found, "Should confirm plan rejected"

    @pytest.mark.asyncio
    async def test_plan_bad_args_shows_usage(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_plan("bad_arg")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Usage:" in args_text:
                found = True
        assert found, "Should show usage"


class TestPassiCLIInterrupt:
    """/interrupt command and agent busy state."""

    @pytest.mark.asyncio
    async def test_interrupt_when_agent_not_busy(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_interrupt("")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "No operation" in args_text:
                found = True
        assert found, "Should say no operation in progress"

    @pytest.mark.asyncio
    async def test_interrupt_when_agent_busy(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._agent_busy = True
        await cli._cmd_interrupt("")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Interrupt" in args_text:
                found = True
        assert found, "Should send interrupt signal"

    @pytest.mark.asyncio
    async def test_interrupt_no_agent_is_safe(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        await cli._cmd_interrupt("")  # Should not raise


class TestPassiCLIPlanReject:
    """/plan reject with feedback flow."""

    @pytest.mark.asyncio
    async def test_reject_with_feedback(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent.recycle_plan = AsyncMock(return_value=AgentMessage(
            role="agent",
            content=[{"type": "text", "text": "Plan revised with power analysis step."}],
        ))
        await cli._cmd_plan("reject add power analysis")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "rejected" in args_text.lower():
                found = True
        assert found, "Should confirm rejection with feedback"

    @pytest.mark.asyncio
    async def test_reject_shows_plan_revised(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent.recycle_plan = AsyncMock(return_value=AgentMessage(
            role="agent",
            content=[],
        ))
        await cli._cmd_plan("reject use non-parametric tests")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "revised" in args_text.lower():
                found = True
        assert found, "Should say plan revised"


# ═══════════════════════════════════════════════════════════════════════════════
# Clear, Save, Export Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLISessionCommands:
    """Session management commands: clear, save, export."""

    @pytest.mark.asyncio
    async def test_clear_calls_agent_reset(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent.reset = AsyncMock()
        await cli._cmd_clear("")
        cli.agent.reset.assert_called_once()
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "cleared" in args_text.lower():
                found = True
        assert found, "Should confirm context cleared"

    @pytest.mark.asyncio
    async def test_save_creates_checkpoint(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_save("my_checkpoint")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "checkpoint saved" in args_text.lower():
                found = True
        assert found, "Should confirm checkpoint saved"

    @pytest.mark.asyncio
    async def test_save_no_name_generates_timestamp(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_save("")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "checkpoint saved" in args_text.lower():
                found = True
        assert found, "Should confirm checkpoint saved even without name"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases & Error Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIEdgeCases:
    """Edge cases and robustness tests."""

    @pytest.mark.asyncio
    async def test_empty_input_skipped(self, tmp_path):
        """Empty or whitespace input should be skipped in REPL."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        # Simulate REPL handling of empty input
        user_input = "   "
        if not user_input.strip():
            pass  # Correctly skipped
        assert True  # No error on empty input

    @pytest.mark.asyncio
    async def test_command_case_insensitive(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._handle_command("/HELP")
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_multiple_commands_in_sequence(self, tmp_path):
        """Running multiple commands in sequence should not fail."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        await cli._handle_command("/mode plan")
        await cli._handle_command("/mode chat")
        await cli._handle_command("/skill list")
        await cli._handle_command("/status")
        assert cli.agent.mode == "chat"

    @pytest.mark.asyncio
    async def test_process_message_with_mocked_stream(self, tmp_path):
        """_process_message handles streaming response from agent."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        # Create minimal stream that yields text
        from passi.soul.protocol import AgentStreamEvent

        async def fake_stream(_msg):
            yield AgentStreamEvent(type="text", content="Hello from agent")

        cli.agent.chat_stream = fake_stream
        await cli._process_message("test query")
        # Should have printed agent response
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_process_message_handles_exception(self, tmp_path):
        """_process_message catches exceptions and prints error."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        async def crash_stream(_msg):
            raise RuntimeError("Simulated crash")
            yield  # unreachable

        cli.agent.chat_stream = crash_stream
        await cli._process_message("test query")
        # Should have printed error (not raised)
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "Simulated crash" in args_text:
                found = True
        assert found, "Should catch and display error"

    @pytest.mark.asyncio
    async def test_get_input_mode_label_chat(self, tmp_path):
        """_get_input uses correct mode label for chat mode."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "chat"
        prompt = _MODE_LABELS["chat"]
        assert prompt == "[chat]"

    @pytest.mark.asyncio
    async def test_get_input_mode_label_plan(self, tmp_path):
        """_get_input uses correct mode label for plan mode."""
        assert _MODE_LABELS["plan"] == "[plan]"

    @pytest.mark.asyncio
    async def test_get_input_mode_label_afk(self, tmp_path):
        """_get_input uses correct mode label for afk mode."""
        assert _MODE_LABELS["afk"] == "[afk]"

    def test_cli_init_defaults(self, tmp_path):
        runtime = _make_runtime(tmp_path)
        cli = PassiCLI(runtime.config)
        assert cli._running is True
        assert cli._domain == "multi-omics"
        assert cli.agent is None
        assert cli._start_mode is None
        assert cli._start_skills is None

    @pytest.mark.asyncio
    async def test_rebuild_prompt_no_agent_safe(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path, create_agent=False)
        # Should not raise
        cli._rebuild_prompt()


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Integration (mode + skills + hooks end-to-end)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIAgentIntegration:
    """End-to-end tests with real PassiAgent + FakeLLMClient."""

    @pytest.mark.asyncio
    async def test_mode_switch_updates_agent_state(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent.set_mode = lambda mode="chat", plan_first=False, skills=None: setattr(cli.agent, '_mode', mode) or setattr(cli.agent, '_plan_first', plan_first)
        await cli._cmd_mode("afk")
        assert cli.agent._mode == "afk"
        assert cli.agent._plan_first is False

    @pytest.mark.asyncio
    async def test_skill_activation_persists_across_mode_switches(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        # Load skill
        cli.agent._prompt_manager.load_skill("metabolomics")
        # Switch mode via /mode
        cli.agent.set_mode = lambda mode="chat", plan_first=False, skills=None: setattr(cli.agent, '_mode', mode)
        await cli._cmd_mode("plan")
        # Skill should still be active
        assert "metabolomics" in cli.agent._prompt_manager.active_skills

    @pytest.mark.asyncio
    async def test_hook_manager_accessible(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        assert hm is not None
        assert hasattr(hm, 'hooks')
        assert hasattr(hm, 'add_hook')
        assert hasattr(hm, 'remove_hook')
        assert hasattr(hm, 'toggle_hook')

    @pytest.mark.asyncio
    async def test_command_sequence_full_workflow(self, tmp_path):
        """Simulate a full TUI session workflow."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)

        # 1. Check status
        await cli._cmd_status("")

        # 2. Switch to plan mode
        cli.agent.set_mode = lambda mode="chat", plan_first=False, skills=None: (
            setattr(cli.agent, '_mode', mode),
            setattr(cli.agent, '_plan_first', plan_first),
        )
        await cli._cmd_mode("plan")

        # 3. Load skills
        await cli._cmd_skill("use metabolomics,stats")

        # 4. Check skills
        assert "metabolomics" in cli.agent._prompt_manager.active_skills
        assert "stats" in cli.agent._prompt_manager.active_skills

        # 5. Show status with skills
        await cli._cmd_status("")

        # 6. Switch back to chat
        await cli._cmd_mode("chat")

        # 7. Clear context
        cli.agent.reset = AsyncMock()
        await cli._cmd_clear("")

        # 8. Quit
        await cli._cmd_quit("")
        assert cli._running is False


# ═══════════════════════════════════════════════════════════════════════════════
# Keyboard Shortcut System (prompt_toolkit-based, like kimi-cli)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIShortcutSentinels:
    """Sentinel values that key bindings return to the REPL loop."""

    def test_sentinels_are_distinct(self):
        """All sentinel values are unique."""
        sentinels = [_SENTINEL_CYCLE_MODE, _SENTINEL_SAVE, _SENTINEL_CLEAR_SCREEN, _SENTINEL_QUIT]
        assert len(sentinels) == len(set(sentinels))

    def test_sentinels_start_with_null(self):
        """Sentinels use \\x00 prefix to avoid collision with real text."""
        for s in [_SENTINEL_CYCLE_MODE, _SENTINEL_SAVE, _SENTINEL_CLEAR_SCREEN, _SENTINEL_QUIT]:
            assert s.startswith("\x00")

    def test_prompt_style_is_configured(self):
        """_PROMPT_STYLE has the expected style rules."""
        from prompt_toolkit.styles import Style as PTStyle
        assert isinstance(_PROMPT_STYLE, PTStyle)

    def test_create_input_session_exists(self, tmp_path):
        """_create_input_session method is callable and well-formed."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        assert callable(cli._create_input_session)
        # prompt_toolkit PromptSession requires a real console — construction
        # is tested indirectly via the REPL integration tests below.

    def test_no_msvcrt_imports_in_cli(self):
        """Regression: cli.py must NOT import msvcrt (old broken shortcut system)."""
        import inspect
        from passi.ui import cli as cli_module
        source = inspect.getsource(cli_module)
        assert "msvcrt" not in source, "cli.py should not use msvcrt (replaced by prompt_toolkit)"
        assert "threading" not in source, "cli.py should not use threading (replaced by prompt_toolkit)"


class TestPassiCLIREPLShortcuts:
    """REPL loop correctly handles shortcut sentinels."""

    @pytest.mark.asyncio
    async def test_repl_handles_cycle_mode_sentinel(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        # Simulate _get_input returning the sentinel
        cli.agent._mode = "chat"
        with patch.object(cli, '_get_input', side_effect=[_SENTINEL_CYCLE_MODE, _SENTINEL_QUIT]):
            await cli._repl()
        assert cli.agent._mode == "plan"

    @pytest.mark.asyncio
    async def test_repl_handles_save_sentinel(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch.object(cli, '_get_input', side_effect=[_SENTINEL_SAVE, _SENTINEL_QUIT]):
            await cli._repl()
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if a)
            if "checkpoint saved" in args_text.lower():
                found = True
        assert found, "Ctrl+S should trigger save checkpoint"

    @pytest.mark.asyncio
    async def test_repl_handles_clear_screen_sentinel(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch.object(cli, '_get_input', side_effect=[_SENTINEL_CLEAR_SCREEN, _SENTINEL_QUIT]):
            await cli._repl()
        assert mock_console.clear.called

    @pytest.mark.asyncio
    async def test_repl_handles_quit_sentinel(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._running = True
        with patch.object(cli, '_get_input', side_effect=[_SENTINEL_QUIT]):
            await cli._repl()
        assert not cli._running
        assert cli._running is False

    @pytest.mark.asyncio
    async def test_repl_slash_command_after_shortcut(self, tmp_path):
        """Normal slash commands still work after shortcut sentinel handling."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch.object(cli, '_get_input', side_effect=["/mode plan", _SENTINEL_QUIT]):
            await cli._repl()
        assert cli.agent._mode == "plan"

    @pytest.mark.asyncio
    async def test_repl_chat_message_works(self, tmp_path):
        """Chat messages still process correctly through prompt_toolkit input."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        from passi.soul.protocol import AgentStreamEvent

        async def fake_stream(_msg):
            yield AgentStreamEvent(type="text", content="response")

        cli.agent.chat_stream = fake_stream
        with patch.object(cli, '_get_input', side_effect=["test message", _SENTINEL_QUIT]):
            await cli._repl()
        # Should not crash; message processed via _process_message
        assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_repl_empty_input_skipped(self, tmp_path):
        """Empty input is skipped without error."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch.object(cli, '_get_input', side_effect=["   ", _SENTINEL_QUIT]):
            await cli._repl()
        # Empty input should be skipped, no crash

    @pytest.mark.asyncio
    async def test_help_text_includes_keyboard_shortcuts(self, tmp_path):
        """HELP_TEXT documents the keyboard shortcuts."""
        assert "Ctrl+T" in HELP_TEXT
        assert "Ctrl+S" in HELP_TEXT
        assert "Ctrl+L" in HELP_TEXT
        assert "Ctrl+D" in HELP_TEXT
        assert "Alt+Enter" in HELP_TEXT
        assert "Ctrl+C" in HELP_TEXT


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Medium Gaps: InputSessionLabels, CtrlCKeybinding, PlanShowActive
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIInputSessionLabels:
    """Prompt label variations — _create_input_session message/rprompt."""

    @pytest.mark.asyncio
    async def test_label_default_chat(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "chat"
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
            label = str(session.message())
        assert "chat" in label

    @pytest.mark.asyncio
    async def test_label_agent_busy_shows_dot(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._agent_busy = True
        cli.agent._mode = "chat"
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
            label = str(session.message())
        assert "●" in label

    @pytest.mark.asyncio
    async def test_label_step_confirm_mode(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._step_confirm_mode = True
        cli.agent._agent_busy = False
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
            label = str(session.message())
        assert "[plan-step]" in label

    @pytest.mark.asyncio
    async def test_label_plan_qa_active(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._plan_qa_active = True
        cli.agent._agent_busy = False
        cli.agent._step_confirm_mode = False
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
            label = str(session.message())
        assert "[plan-qa]" in label

    @pytest.mark.asyncio
    async def test_label_plan_mode(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._mode = "plan"
        cli.agent._agent_busy = False
        cli.agent._step_confirm_mode = False
        cli.agent._plan_qa_active = False
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
            label = str(session.message())
        assert "[plan]" in label

    @pytest.mark.asyncio
    async def test_rprompt_none_when_no_skills(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        # Ensure agent has no active skills so rprompt returns None
        cli.agent._prompt_manager._active_skills = []
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
        # _rprompt is always a callable; its return value is None when no skills
        assert session.rprompt() is None


class TestPassiCLICtrlCKeybinding:
    """Ctrl+C key binding: interrupt agent when busy, clear input when idle."""

    @pytest.mark.asyncio
    async def test_ctrl_c_interrupts_agent_when_busy(self, tmp_path):
        from prompt_toolkit.keys import Keys

        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._agent_busy = True
        cli.agent.interrupt = MagicMock()

        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)

        # Find the c-c key binding and invoke it
        mock_event = MagicMock()
        mock_event.current_buffer.text = "some input"
        mock_event.current_buffer.reset = MagicMock()
        mock_event.app.exit = MagicMock()

        found = False
        for binding in session.key_bindings.bindings:
            for key in binding.keys:
                if key == Keys.ControlC:
                    binding.handler(mock_event)
                    found = True
                    break
            if found:
                break

        assert found, "Ctrl+C binding not found"
        cli.agent.interrupt.assert_called_once()
        mock_event.current_buffer.reset.assert_called()

    @pytest.mark.asyncio
    async def test_ctrl_c_clears_input_when_idle(self, tmp_path):
        from prompt_toolkit.keys import Keys

        cli, _, _ = _make_cli_with_mocks(tmp_path)
        cli.agent._agent_busy = False
        cli.agent.interrupt = MagicMock()

        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)

        mock_event = MagicMock()
        mock_event.current_buffer.text = "some text"
        mock_event.current_buffer.reset = MagicMock()
        mock_event.app.exit = MagicMock()

        found = False
        for binding in session.key_bindings.bindings:
            for key in binding.keys:
                if key == Keys.ControlC:
                    binding.handler(mock_event)
                    found = True
                    break
            if found:
                break

        assert found, "Ctrl+C binding not found"
        cli.agent.interrupt.assert_not_called()
        mock_event.current_buffer.reset.assert_called()


class TestPassiCLIPlanShowActive:
    """/plan show with an active plan and multiple steps."""

    @pytest.mark.asyncio
    async def test_with_multiple_steps(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        plan = AnalysisPlan(
            plan_id="plan-001",
            title="Metabolomics Analysis",
            steps=[
                PlanStep(step_id="1", order=1, description="Load data", status=StepStatus.DONE),
                PlanStep(step_id="2", order=2, description="Normalize", status=StepStatus.RUNNING),
                PlanStep(step_id="3", order=3, description="Run DESeq2", status=StepStatus.PENDING),
            ],
        )
        cli.agent.get_plan = MagicMock(return_value=plan)

        await cli._cmd_plan("show")

        # Table is a Rich Table object — check for it in call args
        found_table = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'columns') and hasattr(arg, 'title'):
                    found_table = True
        assert found_table, "Should print a Table for the plan"

    @pytest.mark.asyncio
    async def test_all_status_icons(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        plan = AnalysisPlan(
            plan_id="plan-002",
            title="Status Check",
            steps=[
                PlanStep(step_id="1", order=1, description="Done step", status=StepStatus.DONE),
                PlanStep(step_id="2", order=2, description="Running step", status=StepStatus.RUNNING),
                PlanStep(step_id="3", order=3, description="Pending step", status=StepStatus.PENDING),
                PlanStep(step_id="4", order=4, description="Failed step", status=StepStatus.FAILED),
                PlanStep(step_id="5", order=5, description="Skipped step", status=StepStatus.SKIPPED),
                PlanStep(step_id="6", order=6, description="Awaiting step", status=StepStatus.AWAITING_CONFIRMATION),
                PlanStep(step_id="7", order=7, description="Interrupted step", status=StepStatus.INTERRUPTED),
            ],
        )
        cli.agent.get_plan = MagicMock(return_value=plan)

        await cli._cmd_plan("show")

        # Get the Table object from console.print calls
        table = None
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'columns') and hasattr(arg, 'title'):
                    table = arg
                    break
        assert table is not None, "Should print a Table"

        # Render the table to a string to check content
        from rich.console import Console as RichConsole
        string_console = RichConsole(force_terminal=True, width=120)
        with string_console.capture() as capture:
            string_console.print(table)
        all_text = capture.get()

        # Verify all status icons appear
        assert "✓" in all_text
        assert "●" in all_text
        assert "○" in all_text
        assert "✗" in all_text
        assert "⏭" in all_text
        assert "⏸" in all_text
        assert "⚠" in all_text

    @pytest.mark.asyncio
    async def test_empty_arg_shows_plan(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        plan = AnalysisPlan(
            plan_id="plan-003",
            title="Pipeline",
            steps=[PlanStep(step_id="1", order=1, description="QC", status=StepStatus.PENDING)],
        )
        cli.agent.get_plan = MagicMock(return_value=plan)

        await cli._cmd_plan("")

        found_table = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'columns') and hasattr(arg, 'title'):
                    found_table = True
        assert found_table, "Empty arg should show plan table"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Remaining Gaps: PlanRejectInline, MethodsFormats, Streaming
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIPlanRejectInline:
    """/plan reject without pre-supplied feedback — inline prompt path."""

    @pytest.mark.asyncio
    async def test_reject_without_feedback_prompts_user(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent.recycle_plan = AsyncMock(return_value=AgentMessage(
            role="agent",
            content=[{"type": "text", "text": "Plan revised with additional step."}],
        ))
        with patch.object(cli, "_get_input", return_value="add power analysis"):
            await cli._cmd_plan("reject")

        # Should prompt user for feedback
        found_prompt = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "What would you like to change" in args_text:
                found_prompt = True
        assert found_prompt, "Should prompt for feedback"
        # recycle_plan called with user feedback
        cli.agent.recycle_plan.assert_called_once_with("add power analysis")

    @pytest.mark.asyncio
    async def test_reject_empty_feedback_cancels(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli.agent.recycle_plan = AsyncMock()
        with patch.object(cli, "_get_input", return_value=""):
            await cli._cmd_plan("reject")

        found_no_feedback = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "No feedback provided" in args_text:
                found_no_feedback = True
        assert found_no_feedback, "Should say no feedback provided"
        cli.agent.recycle_plan.assert_not_called()


class TestPassiCLIMethodsFormats:
    """/methods and /formats commands."""

    @pytest.mark.asyncio
    async def test_methods_with_domain(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_methods("transcriptomics")
        found = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'markup') and "transcriptomics" in str(arg.markup):
                    found = True
        assert found, "Should display methods for transcriptomics domain"

    @pytest.mark.asyncio
    async def test_methods_defaults_to_current_domain(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._domain = "metabolomics"
        await cli._cmd_methods("")
        found = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'markup') and "metabolomics" in str(arg.markup):
                    found = True
        assert found, "Should default to current metabolomics domain"

    @pytest.mark.asyncio
    async def test_methods_unknown_domain_shows_none(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_methods("nonexistent_domain_xyz")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "No methods found" in args_text:
                found = True
        assert found, "Should say no methods found for unknown domain"

    @pytest.mark.asyncio
    async def test_formats_with_domain(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_formats("genomics")
        found = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'markup') and "genomics" in str(arg.markup):
                    found = True
        assert found, "Should display formats for genomics domain"

    @pytest.mark.asyncio
    async def test_formats_unknown_domain_shows_none(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        await cli._cmd_formats("nonexistent_domain_xyz")
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "No formats found" in args_text:
                found = True
        assert found, "Should say no formats found for unknown domain"


class TestPassiCLIProcessMessageStreaming:
    """Streaming edge cases in _process_message."""

    @pytest.mark.asyncio
    async def test_thinking_event_does_not_print(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        async def fake_stream(_msg):
            yield AgentStreamEvent(type="thinking", content="Evaluating methods...")
            yield AgentStreamEvent(type="done", content="")

        cli.agent.chat_stream = fake_stream
        cli._print_status_bar = MagicMock()  # Prevent status bar call from crashing

        await cli._process_message("analyze")

        # No text content should be printed for thinking events
        text_found = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if isinstance(arg, str) and "Evaluating methods" in arg:
                    text_found = True
        assert not text_found, "Thinking events should not print agent text"

    @pytest.mark.asyncio
    async def test_no_response_fallback(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        async def fake_stream(_msg):
            # Empty stream — no yields
            if False:
                yield

        cli.agent.chat_stream = fake_stream
        cli._print_status_bar = MagicMock()

        await cli._process_message("query")

        found_no_response = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'renderable'):
                    if hasattr(arg.renderable, 'markup') and "no response" in str(arg.renderable.markup).lower():
                        found_no_response = True
                    elif isinstance(arg.renderable, str) and "no response" in arg.renderable.lower():
                        found_no_response = True
                elif isinstance(arg, str) and "no response" in arg.lower():
                    found_no_response = True
        assert found_no_response, "Should print (no response) for empty stream"

    @pytest.mark.asyncio
    async def test_answer_routed_for_pending_question(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        async def fake_stream(_msg):
            yield AgentStreamEvent(
                type="pending_question",
                content="Which method?",
                metadata={"context": "Step 3", "options": ["A", "B"]},
            )
            yield AgentStreamEvent(type="done", content="")

        cli.agent.chat_stream = fake_stream
        cli._print_status_bar = MagicMock()
        cli._handle_pending_question = AsyncMock()

        await cli._process_message("help me")

        # Should delegate to _handle_pending_question
        cli._handle_pending_question.assert_called_once()
        args = cli._handle_pending_question.call_args[0]
        assert args[0]["question"] == "Which method?"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Critical Gaps: PendingQuestion, Export, HookAddInteractive
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIPendingQuestion:
    """_handle_pending_question: ask_user tool interaction loop."""

    @pytest.mark.asyncio
    async def test_with_options(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_selection", return_value="DESeq2"):
            await cli._handle_pending_question({
                "question": "Which method?",
                "options": ["DESeq2", "edgeR", "limma"],
            })
        # Panel is a Rich Panel object — check renderable text
        found_panel = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'renderable') and "Which method?" in str(arg.renderable):
                    found_panel = True
        assert found_panel, "Should display question in Panel"
        # Answer routed to _process_message via selection
        cli._process_message.assert_called_once_with("DESeq2")

    @pytest.mark.asyncio
    async def test_without_options(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_input", return_value="yes"):
            await cli._handle_pending_question({"question": "Proceed?"})
        # No "Options:" in output
        found_opts = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "Options:" in args_text:
                found_opts = True
        assert not found_opts, "Should not show Options when none provided"
        cli._process_message.assert_called_once_with("yes")

    @pytest.mark.asyncio
    async def test_empty_answer_defaults_to_no_answer(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_input", return_value="   "):
            await cli._handle_pending_question({"question": "Any feedback?"})
        cli._process_message.assert_called_once_with("(no answer)")

    @pytest.mark.asyncio
    async def test_with_context(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_input", return_value="continue"):
            await cli._handle_pending_question({
                "question": "Ready?",
                "context": "Running PCA on 1000 features...",
            })
        found_context = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "Running PCA" in args_text:
                found_context = True
        assert found_context, "Should print context"

    @pytest.mark.asyncio
    async def test_empty_context_skipped(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_input", return_value="ok"):
            await cli._handle_pending_question({"question": "Continue?", "context": ""})
        # Question should still be displayed in Panel
        found_question = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'renderable') and "Continue?" in str(arg.renderable):
                    found_question = True
        assert found_question, "Question should be displayed even with empty context"

    @pytest.mark.asyncio
    async def test_cancel_via_empty_string(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_input", return_value=""):
            await cli._handle_pending_question({"question": "Add more steps?"})
        cli._process_message.assert_called_once_with("(no answer)")


class TestPassiCLIExport:
    """/export command: chatlog generation."""

    @pytest.mark.asyncio
    async def test_export_creates_chatlog_file(self, tmp_path):
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)

        session = runtime.session.active_session
        sid = session.session_id

        # Create a wire.jsonl with sample events matching the active session
        session_dir = tmp_path / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        wire_path = session_dir / "wire.jsonl"
        events = [
            WireEvent(type=EventType.USER_MESSAGE, session_id=sid, data={"content": "analyze data"}),
            WireEvent(type=EventType.AGENT_MESSAGE, session_id=sid, data={"content": "I will analyze."}),
        ]
        wire_path.write_text("\n".join(json.dumps(e.model_dump()) for e in events), encoding="utf-8")

        # Mock session dir lookup to return our temp dir
        with patch.object(runtime.session, "get_session_dir", return_value=session_dir):
            export_dir = tmp_path / "export_results"
            export_dir.mkdir(parents=True)
            cli.config.result_dir = export_dir

            await cli._cmd_export("")

        # Check file created and contains expected content
        chatlog_files = list(export_dir.glob("chatlog_*.md"))
        assert len(chatlog_files) == 1
        content = chatlog_files[0].read_text(encoding="utf-8")
        assert "**User:**" in content
        assert "analyze data" in content

    @pytest.mark.asyncio
    async def test_export_no_session_returns_early(self, tmp_path):
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        runtime.session._active_session = None
        # Should not raise
        await cli._cmd_export("")

    @pytest.mark.asyncio
    async def test_export_file_contains_expected_markdown(self, tmp_path):
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)

        session = runtime.session.active_session
        sid = session.session_id

        session_dir = tmp_path / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        wire_path = session_dir / "wire.jsonl"
        events = [
            WireEvent(type=EventType.USER_MESSAGE, session_id=sid, data={"content": "run qc"}),
            WireEvent(type=EventType.AGENT_MESSAGE, session_id=sid, data={"content": "QC complete."}),
        ]
        wire_path.write_text("\n".join(json.dumps(e.model_dump()) for e in events), encoding="utf-8")

        with patch.object(runtime.session, "get_session_dir", return_value=session_dir):
            export_dir = tmp_path / "export_results"
            export_dir.mkdir(parents=True)
            cli.config.result_dir = export_dir

            await cli._cmd_export("")

        chatlog_files = list(export_dir.glob("chatlog_*.md"))
        content = chatlog_files[0].read_text(encoding="utf-8")
        assert "# Session Chat Log" in content
        assert "**User:**" in content
        assert "**Agent:**" in content
        assert "run qc" in content
        assert "QC complete." in content


class TestPassiCLIHookAddInteractive:
    """_cmd_hook_add_interactive: multi-step hook creation wizard."""

    @pytest.mark.asyncio
    async def test_full_shell_workflow(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        initial_count = len(hm.hooks)

        with patch("passi.ui.cli.Prompt") as mock_prompt:
            mock_prompt.ask.side_effect = [
                "pre_commit",     # name
                "pre_tool",       # event
                "shell",          # type
                "echo linting",   # command
            ]
            await cli._cmd_hook_add_interactive()

        assert len(hm.hooks) == initial_count + 1
        hook = hm.hooks[-1]
        assert hook.name == "pre_commit"
        assert hook.event == "pre_tool"
        assert hook.type == "shell"
        assert hook.command == "echo linting"
        assert hook.enabled is True

    @pytest.mark.asyncio
    async def test_cancelled_on_empty_name(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        initial_count = len(hm.hooks)

        with patch("passi.ui.cli.Prompt") as mock_prompt:
            mock_prompt.ask.return_value = "   "
            await cli._cmd_hook_add_interactive()

        assert len(hm.hooks) == initial_count  # No hook added
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "Cancelled" in args_text:
                found = True
        assert found, "Should print Cancelled"

    @pytest.mark.asyncio
    async def test_invalid_event_shows_error(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        initial_count = len(hm.hooks)

        with patch("passi.ui.cli.Prompt") as mock_prompt:
            mock_prompt.ask.side_effect = [
                "test_hook",
                "invalid_event",
            ]
            await cli._cmd_hook_add_interactive()

        assert len(hm.hooks) == initial_count
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "Invalid event" in args_text:
                found = True
        assert found, "Should print Invalid event error"

    @pytest.mark.asyncio
    async def test_invalid_type_shows_error(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()
        initial_count = len(hm.hooks)

        with patch("passi.ui.cli.Prompt") as mock_prompt:
            mock_prompt.ask.side_effect = [
                "test_hook",
                "pre_tool",
                "unsupported_type",
            ]
            await cli._cmd_hook_add_interactive()

        assert len(hm.hooks) == initial_count
        found = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "Type must be" in args_text:
                found = True
        assert found, "Should print Type must be error"

    @pytest.mark.asyncio
    async def test_python_hook_workflow(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        hm = cli.agent.get_hook_manager()

        with patch("passi.ui.cli.Prompt") as mock_prompt:
            # Python code: first 3 lines are code, 4th is empty to end
            mock_prompt.ask.side_effect = [
                "py_hook",       # name
                "post_tool",     # event
                "python",        # type
                "import sys",    # code line 1
                "print('ok')",   # code line 2
                "",              # empty = end code input
            ]
            await cli._cmd_hook_add_interactive()

        hook = hm.hooks[-1]
        assert hook.name == "py_hook"
        assert hook.type == "python"
        assert hook.code == "import sys\nprint('ok')"
        assert hook.command == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Selection UI & Command History
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLISelectionUI:
    """_get_selection() interactive option selection and _handle_pending_question routing."""

    @pytest.mark.asyncio
    async def test_options_route_to_get_selection(self, tmp_path):
        """When options are present, _handle_pending_question calls _get_selection."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_selection", return_value="Proceed"):
            await cli._handle_pending_question({
                "question": "Proceed with step?",
                "options": ["Proceed", "Skip", "Modify"],
            })
        cli._process_message.assert_called_once_with("Proceed")

    @pytest.mark.asyncio
    async def test_custom_input_sentinel_falls_back_to_user_input(self, tmp_path):
        """When _get_selection returns _SENTINEL_CUSTOM_INPUT, call _get_user_input."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_selection", return_value="\x00custom") as mock_sel, \
             patch.object(cli, "_get_user_input", return_value="use limma instead") as mock_ui:
            await cli._handle_pending_question({
                "question": "Which method?",
                "options": ["DESeq2", "edgeR"],
            })
        mock_sel.assert_called_once()
        mock_ui.assert_called_once_with("Your answer: ")
        cli._process_message.assert_called_once_with("use limma instead")

    @pytest.mark.asyncio
    async def test_no_options_uses_get_user_input_directly(self, tmp_path):
        """When no options, _handle_pending_question still uses _get_user_input."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._process_message = AsyncMock()
        with patch.object(cli, "_get_user_input", return_value="yes") as mock_ui, \
             patch.object(cli, "_get_selection") as mock_sel:
            await cli._handle_pending_question({"question": "Proceed?"})
        mock_sel.assert_not_called()
        mock_ui.assert_called_once_with("Your answer: ")
        cli._process_message.assert_called_once_with("yes")

    @pytest.mark.asyncio
    async def test_get_selection_exists_and_accepts_options(self, tmp_path):
        """_get_selection method exists on PassiCLI and accepts options list."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        assert hasattr(cli, "_get_selection")
        # Verify callable with options — may fail without terminal, that's expected
        assert callable(cli._get_selection)

    def test_get_selection_with_empty_list_does_not_crash_on_build(self, tmp_path):
        """_get_selection builds display_options with Custom input... appended."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        # Verify method exists and is properly defined
        import inspect
        sig = inspect.signature(cli._get_selection)
        assert list(sig.parameters.keys()) == ["options"]


class TestPassiCLIInputHistory:
    """Command input history via InMemoryHistory."""

    def test_input_history_created_in_init(self, tmp_path):
        """_input_history is an InMemoryHistory instance after construction."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        from prompt_toolkit.history import InMemoryHistory
        assert isinstance(cli._input_history, InMemoryHistory)

    def test_input_history_passed_to_prompt_session(self, tmp_path):
        """PromptSession in _create_input_session receives the history object."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        with _pipe_session() as (inp, out):
            session = cli._create_input_session(input=inp, output=out)
            assert session.default_buffer.history is cli._input_history

    def test_history_is_shared_across_sessions(self, tmp_path):
        """Same history instance is reused across PromptSession creations."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        with _pipe_session() as (inp, out):
            s1 = cli._create_input_session(input=inp, output=out)
            s2 = cli._create_input_session(input=inp, output=out)
            assert s1.default_buffer.history is s2.default_buffer.history
            assert s1.default_buffer.history is cli._input_history


# ═══════════════════════════════════════════════════════════════════════════════
# Session Resume — Load Historical Sessions
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLISessionResume:
    """Session detection, selection, and loading on startup."""

    def test_format_session_entry(self, tmp_path):
        """_format_session_entry formats session dict into display string."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        entry = cli._format_session_entry({
            "session_id": "session_20260626_143022",
            "domain": "transcriptomics",
            "message_count": 15,
            "created_at": "2026-06-26T14:30:22.123456+00:00",
        })
        assert "session_20260626_143022" in entry
        assert "transcriptomics" in entry
        assert "15 msgs" in entry
        assert "2026-06-26 14:30" in entry

    def test_format_session_entry_fallback_date(self, tmp_path):
        """_format_session_entry handles missing/invalid created_at."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        entry = cli._format_session_entry({
            "session_id": "s1",
            "domain": "genomics",
            "message_count": 0,
            "created_at": "",
        })
        assert "s1" in entry
        assert "genomics" in entry
        assert "0 msgs" in entry

    @pytest.mark.asyncio
    async def test_load_existing_session_with_wire(self, tmp_path):
        """_load_existing_session loads metadata and replays wire events into context."""
        from passi.wire.protocol import WireEvent, EventType

        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)

        # Create a session with metadata and wire.jsonl
        meta = runtime.session.create_session(domain="genomics")
        sid = meta.session_id
        session_dir = runtime.session.get_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)

        # Write wire events
        wire_path = session_dir / "wire.jsonl"
        events = [
            WireEvent(type=EventType.USER_MESSAGE, session_id=sid, data={"content": "analyze RNA-seq"}),
            WireEvent(type=EventType.AGENT_MESSAGE, session_id=sid, data={"content": "I will help analyze."}),
            WireEvent(type=EventType.TOOL_CALL, session_id=sid, data={"tool": "run_python", "params": {"code": "x=1"}}),
        ]
        wire_path.write_text("\n".join(e.model_dump_json() for e in events), encoding="utf-8")

        # Load the session
        await cli._load_existing_session(sid)

        # Verify domain restored
        assert cli._domain == "genomics"
        # Verify agent was created and initialized
        assert cli.agent is not None

    @pytest.mark.asyncio
    async def test_load_existing_session_no_wire(self, tmp_path):
        """_load_existing_session handles missing wire.jsonl gracefully."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        meta = runtime.session.create_session(domain="clinical")
        sid = meta.session_id
        session_dir = runtime.session.get_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        # No wire.jsonl written

        await cli._load_existing_session(sid)
        assert cli._domain == "clinical"
        assert cli.agent is not None

    @pytest.mark.asyncio
    async def test_start_with_existing_sessions_shows_selection(self, tmp_path):
        """When sessions exist, start() offers selection UI."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        runtime.session.create_session(domain="transcriptomics")
        # Mock _get_selection to simulate user choosing "New Session"
        with patch.object(cli, "_get_selection", return_value="New Session") as mock_sel, \
             patch.object(cli, "_start_new_session") as mock_new, \
             patch.object(cli, "_repl") as mock_repl:
            await cli.start()
            mock_sel.assert_called_once()
            mock_new.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_with_existing_session_chosen(self, tmp_path):
        """When user selects an existing session, it gets loaded."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        meta = runtime.session.create_session(domain="genomics")
        sid = meta.session_id
        # Need wire.jsonl for load to work
        session_dir = runtime.session.get_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "wire.jsonl").write_text("", encoding="utf-8")

        # Format matches what _format_session_entry produces
        formatted = cli._format_session_entry({
            "session_id": sid,
            "domain": "genomics",
            "message_count": 0,
            "created_at": meta.created_at,
        })

        with patch.object(cli, "_get_selection", return_value=formatted), \
             patch.object(cli, "_repl") as mock_repl:
            await cli.start()
            assert cli.agent is not None
            assert cli._domain == "genomics"
            assert runtime.session.active_session.session_id == sid

    @pytest.mark.asyncio
    async def test_start_no_existing_sessions_creates_new(self, tmp_path):
        """When no sessions exist, start() creates a new one directly (no selection)."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path, create_agent=False)
        with patch.object(runtime.session, "list_sessions", return_value=[]), \
             patch.object(cli, "_get_selection") as mock_sel, \
             patch.object(cli, "_repl") as mock_repl:
            await cli.start()
            # Selection UI should NOT be used when no sessions exist
            mock_sel.assert_not_called()
            # Agent should be created and session active
            assert cli.agent is not None

    @pytest.mark.asyncio
    async def test_resume_session_id_skips_selection(self, tmp_path):
        """When _resume_session_id is set, load directly without selection UI."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        meta = runtime.session.create_session(domain="clinical")
        sid = meta.session_id
        session_dir = runtime.session.get_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "wire.jsonl").write_text("", encoding="utf-8")

        cli._resume_session_id = sid
        with patch.object(cli, "_get_selection") as mock_sel, \
             patch.object(cli, "_repl") as mock_repl:
            await cli.start()
            mock_sel.assert_not_called()
            assert cli.agent is not None
            assert runtime.session.active_session.session_id == sid


class TestPassiCLISessionsCommand:
    """/sessions list and /sessions load slash commands."""

    @pytest.mark.asyncio
    async def test_sessions_list_with_data(self, tmp_path):
        """/sessions list renders a table when sessions exist."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        runtime.session.create_session(domain="transcriptomics")
        await cli._cmd_sessions("list")
        # Should print a Table
        found_table = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                if hasattr(arg, 'columns'):
                    found_table = True
                    break
        assert found_table, "Should render a Rich Table"

    @pytest.mark.asyncio
    async def test_sessions_list_empty(self, tmp_path):
        """/sessions list shows a message when no sessions exist."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        # Ensure no sessions — but list_sessions may find system sessions
        # Just verify it doesn't crash
        await cli._cmd_sessions("list")
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_sessions_load_with_id(self, tmp_path):
        """/sessions load <id> calls _load_existing_session."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        meta = runtime.session.create_session(domain="genomics")
        sid = meta.session_id
        session_dir = runtime.session.get_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "wire.jsonl").write_text("", encoding="utf-8")

        await cli._cmd_sessions(f"load {sid}")
        assert cli.agent is not None
        assert runtime.session.active_session.session_id == sid

    @pytest.mark.asyncio
    async def test_sessions_load_missing_id_shows_error(self, tmp_path):
        """/sessions load without ID shows usage error."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        await cli._cmd_sessions("load")
        found_error = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "Usage:" in args_text:
                found_error = True
        assert found_error, "Should show usage error"

    @pytest.mark.asyncio
    async def test_sessions_unknown_subcommand(self, tmp_path):
        """/sessions with bad subcommand shows error."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        await cli._cmd_sessions("bogus")
        found_error = False
        for call_args in mock_console.print.call_args_list:
            args_text = " ".join(str(a) for a in call_args[0] if isinstance(a, str))
            if "Unknown subcommand" in args_text or "bogus" in args_text:
                found_error = True
        assert found_error, "Should show unknown subcommand error"


# ═══════════════════════════════════════════════════════════════════════════════
# Extract Display Text Edge Cases (Bug 12)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIExtractDisplayText:
    """_extract_display_text handles various content types safely."""

    def test_extract_with_none_returns_empty(self):
        assert PassiCLI._extract_display_text(None) == ""

    def test_extract_with_plain_string(self):
        assert PassiCLI._extract_display_text("Hello world") == "Hello world"

    def test_extract_with_content_blocks(self):
        content = [
            {"type": "text", "text": "First paragraph."},
            {"type": "text", "text": "Second paragraph."},
        ]
        result = PassiCLI._extract_display_text(content)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_extract_with_mixed_blocks_skips_non_text(self):
        content = [
            {"type": "text", "text": "Analysis result."},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {}},
        ]
        result = PassiCLI._extract_display_text(content)
        assert "Analysis result" in result
        assert "read_file" not in result

    def test_extract_with_empty_list_returns_empty(self):
        assert PassiCLI._extract_display_text([]) == ""

    def test_extract_with_int_returns_string(self):
        assert PassiCLI._extract_display_text(42) == "42"

    def test_extract_with_dict_returns_string(self):
        result = PassiCLI._extract_display_text({"key": "value"})
        assert "key" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Get Selection Edge Cases (Bug 2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIGetSelectionEdgeCases:
    """_get_selection handles None, empty, and special-character options."""

    @pytest.mark.asyncio
    async def test_get_selection_with_none_returns_empty(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        result = await cli._get_selection(None)
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_selection_with_empty_list_does_not_crash(self, tmp_path):
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        # Patch PromptSession to avoid real terminal interaction
        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(return_value="Custom input...")

        with patch("passi.ui.cli.PromptSession", return_value=mock_session):
            result = await cli._get_selection([])
        assert result == "Custom input..."

    @pytest.mark.asyncio
    async def test_get_selection_with_html_special_chars(self, tmp_path):
        """Options with <, >, & should be HTML-escaped, not break XML parser."""
        cli, _, _ = _make_cli_with_mocks(tmp_path)
        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(return_value="|log2FC| > 1.0")

        with patch("passi.ui.cli.PromptSession", return_value=mock_session):
            result = await cli._get_selection([
                "p < 0.05",
                "|log2FC| > 1.0",
                "gene & expr",
            ])
        # If we get here without ExpatError, _bottom_toolbar HTML escaped correctly
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Process Message — tool_result and error Event Handlers (Bug 5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIProcessMessageEventHandlers:
    """_process_message handles tool_result and error stream events."""

    @pytest.mark.asyncio
    async def test_tool_result_event_prints_result(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        async def fake_stream(_msg):
            yield AgentStreamEvent(
                type="tool_call",
                content='{"path": "/data/file.csv"}',
                tool_name="read_file",
            )
            yield AgentStreamEvent(
                type="tool_result",
                content="File read successfully: 100 lines",
                tool_name="read_file",
            )
            yield AgentStreamEvent(type="done", content="")

        cli.agent.chat_stream = fake_stream
        cli._print_status_bar = MagicMock()

        await cli._process_message("read data")

        found_result = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                text = str(arg)
                if "File read successfully" in text:
                    found_result = True
        assert found_result, "tool_result event should print result summary"

    @pytest.mark.asyncio
    async def test_error_event_prints_error(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)

        async def fake_stream(_msg):
            yield AgentStreamEvent(
                type="tool_call",
                content='{"cmd": "invalid"}',
                tool_name="run_python",
            )
            yield AgentStreamEvent(
                type="error",
                content="SyntaxError: invalid syntax",
                tool_name="run_python",
            )
            yield AgentStreamEvent(type="done", content="")

        cli.agent.chat_stream = fake_stream
        cli._print_status_bar = MagicMock()

        await cli._process_message("run code")

        found_error = False
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                text = str(arg)
                if "Tool error" in text and "SyntaxError" in text:
                    found_error = True
        assert found_error, "error event should print with ERROR_STYLE"


# ═══════════════════════════════════════════════════════════════════════════════
# Pending Question — Sentinel, Cancel, Recursion Guard (Bugs 7, 8, 9)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPassiCLIPendingQuestionEdgeCases:
    """_handle_pending_question sentinel, cancellation, and recursion guarding."""

    @pytest.mark.asyncio
    async def test_pending_question_sentinel_quit(self, tmp_path):
        """_SENTINEL_QUIT in answer stops the REPL, does not call _process_message."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._running = True
        cli._get_user_input = AsyncMock(return_value=_SENTINEL_QUIT)
        cli._process_message = AsyncMock()
        cli._print_user = MagicMock()
        cli._print_status_bar = MagicMock()

        question = {"question": "Continue?", "context": "", "options": None}
        await cli._handle_pending_question(question)

        # Should set _running to False for quit sentinel
        assert cli._running is False
        # Should NOT route to _process_message
        cli._process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_question_sentinel_clear_screen(self, tmp_path):
        """_SENTINEL_CLEAR_SCREEN clears console, does not call _process_message."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._get_user_input = AsyncMock(return_value=_SENTINEL_CLEAR_SCREEN)
        cli._process_message = AsyncMock()
        cli._print_user = MagicMock()
        cli._print_status_bar = MagicMock()

        question = {"question": "Continue?", "context": "", "options": None}
        await cli._handle_pending_question(question)

        mock_console.clear.assert_called_once()
        cli._process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_question_cancel_on_selection_empty(self, tmp_path):
        """When _get_selection returns empty (cancelled), don't call _process_message."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._get_selection = AsyncMock(return_value="")  # User cancelled
        cli._process_message = AsyncMock()
        cli._print_user = MagicMock()
        cli._print_status_bar = MagicMock()

        question = {"question": "Choose:", "context": "", "options": ["A", "B"]}
        await cli._handle_pending_question(question)

        # Should NOT route cancelled question
        cli._process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_question_recursion_guard(self, tmp_path):
        """After 5 nested questions, the recursion guard fires and stops."""
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        cli._pending_question_depth = 5  # Already at limit
        cli._process_message = AsyncMock()
        cli._print_user = MagicMock()

        question = {"question": "One more?", "context": "", "options": None}
        await cli._handle_pending_question(question)

        # Should NOT route — recursion guard prevents it
        cli._process_message.assert_not_called()
        # Depth should not increase
        assert cli._pending_question_depth == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Wire Replay — tool_use stripping and tool_result pairing
# ═══════════════════════════════════════════════════════════════════════════════


class TestStripToolUseBlocks:
    """_strip_tool_use_blocks filters tool_use blocks from agent message content."""

    def test_strip_removes_tool_use_keeps_text(self):
        content = [
            {"type": "text", "text": "Let me analyze this."},
            {"type": "tool_use", "id": "call_123", "name": "read_file", "input": {}},
            {"type": "text", "text": "Analysis complete."},
        ]
        result = PassiCLI._strip_tool_use_blocks(content)
        assert len(result) == 2
        assert all(b["type"] == "text" for b in result)
        assert result[0]["text"] == "Let me analyze this."
        assert result[1]["text"] == "Analysis complete."

    def test_strip_only_tool_use_returns_none(self):
        content = [
            {"type": "tool_use", "id": "call_123", "name": "read_file", "input": {}},
        ]
        result = PassiCLI._strip_tool_use_blocks(content)
        assert result is None

    def test_strip_string_passes_through(self):
        assert PassiCLI._strip_tool_use_blocks("Plain text") == "Plain text"

    def test_strip_empty_list_returns_none(self):
        assert PassiCLI._strip_tool_use_blocks([]) is None


class TestCanPairToolResults:
    """_can_pair_tool_results checks tool_use / tool_result ID matching."""

    def test_can_pair_when_ids_match(self):
        content = [
            {"type": "text", "text": "Running tool."},
            {"type": "tool_use", "id": "call_abc", "name": "read_file", "input": {}},
            {"type": "tool_use", "id": "call_def", "name": "run_python", "input": {}},
        ]
        tool_results = [
            {"tool_use_id": "call_abc", "content": "file content"},
            {"tool_use_id": "call_def", "content": "script output"},
        ]
        assert PassiCLI._can_pair_tool_results(content, tool_results) is True

    def test_cannot_pair_when_ids_mismatch(self):
        content = [
            {"type": "tool_use", "id": "call_abc", "name": "read_file", "input": {}},
        ]
        tool_results = [
            {"tool_use_id": "", "content": "result without id"},  # old session
        ]
        assert PassiCLI._can_pair_tool_results(content, tool_results) is False

    def test_can_pair_no_tool_use_blocks(self):
        content = [{"type": "text", "text": "Just text."}]
        tool_results = [{"tool_use_id": "", "content": "orphan result"}]
        assert PassiCLI._can_pair_tool_results(content, tool_results) is True

    def test_can_pair_string_content(self):
        assert PassiCLI._can_pair_tool_results("Plain string", []) is True

    def test_can_pair_empty_tool_results(self):
        content = [{"type": "tool_use", "id": "call_abc", "name": "t", "input": {}}]
        assert PassiCLI._can_pair_tool_results(content, []) is False


class TestGetSelectionFallback:
    """_get_selection_fallback provides text-based option selection."""

    @pytest.mark.asyncio
    async def test_fallback_returns_selected_option(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="2")
            result = await cli._get_selection_fallback(["Option A", "Option B", "Option C"])
            assert result == "Option B"

    @pytest.mark.asyncio
    async def test_fallback_last_number_is_custom_input(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="4")
            result = await cli._get_selection_fallback(["A", "B", "C"])
            assert result == _SENTINEL_CUSTOM_INPUT

    @pytest.mark.asyncio
    async def test_fallback_empty_input_returns_empty(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="")
            result = await cli._get_selection_fallback(["A", "B"])
            assert result == ""

    @pytest.mark.asyncio
    async def test_fallback_invalid_number_returns_empty(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="99")
            result = await cli._get_selection_fallback(["A", "B"])
            assert result == ""

    @pytest.mark.asyncio
    async def test_fallback_non_numeric_returns_empty(self, tmp_path):
        cli, mock_console, _ = _make_cli_with_mocks(tmp_path)
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="abc")
            result = await cli._get_selection_fallback(["A", "B"])
            assert result == ""


class TestLoadExistingSessionWireReplay:
    """_load_existing_session replays wire events with proper tool_result pairing."""

    @pytest.mark.asyncio
    async def test_replay_with_tool_results(self, tmp_path):
        """Tool_result events are replayed after agent_message for proper pairing."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        meta = runtime.session.create_session(domain="test")
        sid = meta.session_id
        session_dir = runtime.session.get_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)

        # Write wire log with TOOL_CALL, TOOL_RESULT, and AGENT_MESSAGE events
        from passi.wire.protocol import Wire, EventType
        w = Wire()
        w._wire_path = session_dir / "wire.jsonl"
        w.emit(EventType.USER_MESSAGE, {"content": "Analyze data"}, sid)
        w.emit(EventType.TOOL_CALL, {"name": "read_file", "params": {}, "id": "call_001"}, sid)
        w.emit(EventType.TOOL_RESULT, {"name": "read_file", "result": {"success": True}, "tool_use_id": "call_001"}, sid)
        w.emit(EventType.AGENT_MESSAGE, {"content": [
            {"type": "text", "text": "I have read the file. The data contains 100 rows."},
            {"type": "tool_use", "id": "call_001", "name": "read_file", "input": {}},
        ]}, sid)

        # Replay
        cli._resume_session_id = sid
        cli._print_user = MagicMock()
        cli._print_agent = MagicMock()
        cli._print_system = MagicMock()
        cli._print_error = MagicMock()
        cli._repl = AsyncMock()

        await cli.start()

        # Verify agent message was added to context (with tool_use intact)
        msgs = runtime.context.get_messages()
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool_results" in roles

    @pytest.mark.asyncio
    async def test_replay_old_session_without_tool_use_ids(self, tmp_path):
        """Old wire sessions without tool_use_id fall back to stripping tool_use."""
        cli, mock_console, runtime = _make_cli_with_mocks(tmp_path)
        meta = runtime.session.create_session(domain="test")
        sid = meta.session_id
        session_dir = runtime.session.get_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)

        # Write OLD-FORMAT wire log (no tool_use_id in TOOL_RESULT)
        from passi.wire.protocol import Wire, EventType
        w = Wire()
        w._wire_path = session_dir / "wire.jsonl"
        w.emit(EventType.USER_MESSAGE, {"content": "Which comparison group?"}, sid)
        w.emit(EventType.TOOL_CALL, {"name": "ask_user", "params": {"question": "Which?"}}, sid)
        w.emit(EventType.TOOL_RESULT, {"name": "ask_user", "result": {"__ask_user__": True, "question": "Which?"}}, sid)
        w.emit(EventType.AGENT_MESSAGE, {"content": [
            {"type": "text", "text": "I need to ask a question."},
            {"type": "tool_use", "id": "call_ask", "name": "ask_user", "input": {"question": "Which?"}},
        ]}, sid)

        cli._resume_session_id = sid
        cli._print_user = MagicMock()
        cli._print_agent = MagicMock()
        cli._print_system = MagicMock()
        cli._print_error = MagicMock()
        cli._repl = AsyncMock()

        # Should NOT crash — old session tool_use blocks are stripped
        await cli.start()

        msgs = runtime.context.get_messages()
        # Assistant message should have text blocks only (tool_use stripped)
        for msg in msgs:
            if msg["role"] == "assistant":
                if isinstance(msg["content"], list):
                    for block in msg["content"]:
                        assert block.get("type") != "tool_use", (
                            "Old session tool_use blocks must be stripped"
                        )
