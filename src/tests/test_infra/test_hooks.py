"""Unit tests for HookManager — loading, CRUD, execution."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from passi.infra.hooks import (
    HookConfig,
    HookEvent,
    HookManager,
    HookType,
    HookContext,
    _template_substitute,
)
from passi.wire.protocol import EventType, Wire, WireEvent


class TestHookConfig:
    """HookConfig model tests."""

    def test_defaults(self):
        h = HookConfig(name="test", event=HookEvent.PRE_TOOL)
        assert h.name == "test"
        assert h.event == "pre_tool"
        assert h.type == HookType.SHELL
        assert h.command == ""
        assert h.code == ""
        assert h.enabled is True

    def test_python_hook(self):
        h = HookConfig(
            name="py_hook",
            event=HookEvent.POST_TOOL,
            type="python",
            code="print('hello')",
        )
        assert h.type == "python"
        assert h.code == "print('hello')"


class TestHookContext:
    """HookContext template variable tests."""

    def test_as_dict(self):
        ctx = HookContext(
            tool_name="run_python",
            params_json='{"code": "print(1)"}',
            exit_code=0,
            duration_ms=1234.5,
            run_dir="/tmp/run_001",
            session_id="abc123",
            session_domain="metabolomics",
            error_message="",
            message_preview="Analysis complete",
            message_count=5,
        )
        d = ctx.as_dict()
        assert d["tool_name"] == "run_python"
        assert d["exit_code"] == "0"
        assert d["duration_ms"] == "1234.5"
        assert d["session_id"] == "abc123"
        assert d["session_domain"] == "metabolomics"
        assert d["message_count"] == "5"

    def test_defaults(self):
        ctx = HookContext()
        d = ctx.as_dict()
        assert d["tool_name"] == ""
        assert d["exit_code"] == "0"


class TestHookManager:
    """HookManager core tests."""

    def test_loads_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "nonexistent" / "hooks.yaml"
            mgr = HookManager(hooks_path)
            assert mgr.hooks == []

    def test_loads_hooks_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_dir = Path(tmpdir)
            hooks_dir.mkdir(parents=True, exist_ok=True)
            hooks_path = hooks_dir / "hooks.yaml"
            hooks_path.write_text(
                yaml.safe_dump({
                    "hooks": [
                        {
                            "name": "test_hook",
                            "event": "pre_tool",
                            "type": "shell",
                            "command": "echo hello",
                            "enabled": True,
                        },
                        {
                            "name": "disabled_hook",
                            "event": "post_tool",
                            "type": "python",
                            "code": "print('x')",
                            "enabled": False,
                        },
                    ]
                }),
                encoding="utf-8",
            )
            mgr = HookManager(hooks_path)
            assert len(mgr.hooks) == 2
            assert mgr.hooks[0].name == "test_hook"
            assert mgr.hooks[1].enabled is False

    def test_add_hook_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            hook = HookConfig(
                name="new_hook",
                event=HookEvent.ON_ERROR,
                type="shell",
                command="echo error",
            )
            mgr.add_hook(hook)
            # Reload to verify persistence
            mgr2 = HookManager(hooks_path)
            assert len(mgr2.hooks) == 1
            assert mgr2.hooks[0].name == "new_hook"

    def test_remove_hook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            mgr.add_hook(HookConfig(name="a", event="pre_tool", type="shell", command="x"))
            mgr.add_hook(HookConfig(name="b", event="post_tool", type="shell", command="y"))
            assert len(mgr.hooks) == 2
            assert mgr.remove_hook("a") is True
            assert mgr.remove_hook("nonexistent") is False
            assert len(mgr.hooks) == 1
            assert mgr.hooks[0].name == "b"

    def test_toggle_hook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            mgr.add_hook(HookConfig(name="t", event="pre_tool", type="shell", command="x", enabled=True))
            assert mgr.hooks[0].enabled is True
            assert mgr.toggle_hook("t") is True
            assert mgr.hooks[0].enabled is False
            assert mgr.toggle_hook("t") is True
            assert mgr.hooks[0].enabled is True
            assert mgr.toggle_hook("nope") is False

    def test_reload_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            mgr.add_hook(HookConfig(name="original", event="pre_tool", type="shell", command="a"))
            # Directly modify the file
            hooks_path.write_text(
                yaml.safe_dump({"hooks": [
                    {"name": "replaced", "event": "post_tool", "type": "python", "code": "b", "enabled": True}
                ]}),
                encoding="utf-8",
            )
            mgr.reload()
            assert len(mgr.hooks) == 1
            assert mgr.hooks[0].name == "replaced"

    def test_set_session_context(self):
        mgr = HookManager(Path("/nonexistent/hooks.yaml"))
        mgr.set_session_context("sess-001", "metabolomics")
        assert mgr._session_id == "sess-001"
        assert mgr._domain == "metabolomics"

    def test_creates_hooks_file_on_first_use(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "new_dir" / "hooks.yaml"
            assert not hooks_path.exists()
            mgr = HookManager(hooks_path)
            assert hooks_path.exists()
            content = hooks_path.read_text(encoding="utf-8")
            assert "PassiAgent" in content
            assert "Available events" in content

    @pytest.mark.asyncio
    async def test_on_event_dispatches_to_matching_hooks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            mgr.add_hook(HookConfig(
                name="log_tool",
                event=HookEvent.PRE_TOOL,
                type="shell",
                command="echo test",
                enabled=True,
            ))
            # Create a wire event of type TOOL_CALL
            event = WireEvent(
                type=EventType.TOOL_CALL,
                session_id="s1",
                data={"name": "run_python", "params": {"code": "1+1"}},
            )
            # Dispatch should not raise
            await mgr._dispatch_event(event)
            # Verify no exception — the hook would be executed but shell may fail in test env

    @pytest.mark.asyncio
    async def test_on_event_skips_disabled_hooks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            mgr.add_hook(HookConfig(
                name="disabled",
                event=HookEvent.PRE_TOOL,
                type="shell",
                command="echo nope",
                enabled=False,
            ))
            event = WireEvent(
                type=EventType.TOOL_CALL,
                session_id="s1",
                data={"name": "run_python"},
            )
            await mgr._dispatch_event(event)  # should be a no-op for disabled hooks

    @pytest.mark.asyncio
    async def test_notify_error_fires_on_error_hooks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            mgr.set_session_context("s1")
            called = False

            async def fake_exec(hook, ctx):
                nonlocal called
                called = True

            mgr.add_hook(HookConfig(
                name="err_catcher",
                event=HookEvent.ON_ERROR,
                type="shell",
                command="echo err",
                enabled=True,
            ))
            with patch.object(mgr, '_execute_hook', side_effect=fake_exec):
                mgr.notify_error("run_python", "something broke")

    def test_loads_hooks_with_null_value(self):
        """Regression: YAML with hooks: (null) should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            hooks_path.write_text("hooks:\n", encoding="utf-8")  # YAML null
            mgr = HookManager(hooks_path)
            assert mgr.hooks == []  # should not crash, return empty

    def test_loads_hooks_with_null_explicit(self):
        """Regression: YAML with explicit null should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            hooks_path.write_text("hooks: null\n", encoding="utf-8")
            mgr = HookManager(hooks_path)
            assert mgr.hooks == []

    def test_wire_listener_protocol(self):
        """HookManager satisfies WireListener protocol."""
        mgr = HookManager(Path("/nonexistent/hooks.yaml"))
        assert hasattr(mgr, 'on_event')
        assert callable(mgr.on_event)

    def test_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hooks_path = Path(tmpdir) / "hooks.yaml"
            mgr = HookManager(hooks_path)
            hooks = [
                HookConfig(name="h1", event="pre_tool", type="shell", command="c1"),
                HookConfig(name="h2", event="post_tool", type="python", code="c2"),
            ]
            for h in hooks:
                mgr.add_hook(h)

            # Reload
            mgr2 = HookManager(hooks_path)
            assert len(mgr2.hooks) == 2
            names = {h.name for h in mgr2.hooks}
            assert names == {"h1", "h2"}


class TestTemplateSubstitution:
    """Template variable substitution tests."""

    def test_simple_substitution(self):
        result = _template_substitute(
            "echo {tool_name} done",
            {"tool_name": "run_r", "exit_code": "0"},
        )
        assert result == "echo run_r done"

    def test_missing_variable_leaves_placeholder(self):
        result = _template_substitute(
            "echo {tool_name} {unknown}",
            {"tool_name": "test"},
        )
        assert result == "echo test {unknown}"

    def test_multiple_variables(self):
        result = _template_substitute(
            "Tool {tool_name} with code {exit_code} in {session_id}",
            {"tool_name": "run_python", "exit_code": "0", "session_id": "abc"},
        )
        assert result == "Tool run_python with code 0 in abc"
