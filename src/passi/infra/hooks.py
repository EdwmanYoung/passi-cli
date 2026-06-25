"""HookManager — user-configurable event hooks.

Hooks are shell commands or Python snippets that fire on specific events
during the agent's lifecycle. Configured via ~/.passi/hooks.yaml.

Mirrors Claude Code's hook system, tailored for bioinformatics workflows.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import yaml
from pydantic import BaseModel, Field

from passi.wire.protocol import EventType, Wire, WireEvent, WireListener

logger = logging.getLogger(__name__)

# ── Hook data models ──────────────────────────────────────────────────────


class HookType:
    SHELL = "shell"
    PYTHON = "python"


class HookEvent:
    """Event names usable in hook configuration."""

    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    ON_ERROR = "on_error"
    ON_MESSAGE = "on_message"
    ON_SESSION_START = "on_session_start"
    ON_SESSION_END = "on_session_end"

    _WIRE_MAP: dict[str, str] = {
        PRE_TOOL: EventType.TOOL_CALL,
        POST_TOOL: EventType.TOOL_RESULT,
        ON_MESSAGE: EventType.AGENT_MESSAGE,
        ON_SESSION_START: EventType.SESSION_START,
        ON_SESSION_END: EventType.SESSION_END,
        # ON_ERROR has no dedicated wire event — triggered by HookManager on error detection
    }


class HookConfig(BaseModel):
    """A single hook definition."""

    name: str
    event: str  # one of HookEvent values
    type: str = HookType.SHELL  # "shell" or "python"
    command: str = ""  # shell command (for shell type)
    code: str = ""  # Python snippet (for python type)
    enabled: bool = True


class HooksFile(BaseModel):
    """Root model for ~/.passi/hooks.yaml."""

    hooks: list[HookConfig] = Field(default_factory=list)


@dataclass
class HookContext:
    """Context variables available in hook commands/templates."""

    tool_name: str = ""
    params_json: str = "{}"
    exit_code: int = 0
    duration_ms: float = 0
    run_dir: str = ""
    session_id: str = ""
    session_domain: str = ""
    error_message: str = ""
    message_preview: str = ""
    message_count: int = 0

    def as_dict(self) -> dict[str, str]:
        return {
            "tool_name": self.tool_name,
            "params_json": self.params_json,
            "exit_code": str(self.exit_code),
            "duration_ms": str(self.duration_ms),
            "run_dir": self.run_dir,
            "session_id": self.session_id,
            "session_domain": self.session_domain,
            "error_message": self.error_message,
            "message_preview": self.message_preview,
            "message_count": str(self.message_count),
        }


# ── Hook templates (written on first use) ──────────────────────────────────

HOOKS_TEMPLATE = """\
# PassiAgent — User Hooks Configuration
# Location: ~/.passi/hooks.yaml
# Fires shell commands or Python snippets on agent lifecycle events.
#
# Available events:
#   pre_tool          Before any tool execution
#   post_tool         After tool execution completes
#   on_error          When a tool or agent step fails
#   on_message        When agent sends a text response
#   on_session_start  When a new session is created
#   on_session_end    When session ends
#
# Available context variables: {tool_name}, {params_json}, {exit_code},
#   {duration_ms}, {run_dir}, {session_id}, {session_domain},
#   {error_message}, {message_preview}, {message_count}
#
# Type: "shell" (system command) or "python" (inline snippet)

hooks:
  # - name: log_tool_calls
  #   event: pre_tool
  #   type: shell
  #   command: 'echo "[{session_id}] {tool_name} called at $(date)" >> ~/.passi/tool_calls.log'
  #   enabled: false
  #
  # - name: auto_backup_results
  #   event: post_tool
  #   type: python
  #   code: |
  #     import shutil
  #     from pathlib import Path
  #     if "{run_dir}" and Path("{run_dir}").exists():
  #         backup = Path.home() / ".passi" / "backups"
  #         backup.mkdir(parents=True, exist_ok=True)
  #         for f in Path("{run_dir}").iterdir():
  #             if f.is_file() and f.suffix in (".csv", ".pdf", ".png"):
  #                 shutil.copy2(str(f), str(backup / f.name))
  #   enabled: false
"""


# ── HookManager ───────────────────────────────────────────────────────────


class HookManager(WireListener):
    """Manage and execute user-configured event hooks.

    Usage:
        manager = HookManager(Path.home() / ".passi" / "hooks.yaml")
        wire.subscribe(manager)  # manager is a WireListener
        agent = PassiAgent(runtime)
        agent.set_hook_manager(manager)
    """

    def __init__(
        self,
        hooks_path: Path | None = None,
        *,
        wire: Wire | None = None,
    ) -> None:
        self._hooks_path = hooks_path or (Path.home() / ".passi" / "hooks.yaml")
        self._hooks: list[HookConfig] = []
        self._last_tool_error: str = ""
        self._msg_count: int = 0
        self._session_id: str = ""
        self._domain: str = ""
        self._wire: Wire | None = wire

        # Ensure hooks file exists
        self._ensure_hooks_file()
        self.reload()

    # ── Public API ──

    @property
    def hooks(self) -> list[HookConfig]:
        return list(self._hooks)

    def reload(self) -> None:
        """Reload hooks from disk."""
        try:
            if self._hooks_path.exists():
                data = yaml.safe_load(self._hooks_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "hooks" in data:
                    self._hooks = [
                        HookConfig(**h)
                        for h in data["hooks"]
                        if isinstance(h, dict)
                    ]
                else:
                    self._hooks = []
            else:
                self._hooks = []
        except Exception:
            logger.exception("Failed to load hooks from %s", self._hooks_path)
            self._hooks = []

    def save(self) -> None:
        """Persist current hooks to disk."""
        model = HooksFile(hooks=self._hooks)
        self._hooks_path.parent.mkdir(parents=True, exist_ok=True)
        self._hooks_path.write_text(
            yaml.safe_dump(model.model_dump(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def add_hook(self, hook: HookConfig) -> None:
        self._hooks.append(hook)
        self.save()

    def remove_hook(self, name: str) -> bool:
        before = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.name != name]
        if len(self._hooks) < before:
            self.save()
            return True
        return False

    def toggle_hook(self, name: str) -> bool:
        for hook in self._hooks:
            if hook.name == name:
                hook.enabled = not hook.enabled
                self.save()
                return True
        return False

    def set_session_context(self, session_id: str, domain: str = "") -> None:
        """Set session context for template variable substitution."""
        self._session_id = session_id
        self._domain = domain

    # ── WireListener implementation ──

    async def on_event(self, event: WireEvent) -> None:
        """Receive wire event and dispatch to matching hooks."""
        if event.type == EventType.AGENT_MESSAGE:
            self._msg_count += 1
        await self._dispatch_event(event)

    async def _dispatch_event(self, event: WireEvent) -> None:
        """Match wire event type to hook event names and execute matching hooks."""
        # Map wire type -> hook event names
        matching: list[str] = []
        for hook_event, wire_type in HookEvent._WIRE_MAP.items():
            if wire_type == event.type:
                matching.append(hook_event)

        if not matching:
            return

        for hook in self._hooks:
            if not hook.enabled or hook.event not in matching:
                continue

            ctx = self._build_context(event, error="")
            try:
                await self._execute_hook(hook, ctx)
            except Exception:
                logger.exception("Hook '%s' execution failed", hook.name)

    def notify_error(self, tool_name: str, error_message: str) -> None:
        """Trigger on_error hooks explicitly (no dedicated wire event)."""
        ctx = HookContext(
            tool_name=tool_name,
            error_message=error_message,
            session_id=self._session_id,
            session_domain=self._domain,
        )
        for hook in self._hooks:
            if hook.enabled and hook.event == HookEvent.ON_ERROR:
                try:
                    asyncio.create_task(self._execute_hook(hook, ctx))
                except Exception:
                    logger.exception("Hook '%s' error notification failed", hook.name)

    # ── Internal ──

    def _build_context(self, event: WireEvent, error: str) -> HookContext:
        data = event.data or {}
        return HookContext(
            tool_name=data.get("name", ""),
            params_json=_safe_json(data.get("params", {})),
            exit_code=data.get("exit_code", 0) if isinstance(data, dict) else 0,
            duration_ms=data.get("duration_ms", 0) if isinstance(data, dict) else 0,
            run_dir=data.get("run_dir", "") if isinstance(data, dict) else "",
            session_id=event.session_id or self._session_id,
            session_domain=self._domain,
            error_message=error,
            message_preview=_truncate(
                data.get("content", "") if isinstance(data, dict) else "",
                200,
            ),
            message_count=self._msg_count,
        )

    async def _execute_hook(self, hook: HookConfig, ctx: HookContext) -> None:
        """Execute a single hook."""
        context_vars = ctx.as_dict()

        if hook.type == HookType.SHELL and hook.command:
            cmd = _template_substitute(hook.command, context_vars)
            await _run_shell(cmd)

        elif hook.type == HookType.PYTHON and hook.code:
            code = _template_substitute(hook.code, context_vars)
            await _run_python_snippet(code, context_vars)

    def _ensure_hooks_file(self) -> None:
        """Create ~/.passi/hooks.yaml with template on first use."""
        self._hooks_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._hooks_path.exists():
            self._hooks_path.write_text(HOOKS_TEMPLATE, encoding="utf-8")


# ── Helpers ──


def _safe_json(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _truncate(text: Any, max_len: int) -> str:
    s = str(text)
    return s if len(s) <= max_len else s[:max_len] + "..."


def _template_substitute(template: str, variables: dict[str, str]) -> str:
    """Substitute {var} placeholders with context values.

    Uses simple string replacement (not string.Template) because hooks use
    {var} syntax, not $var. Unknown variables are left as-is.
    """
    result = template
    for key, value in variables.items():
        result = result.replace("{" + key + "}", value)
    return result


async def _run_shell(cmd: str) -> None:
    """Execute a shell command asynchronously (non-blocking)."""
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            msg = stderr.decode("utf-8", errors="replace")[:500] if stderr else ""
            logger.warning("Hook shell command failed (exit %d): %s", proc.returncode, msg)
    except asyncio.TimeoutError:
        logger.warning("Hook shell command timed out after 30s: %s", cmd[:200])
    except Exception:
        logger.exception("Hook shell command error: %s", cmd[:200])


async def _run_python_snippet(code: str, variables: dict[str, str]) -> None:
    """Execute a Python snippet in a subprocess (isolated)."""
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        for k, v in variables.items():
            env[f"HOOK_{k.upper()}"] = str(v)

        proc = await asyncio.create_subprocess_exec(
            "python",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            msg = stderr.decode("utf-8", errors="replace")[:500] if stderr else ""
            logger.warning("Hook python snippet failed (exit %d): %s", proc.returncode, msg)
    except asyncio.TimeoutError:
        logger.warning("Hook python snippet timed out after 30s")
    except Exception:
        logger.exception("Hook python snippet error")
