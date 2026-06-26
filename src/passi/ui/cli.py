"""Rich TUI-based interactive CLI for PassiAgent.

Provides a professional bioinformatics chat interface with:
- Agent mode system (chat / plan / afk)
- Skill system (metabolomics, pathway, stats, qc, multi_omics)
- Hook system (pre_tool, post_tool, on_error, etc.)
- Streaming agent responses
- Status bar with mode/skills/session info
- Keyboard shortcuts via prompt_toolkit (Ctrl+T, Ctrl+S, Ctrl+L, Ctrl+D, Alt+Enter)
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime
from html import escape
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Always
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.style import Style
from rich.table import Table

from passi.config import PassiConfig
from passi.infra.hooks import HookConfig, HookEvent, HookManager, HookType
from passi.infra.runtime import Runtime
from passi.prompts.manager import PromptManager
from passi.soul.passi_agent import PassiAgent

logger = logging.getLogger(__name__)

# Brand styles
USER_STYLE = Style(color="#3B82F6", bold=True)
AGENT_STYLE = Style(color="#F8FAFC")
TOOL_STYLE = Style(color="#10B981")
ERROR_STYLE = Style(color="#EF4444")
SYSTEM_STYLE = Style(color="#F59E0B")
HEADER_STYLE = Style(color="#2563EB", bold=True)
STATUS_STYLE = Style(color="#6B7280")

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════╗
║     P  PassiAgent  v0.2.0                            ║
║     Multi-Omics Bioinformatics Analysis Agent         ║
║                                                       ║
║  Type your analysis request or /help for commands     ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
**Keyboard Shortcuts:**
| Shortcut | Action |
|----------|--------|
| `Ctrl+T` | Cycle agent mode: chat → plan → afk → chat |
| `Ctrl+S` | Save session checkpoint |
| `Ctrl+L` | Clear screen |
| `Ctrl+D` | Exit PassiAgent (on empty input) |
| `Alt+Enter` | Insert newline for multi-line input |
| `Ctrl+C` | Interrupt agent (during execution) / Clear input (idle) |

**Agent Modes:**
| Command | Description |
|---------|-------------|
| `/mode` | Cycle mode: chat → plan → afk → chat |
| `/mode [chat|plan|afk]` | Switch to a specific mode |
| `/plan show` | Display the current analysis plan |
| `/plan approve` | Approve plan for execution (plan mode) |
| `/plan reject <feedback>` | Reject plan with feedback for revision |

**Control:**
| Command | Description |
|---------|-------------|
| `/interrupt` | Interrupt a running analysis task |

**Skills:**
| Command | Description |
|---------|-------------|
| `/skill list` | Show all available skills |
| `/skill use <name,...>` | Activate skills (metabolomics, pathway, stats, qc, multi_omics) |
| `/skill off` | Deactivate all skills |
| `/skill show` | Show currently active skills |

**Hooks:**
| Command | Description |
|---------|-------------|
| `/hook list` | Show all configured hooks |
| `/hook add` | Add a new hook interactively |
| `/hook remove <name>` | Remove a hook |
| `/hook toggle <name>` | Enable/disable a hook |

**Session:**
| Command | Description |
|---------|-------------|
| `/status` | Show agent status and statistics |
| `/config` | Show current configuration |
| `/save <name>` | Save session checkpoint |
| `/clear` | Clear conversation context |
| `/export` | Export session as chat log |
| `/sessions list` | List available sessions |
| `/sessions load <id>` | Load a different session |
| `/domain <name>` | Switch analysis domain |

**Other:**
| `/help` | Show this help |
| `/quit` or `/exit` | Exit PassiAgent |
"""

_MODE_CYCLE = ["chat", "plan", "afk"]
_MODE_LABELS = {"chat": "[chat]", "plan": "[plan]", "afk": "[afk]"}

# Sentinel values returned by prompt_toolkit key bindings (via event.app.exit)
_SENTINEL_CYCLE_MODE = "\x00mode"
_SENTINEL_SAVE = "\x00save"
_SENTINEL_CLEAR_SCREEN = "\x00clear_screen"
_SENTINEL_QUIT = "\x00quit"
_SENTINEL_CUSTOM_INPUT = "\x00custom"

# prompt_toolkit style for the input prompt
_PROMPT_STYLE = PTStyle.from_dict({
    "prompt": "#3B82F6 bold",
    "rprompt": "#6B7280 italic",
})


class PassiCLI:
    """Interactive CLI chat interface for PassiAgent."""

    def __init__(self, config: PassiConfig) -> None:
        self.config = config
        self.console = Console()
        self.runtime = Runtime(config)
        self.agent: PassiAgent | None = None
        self._domain: str = "multi-omics"
        self._running: bool = True
        self._start_mode: str | None = None
        self._start_skills: list[str] | None = None
        self._resume_session_id: str | None = None
        self._input_history = InMemoryHistory()
        self._resize_task: asyncio.Task | None = None
        self._prompt_session: PromptSession[str] | None = None
        self._render_buffer: list[tuple[str, tuple[Any, ...]]] = []
        self._max_buffer_size = 200

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize and start the interactive CLI."""
        self.console.clear()
        self.console.print(WELCOME_BANNER, style=HEADER_STYLE)
        self._buffer_add("banner")

        # Start terminal resize monitor for auto-refresh
        self._resize_task = asyncio.create_task(self._resize_monitor())

        existing_sessions = self.runtime.session.list_sessions()
        force_select = self._resume_session_id == "__select__"

        # Determine which session to use
        if self._resume_session_id and self._resume_session_id != "__select__":
            # CLI flag specified with ID — load directly
            await self._load_existing_session(self._resume_session_id)
        elif force_select or existing_sessions:
            # Sessions exist — let user choose
            self._print_system(
                f"Found {len(existing_sessions)} existing session(s) in this project."
            )
            options = ["New Session"]
            for s in existing_sessions:
                options.append(self._format_session_entry(s))

            choice = await self._get_selection(options)
            if choice == "New Session" or not choice:
                # Start fresh
                await self._start_new_session()
            else:
                # Extract session_id from the formatted entry
                sid = choice.split(" | ")[0].strip()
                await self._load_existing_session(sid)
        else:
            await self._start_new_session()

        # Apply startup mode/skills from CLI flags
        if self._start_mode:
            plan_first = (self._start_mode == "plan")
            self.agent.set_mode(mode=self._start_mode, plan_first=plan_first, skills=self._start_skills)
        elif getattr(self.config, "afk_mode", False):
            self.agent.set_mode("afk")
        elif self._start_skills:
            self.agent.set_mode(skills=self._start_skills)

        self._print_system(
            f"Session: {self.runtime.session.active_session.session_id}  |  "
            f"Domain: {self._domain}  |  Provider: {self.config.default_provider}"
        )
        self._print_status_bar()

        # Main loop
        try:
            await self._repl()
        finally:
            pass

    async def _start_new_session(self) -> None:
        """Create a new session and initialize the agent."""
        self.runtime.session.create_session(domain=self._domain)
        self.agent = PassiAgent(self.runtime)
        await self.agent.initialize()

    async def _load_existing_session(self, session_id: str) -> None:
        """Load an existing session and restore conversation context.

        Loads session metadata, creates and initializes the agent, then replays
        user/assistant messages from the session's wire log into the context.
        """
        try:
            meta = self.runtime.session.load_session(session_id)
        except FileNotFoundError:
            self._print_error(f"Session not found: {session_id}")
            await self._start_new_session()
            return
        except Exception as exc:
            self._print_error(f"Failed to load session {session_id}: {exc}")
            await self._start_new_session()
            return
        self._domain = meta.domain
        self._print_system(f"Loading session: {session_id}")

        # Create and initialize agent (plan is auto-loaded by initialize)
        self.agent = PassiAgent(self.runtime)
        await self.agent.initialize()

        # Restore conversation context from wire log
        session_dir = self.runtime.session.get_session_dir()
        wire_path = session_dir / "wire.jsonl"
        if wire_path.exists():
            from passi.wire.persistence import WirePersistence
            wp = WirePersistence(wire_path)
            events = wp.read_session(session_id)
            restored = 0
            pending_tool_results: list[dict[str, Any]] = []
            for event in events:
                try:
                    if event.type == "user_message":
                        # Flush pending tool results before a new user turn
                        if pending_tool_results:
                            self.runtime.context.add_message("tool_results", pending_tool_results)
                            pending_tool_results = []
                        content = event.data.get("content", "")
                        if content:
                            self.runtime.context.add_message("user", content)
                            display = self._extract_display_text(content)
                            if display:
                                self._print_user(display)
                            restored += 1
                    elif event.type == "tool_result":
                        result_data = event.data.get("result", {})
                        result_text = json.dumps(result_data, ensure_ascii=False, default=str)
                        pending_tool_results.append({
                            "tool_use_id": event.data.get("tool_use_id", ""),
                            "content": result_text,
                        })
                    elif event.type == "agent_message":
                        content = event.data.get("content", "")
                        if content:
                            # Check if we can properly pair tool_use blocks with tool_results.
                            # Old wire sessions lack tool_use_id in TOOL_RESULT events.
                            can_pair = self._can_pair_tool_results(content, pending_tool_results)
                            if can_pair:
                                self.runtime.context.add_message("assistant", content)
                                if pending_tool_results:
                                    self.runtime.context.add_message("tool_results", pending_tool_results)
                                    pending_tool_results = []
                            else:
                                clean = self._strip_tool_use_blocks(content)
                                if clean:
                                    self.runtime.context.add_message("assistant", clean)
                                pending_tool_results = []  # discard unpaired results
                            display = self._extract_display_text(content)
                            if display:
                                self._print_agent(display)
                            restored += 1
                except Exception:
                    pass
            # Flush any remaining tool results
            if pending_tool_results:
                self.runtime.context.add_message("tool_results", pending_tool_results)
            self._print_system(
                f"Session loaded: {session_id}  |  "
                f"Domain: {meta.domain}  |  "
                f"Messages restored: {restored}"
            )
        else:
            self._print_system(
                f"Session loaded: {session_id}  |  "
                f"Domain: {meta.domain}  |  "
                f"No wire log found (context is empty)"
            )

    @staticmethod
    def _format_session_entry(session: dict[str, Any]) -> str:
        """Format a session summary for display in the selection list."""
        sid = session["session_id"]
        domain = session.get("domain", "multi-omics")
        msg_count = session.get("message_count", 0)
        created = session.get("created_at", "")
        # Format the ISO timestamp to a shorter form
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(created)
            created_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            created_str = created[:16] if created else "unknown"
        return f"{sid} | {domain} | {msg_count} msgs | {created_str}"

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        if self._resize_task:
            self._resize_task.cancel()
        if self.agent:
            await self.agent.shutdown()
        self.console.print("\n[dim]Session ended.[/dim]")

    def _buffer_add(self, entry_type: str, *args: Any) -> None:
        """Record an output entry for replay on terminal resize."""
        self._render_buffer.append((entry_type, args))
        if len(self._render_buffer) > self._max_buffer_size:
            self._render_buffer = self._render_buffer[-self._max_buffer_size:]

    def _repaint(self) -> None:
        """Clear screen and re-render all buffered content at current width."""
        self.console.clear()
        for entry_type, args in self._render_buffer:
            if entry_type == "banner":
                self.console.print(WELCOME_BANNER, style=HEADER_STYLE)
            elif entry_type == "user":
                self.console.print(Panel(args[0], style=USER_STYLE, title="You", title_align="left"))
            elif entry_type == "agent":
                self.console.print(Panel(
                    Markdown(args[0]), style=AGENT_STYLE,
                    title="PassiAgent", title_align="left",
                ))
            elif entry_type == "tool_call":
                self.console.print(f"🔧 {args[0]}({args[1]})", style=TOOL_STYLE)
            elif entry_type == "tool_result":
                self.console.print(f"  └─ {args[0]}", style=Style(color="#6B7280", dim=True))
            elif entry_type == "system":
                if len(args) > 1 and args[1]:
                    self.console.print(args[0])
                else:
                    self.console.print(args[0], style=SYSTEM_STYLE)
            elif entry_type == "error":
                self.console.print(f"✗ {args[0]}", style=ERROR_STYLE)
            elif entry_type == "status_bar":
                self.console.print(args[0], style=STATUS_STYLE)

    async def _resize_monitor(self) -> None:
        """Poll terminal size; force Rich and prompt_toolkit to adapt on resize."""
        while self._running:
            await asyncio.sleep(0.5)
            try:
                current_columns = shutil.get_terminal_size().columns
            except Exception:
                continue
            if current_columns != self.console.width:
                # Clear Rich's cached size so it re-queries the OS on every render
                self.console._width = None
                self.console._height = None
                # Force prompt_toolkit to re-render the prompt layout at new width
                try:
                    from prompt_toolkit.application import get_app_or_none
                    app = get_app_or_none()
                    if app is not None:
                        app._on_resize()
                except Exception:
                    pass
                # Re-render buffered content at new width when idle
                if not (self.agent and self.agent.agent_busy):
                    self._repaint()

    # ── Main REPL ──────────────────────────────────────────────────────

    async def _repl(self) -> None:
        """Read-Eval-Print loop with keyboard shortcut support."""
        while self._running:
            try:
                user_input = await self._get_input()

                # Handle keyboard shortcut sentinels (returned by prompt_toolkit key bindings)
                if user_input == _SENTINEL_CYCLE_MODE:
                    if self.agent:
                        self._cycle_mode()
                        self._print_status_bar()
                    continue
                elif user_input == _SENTINEL_SAVE:
                    await self._cmd_save("")
                    continue
                elif user_input == _SENTINEL_CLEAR_SCREEN:
                    self.console.clear()
                    continue
                elif user_input == _SENTINEL_QUIT:
                    self._running = False
                    continue

                if not user_input.strip():
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    try:
                        await self._handle_command(user_input)
                    except Exception as exc:
                        self._print_error(f"Command error: {exc}")
                    continue

                # Chat message
                self._print_user(user_input)
                await self._process_message(user_input)

            except KeyboardInterrupt:
                self._print_system("Interrupted. Type /quit to exit.")
            except EOFError:
                break

        await self._shutdown()

    def _create_input_session(
        self, input: Any = None, output: Any = None
    ) -> PromptSession[str]:
        """Create a prompt_toolkit PromptSession with key bindings.

        Follows kimi-cli pattern: all key bindings are registered on a single
        KeyBindings object, no background threads, no stdin conflict.

        input/output params allow test injection via create_pipe_input().
        """
        kb = KeyBindings()

        @kb.add("c-t", eager=True)
        def _(event: Any) -> None:
            """Ctrl+T: cycle agent mode (chat -> plan -> afk -> chat)."""
            event.app.exit(result=_SENTINEL_CYCLE_MODE)

        @kb.add("c-s", eager=True)
        def _(event: Any) -> None:
            """Ctrl+S: save session checkpoint."""
            event.app.exit(result=_SENTINEL_SAVE)

        @kb.add("c-l", eager=True)
        def _(event: Any) -> None:
            """Ctrl+L: clear screen."""
            event.app.exit(result=_SENTINEL_CLEAR_SCREEN)

        @kb.add("escape", "enter", eager=True)
        def _(event: Any) -> None:
            """Alt+Enter: insert newline for multi-line input."""
            event.current_buffer.insert_text("\n")

        @kb.add("c-d")
        def _(event: Any) -> None:
            """Ctrl+D: exit on empty input, otherwise delete forward."""
            if not event.current_buffer.text:
                event.app.exit(result=_SENTINEL_QUIT)
            else:
                event.current_buffer.delete()

        @kb.add("c-c")
        def _(event: Any) -> None:
            """Ctrl+C: interrupt agent if busy, otherwise clear current input."""
            if self.agent and self.agent.agent_busy:
                self.agent.interrupt()
                event.current_buffer.reset()
                event.app.exit(result="")
            else:
                event.current_buffer.reset()
                event.app.exit(result="")

        # Build the prompt message function (dynamic — shows current mode and phase)
        def _message() -> HTML:
            mode = self.agent.mode if self.agent else "chat"
            label = _MODE_LABELS.get(mode, "[chat]")

            # Override label for specific phases
            if self.agent and self.agent.agent_busy:
                label = f"[{mode}] ●"
            elif self.agent and hasattr(self.agent, '_step_confirm_mode') and self.agent._step_confirm_mode:
                label = f"[plan-step]"
            elif self.agent and hasattr(self.agent, '_plan_qa_active') and self.agent._plan_qa_active:
                label = f"[plan-qa]"

            return HTML(f"<prompt>{label} &gt; </prompt>")

        def _rprompt() -> HTML | None:
            """Right-side prompt: active skills hint."""
            if self.agent and self.agent.active_skills:
                skills = ",".join(escape(s) for s in self.agent.active_skills)
                return HTML(f"<rprompt>skills:{skills}</rprompt>")
            return None

        return PromptSession(
            key_bindings=kb,
            style=_PROMPT_STYLE,
            message=_message,
            rprompt=_rprompt,
            history=self._input_history,
            complete_while_typing=False,
            input=input,
            output=output,
        )

    async def _get_input(self) -> str:
        """Get user input via prompt_toolkit PromptSession with keyboard shortcuts.

        Returns the user's text, or a sentinel string (_SENTINEL_*) if a
        keyboard shortcut was triggered. The REPL loop dispatches sentinels
        to the appropriate action.

        Falls back to basic input() when not running in a real console
        (e.g., piped stdin, IDE terminal, test environments).
        """
        try:
            if self._prompt_session is None:
                self._prompt_session = self._create_input_session()
            result = await self._prompt_session.prompt_async()
            return result
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return _SENTINEL_QUIT
        except Exception:
            # Fallback for non-console environments (IDE, pipeline, test)
            # prompt_toolkit requires a real terminal; fall back to basic input()
            return await self._get_input_fallback()

    async def _get_input_fallback(self) -> str:
        """Basic input() fallback when prompt_toolkit is unavailable."""
        loop = asyncio.get_running_loop()
        mode = self.agent.mode if self.agent else "chat"
        label = _MODE_LABELS.get(mode, "[chat]")

        try:
            return await loop.run_in_executor(
                None,
                lambda: input(f"{label} > "),
            )
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return _SENTINEL_QUIT

    async def _get_user_input(self, prompt_text: str) -> str:
        """Get user input with a custom prompt label, delegating to _get_input."""
        self._print_system(prompt_text)
        return await self._get_input()

    async def _get_selection(self, options: list[str] | None) -> str:
        """Present options as an interactive list with arrow key navigation.

        Up/down arrows move the selection highlight. Enter confirms the choice.
        Number keys 1-9 select directly. The last option is always "Custom input..."
        which returns a sentinel so the caller can switch to free-text mode.

        Returns the selected option text, _SENTINEL_CUSTOM_INPUT for custom input,
        or empty string if cancelled.
        """
        if options is None:
            return ""
        display_options = list(options)
        display_options.append("Custom input...")

        selected = [0]  # mutable closure for key binding access

        kb = KeyBindings()

        @kb.add("up", eager=True)
        def _(event: Any) -> None:
            selected[0] = (selected[0] - 1) % len(display_options)
            event.app.invalidate()

        @kb.add("down", eager=True)
        def _(event: Any) -> None:
            selected[0] = (selected[0] + 1) % len(display_options)
            event.app.invalidate()

        @kb.add("enter", eager=True)
        def _(event: Any) -> None:
            if selected[0] == len(display_options) - 1:
                event.app.exit(result=_SENTINEL_CUSTOM_INPUT)
            else:
                event.app.exit(result=options[selected[0]])

        # Number keys for direct selection
        for i in range(min(len(display_options), 9)):
            @kb.add(str(i + 1), eager=True)
            def _(event: Any, idx: int = i) -> None:
                if idx == len(display_options) - 1:
                    event.app.exit(result=_SENTINEL_CUSTOM_INPUT)
                else:
                    event.app.exit(result=options[idx])

        @kb.add("c-c", eager=True)
        def _(event: Any) -> None:
            event.app.exit(result="")

        def _bottom_toolbar() -> HTML:
            lines = []
            for i, opt in enumerate(display_options):
                safe_opt = escape(str(opt))
                if i == selected[0]:
                    lines.append(f"<b>> {i + 1}. {safe_opt}</b>")
                else:
                    lines.append(f"  {i + 1}. {safe_opt}")
            return HTML("\n".join(lines))

        def _message() -> HTML:
            mode = self.agent.mode if self.agent else "chat"
            label = _MODE_LABELS.get(mode, "[chat]")
            if self.agent and self.agent.agent_busy:
                label = f"[{mode}] ●"
            return HTML(f"<prompt>{label} &gt; [select] </prompt>")

        try:
            session = PromptSession(
                key_bindings=kb,
                style=_PROMPT_STYLE,
                message=_message,
                bottom_toolbar=_bottom_toolbar,
            )
            session.default_buffer.read_only = Always()
            result: str = await session.prompt_async()
            return result
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return ""
        except Exception:
            # Fallback: prompt_toolkit can't drive the terminal (non-TTY, IDE, etc.)
            # Print a numbered list and use basic input()
            return await self._get_selection_fallback(options)

    async def _get_selection_fallback(self, options: list[str]) -> str:
        """Basic input() fallback when prompt_toolkit can't render selection UI.

        Prints a numbered list and reads the user's choice from stdin.
        """
        for i, opt in enumerate(options):
            self.console.print(f"  {i + 1}. {opt}")
        self.console.print(f"  {len(options) + 1}. Custom input...")

        loop = asyncio.get_running_loop()
        try:
            choice = await loop.run_in_executor(
                None,
                lambda: input("Enter number (or empty to cancel): "),
            )
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return ""

        if not choice.strip():
            return ""
        try:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(options):
                return options[idx]
            elif idx == len(options):
                return _SENTINEL_CUSTOM_INPUT
            else:
                self._print_system(f"Invalid choice: {choice}. Please enter 1-{len(options) + 1}.")
                return ""
        except ValueError:
            self._print_system(f"Invalid input: {choice}. Please enter a number.")
            return ""

    # ── Message Processing ─────────────────────────────────────────────

    async def _process_message(self, message: str) -> None:
        """Process a user message through the agent with streaming.

        Handles pending_question events from ask_user tool (plan Q&A, step confirmation).
        After answering, loops back to process the answer through the agent.
        """
        assert self.agent is not None

        response_text: list[str] = []
        tool_calls_shown: set[str] = set()
        pending_question: dict[str, Any] | None = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("[dim]Analyzing...[/dim]", total=None)

            try:
                stream = self.agent.chat_stream(message)
                async for event in stream:
                    if event.type == "thinking":
                        progress.update(task, description=f"[dim]{event.content}[/dim]")
                    elif event.type == "text":
                        progress.stop()
                        response_text.append(event.content)
                        self._print_agent(event.content)
                        progress.start()
                        progress.update(task, description="[dim]Continuing...[/dim]")
                    elif event.type == "tool_call":
                        tc_key = f"{event.tool_name}:{event.content}"
                        if tc_key not in tool_calls_shown:
                            tool_calls_shown.add(tc_key)
                            progress.stop()
                            self._print_tool_call(event.tool_name or "", event.content or "")
                            progress.start()
                    elif event.type == "tool_result":
                        progress.stop()
                        self._print_tool_result(event.tool_name or "", event.content or "")
                        progress.start()
                    elif event.type == "error":
                        progress.stop()
                        self._print_error(f"Tool error: {event.content}")
                        progress.start()
                    elif event.type == "pending_question":
                        pending_question = {
                            "question": event.content,
                            "context": event.metadata.get("context", ""),
                            "options": event.metadata.get("options"),
                        }
            except Exception as e:
                progress.stop()
                self._print_error(f"Agent error: {e}")
                return

            progress.stop()

        if not response_text and not tool_calls_shown and not pending_question:
            self._print_agent("(no response)")

        # Handle pending question (from ask_user in plan Q&A or step confirmation)
        if pending_question:
            await self._handle_pending_question(pending_question)

        self._print_status_bar()

    async def _handle_pending_question(self, question: dict[str, Any]) -> None:
        """Present a pending question to the user and route the answer back to the agent."""
        # Bug 8: recursion depth guard — avoid infinite recursion from nested questions
        depth = getattr(self, "_pending_question_depth", 0) + 1
        self._pending_question_depth = depth
        if depth > 5:
            self._print_error("Too many nested questions — aborting question chain.")
            self._pending_question_depth = depth - 1
            return

        try:
            q_text = question.get("question", "")
            q_context = question.get("context", "")
            q_options = question.get("options") or []

            # Display the question
            if q_context:
                self._print_system(f"[dim]{q_context}[/dim]")
            self.console.print(
                Panel(q_text, style=SYSTEM_STYLE, title="PassiAgent asks", title_align="left")
            )

            # Get answer from user — use selection UI when options are provided
            if q_options:
                answer = await self._get_selection(q_options)
                if answer == _SENTINEL_CUSTOM_INPUT:
                    answer = await self._get_user_input("Your answer: ")
                elif not answer:
                    # Bug 9: Cancel during selection (empty string) — don't proceed
                    self._print_system("Question cancelled.")
                    return
            else:
                answer = await self._get_user_input("Your answer: ")

            # Bug 7: Sentinel leakage — check for sentinel values before processing
            if answer and answer.startswith("\x00"):
                if answer == _SENTINEL_QUIT:
                    self._running = False
                elif answer == _SENTINEL_CLEAR_SCREEN:
                    self.console.clear()
                # For other sentinels, silently skip
                return

            if not answer.strip():
                answer = "(no answer)"

            self._print_user(f"[response] {answer}")

            # Route answer back to agent for further processing
            await self._process_message(answer)
        finally:
            self._pending_question_depth = depth - 1

    # ── Command Handling ───────────────────────────────────────────────

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/save": self._cmd_save,
            "/export": self._cmd_export,
            "/domain": self._cmd_domain,
            "/methods": self._cmd_methods,
            "/formats": self._cmd_formats,
            "/mode": self._cmd_mode,
            "/skill": self._cmd_skill,
            "/hook": self._cmd_hook,
            "/status": self._cmd_status,
            "/config": self._cmd_config,
            "/sessions": self._cmd_sessions,
            "/plan": self._cmd_plan,
            "/interrupt": self._cmd_interrupt,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler:
            await handler(arg)
        else:
            self._print_system(f"Unknown command: {cmd}. Type /help for available commands.")

    # ── Individual Command Handlers ────────────────────────────────────

    async def _cmd_help(self, _: str) -> None:
        self._print_system(Markdown(HELP_TEXT))

    async def _cmd_clear(self, _: str) -> None:
        if self.agent:
            await self.agent.reset()
        self._print_system("Context cleared.")

    async def _cmd_save(self, arg: str) -> None:
        name = arg or f"checkpoint_{datetime.now():%Y%m%d_%H%M%S}"
        state = {"domain": self._domain, "timestamp": datetime.now().isoformat()}
        self.runtime.session.checkpoint(state)
        self._print_system(f"Session checkpoint saved: {name}")

    async def _cmd_export(self, _: str) -> None:
        from passi.wire.persistence import WirePersistence

        session = self.runtime.session.active_session
        if session:
            wire_path = self.runtime.session.get_session_dir() / "wire.jsonl"
            persistence = WirePersistence(wire_path)
            chatlog = persistence.export_chatlog(session.session_id)
            export_dir = self.config.result_dir
            export_dir.mkdir(parents=True, exist_ok=True)
            report_path = export_dir / f"chatlog_{session.session_id}.md"
            report_path.write_text(chatlog, encoding="utf-8")
            self._print_system(f"Chat log exported: {report_path}")

    async def _cmd_sessions(self, arg: str) -> None:
        """Handle /sessions list|load commands."""
        parts = arg.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts and parts[0] else "list"
        subarg = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            sessions = self.runtime.session.list_sessions()
            if not sessions:
                self._print_system("No saved sessions found.")
                return
            table = Table(title="Sessions", style=SYSTEM_STYLE, border_style="dim")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Domain")
            table.add_column("Messages", justify="right")
            table.add_column("Created")
            for s in sessions:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(s.get("created_at", ""))
                    created_str = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    created_str = s.get("created_at", "")[:16]
                table.add_row(
                    s["session_id"],
                    s.get("domain", "multi-omics"),
                    str(s.get("message_count", 0)),
                    created_str,
                )
            self.console.print(table)

        elif subcmd == "load":
            if not subarg:
                self._print_error("Usage: /sessions load <session_id>")
                return
            self._print_system(f"Loading session: {subarg} ...")
            await self._load_existing_session(subarg)
            self._print_system(
                f"Session: {self.runtime.session.active_session.session_id}  |  "
                f"Domain: {self._domain}"
            )
            self._print_status_bar()

        else:
            self._print_error(f"Unknown subcommand: {subcmd}. Use 'list' or 'load <id>'.")

    async def _cmd_domain(self, arg: str) -> None:
        if arg:
            self._domain = arg
            self._print_system(f"Domain set to: {self._domain}")
        else:
            self._print_system(f"Current domain: {self._domain}")

    async def _cmd_methods(self, arg: str) -> None:
        from passi.knowledge.methods import get_methods_by_domain

        domain = arg or self._domain
        methods = get_methods_by_domain(domain)
        if methods:
            text = "\n".join(
                f"- **{mid}**: {m['name']} ({m['backend']})" for mid, m in methods.items()
            )
            self._print_system(Markdown(f"**Methods for {domain}:**\n{text}"))
        else:
            self._print_system(f"No methods found for domain: {domain}")

    async def _cmd_formats(self, arg: str) -> None:
        from passi.knowledge.formats import get_formats_by_domain

        domain = arg or self._domain
        formats = get_formats_by_domain(domain)
        if formats:
            text = "\n".join(
                f"- **{f['format']}** ({', '.join(f['suffixes'])}): {f['description']}"
                for f in formats
            )
            self._print_system(Markdown(f"**Formats for {domain}:**\n{text}"))
        else:
            self._print_system(f"No formats found for domain: {domain}")

    async def _cmd_interrupt(self, _: str) -> None:
        """Interrupt the currently running agent operation."""
        if self.agent is None:
            return
        if self.agent.agent_busy:
            self.agent.interrupt()
            self._print_system("Interrupt signal sent. Waiting for agent to respond...")
        else:
            self._print_system("No operation in progress to interrupt.")

    async def _cmd_quit(self, _: str) -> None:
        self._running = False
        self._print_system("Goodbye!")

    # ── Mode Commands ──────────────────────────────────────────────────

    async def _cmd_mode(self, arg: str) -> None:
        """Switch agent mode: /mode [chat|plan|afk].  No argument = cycle."""
        assert self.agent is not None
        valid = {"chat", "plan", "afk"}

        if not arg:
            # Cycle to next mode
            self._cycle_mode()
            return

        mode = arg.strip().lower()
        if mode == self.agent.mode:
            self._print_system(f"Already in [bold]{mode}[/bold] mode.")
            return
        if mode not in valid:
            self._print_system(f"Invalid mode '{mode}'. Choose: {', '.join(valid)}")
            return

        self.agent.set_mode(mode=mode, plan_first=(mode == "plan"))
        self._print_system(f"Mode switched to: [bold]{mode}[/bold]")
        self._print_status_bar()

    def _cycle_mode(self) -> None:
        """Cycle to the next mode."""
        if self.agent is None:
            return
        current = self.agent.mode
        try:
            idx = _MODE_CYCLE.index(current)
            next_idx = (idx + 1) % len(_MODE_CYCLE)
        except ValueError:
            next_idx = 0
        next_mode = _MODE_CYCLE[next_idx]
        self.agent.set_mode(mode=next_mode, plan_first=(next_mode == "plan"))
        self._print_system(f"Mode cycled: [bold]{next_mode}[/bold]")
        self._print_status_bar()

    # ── Skill Commands ─────────────────────────────────────────────────

    async def _cmd_skill(self, arg: str) -> None:
        """Manage skills: /skill [list|use <names>|off|show]"""
        assert self.agent is not None
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if sub == "list":
            available = PromptManager.available_skills()
            active = self.agent.active_skills
            lines = []
            for s in available:
                marker = "[✓]" if s in active else "[ ]"
                lines.append(f"  {marker} **{s}**")
            self._print_system(Markdown(f"**Available Skills:**\n" + "\n".join(lines)))

        elif sub == "use":
            if not sub_arg:
                self._print_system("Usage: /skill use <name,...>  (e.g., /skill use metabolomics,stats)")
                return
            names = [n.strip() for n in sub_arg.split(",")]
            for name in names:
                if self.agent._prompt_manager is not None:
                    ok = self.agent._prompt_manager.load_skill(name)
                    if ok:
                        self._print_system(f"Skill activated: [bold]{name}[/bold]")
                    else:
                        self._print_error(f"Unknown skill: {name}. Use /skill list to see available skills.")
            # Rebuild system prompt with new skills
            self._rebuild_prompt()
            self._print_status_bar()

        elif sub == "off":
            self.agent._prompt_manager.clear_skills() if self.agent._prompt_manager else None
            self._rebuild_prompt()
            self._print_system("All skills deactivated.")
            self._print_status_bar()

        elif sub == "show":
            active = self.agent.active_skills
            if active:
                self._print_system(f"Active skills: [bold]{', '.join(active)}[/bold]")
            else:
                self._print_system("No skills active. Use /skill list to see available skills.")
        else:
            self._print_system("Usage: /skill [list|use <name,...>|off|show]")

    # ── Hook Commands ──────────────────────────────────────────────────

    async def _cmd_hook(self, arg: str) -> None:
        """Manage hooks: /hook [list|add|remove|toggle]"""
        hm = self.agent.get_hook_manager() if self.agent else None
        if hm is None:
            self._print_error("Hook manager not available.")
            return

        parts = arg.split(maxsplit=2)
        sub = parts[0].lower() if parts else ""
        sub_arg1 = parts[1] if len(parts) > 1 else ""
        sub_arg2 = parts[2] if len(parts) > 2 else ""

        if sub == "list":
            hooks = hm.hooks
            if not hooks:
                self._print_system("No hooks configured. Use /hook add to create one.")
                return
            table = Table(title="Configured Hooks")
            table.add_column("Name", style="cyan")
            table.add_column("Event", style="green")
            table.add_column("Type")
            table.add_column("Enabled")
            for h in hooks:
                status = "[✓]" if h.enabled else "[ ]"
                table.add_row(h.name, h.event, h.type, status)
            self.console.print(table)

        elif sub == "add":
            await self._cmd_hook_add_interactive()

        elif sub == "remove":
            if not sub_arg1:
                self._print_system("Usage: /hook remove <name>")
                return
            if hm.remove_hook(sub_arg1):
                self._print_system(f"Hook removed: {sub_arg1}")
            else:
                self._print_error(f"Hook not found: {sub_arg1}")

        elif sub == "toggle":
            if not sub_arg1:
                self._print_system("Usage: /hook toggle <name>")
                return
            if hm.toggle_hook(sub_arg1):
                self._print_system(f"Hook toggled: {sub_arg1}")
            else:
                self._print_error(f"Hook not found: {sub_arg1}")

        elif sub == "test":
            if not sub_arg1:
                self._print_system("Usage: /hook test <name>")
                return
            self._print_system("Hook testing not yet implemented. Use the hook in a real session.")
        else:
            self._print_system("Usage: /hook [list|add|remove|toggle <name>]")

    async def _cmd_hook_add_interactive(self) -> None:
        """Interactive hook creation wizard."""
        hm = self.agent.get_hook_manager() if self.agent else None
        if hm is None:
            return

        loop = asyncio.get_running_loop()

        # Step 1: Name
        name = await loop.run_in_executor(
            None,
            lambda: Prompt.ask("  Hook name", console=self.console),
        )
        if not name.strip():
            self._print_system("Cancelled.")
            return

        # Step 2: Event
        events = [
            HookEvent.PRE_TOOL,
            HookEvent.POST_TOOL,
            HookEvent.ON_ERROR,
            HookEvent.ON_MESSAGE,
            HookEvent.ON_SESSION_START,
            HookEvent.ON_SESSION_END,
        ]
        self.console.print("  Events: " + ", ".join(f"[bold]{e}[/bold]" for e in events))
        event = await loop.run_in_executor(
            None,
            lambda: Prompt.ask("  Event", console=self.console),
        )
        if event not in events:
            self._print_error(f"Invalid event. Choose from: {', '.join(events)}")
            return

        # Step 3: Type
        hook_type = await loop.run_in_executor(
            None,
            lambda: Prompt.ask("  Type [shell/python]", default="shell", console=self.console),
        )
        if hook_type not in (HookType.SHELL, HookType.PYTHON):
            self._print_error("Type must be 'shell' or 'python'")
            return

        # Step 4: Command / Code
        if hook_type == HookType.SHELL:
            command = await loop.run_in_executor(
                None,
                lambda: Prompt.ask("  Shell command", console=self.console),
            )
            code = ""
        else:
            command = ""
            self.console.print("  Python code (end with empty line):")
            code_lines: list[str] = []
            while True:
                line = await loop.run_in_executor(
                    None,
                    lambda: Prompt.ask("    ", console=self.console),
                )
                if not line:
                    break
                code_lines.append(line)
            code = "\n".join(code_lines)

        hook = HookConfig(
            name=name.strip(),
            event=event,
            type=hook_type,
            command=command,
            code=code,
            enabled=True,
        )
        hm.add_hook(hook)
        self._print_system(f"Hook [bold]{name}[/bold] added and enabled.")

    # ── Status & Config Commands ───────────────────────────────────────

    async def _cmd_status(self, _: str) -> None:
        """Show agent status overview."""
        assert self.agent is not None

        session = self.runtime.session.active_session
        plan = self.agent.get_plan()
        tasks = self.agent.get_tasks()

        table = Table(title="Agent Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Mode", self.agent.mode)
        table.add_row("Plan-First", str(self.agent.plan_first))
        table.add_row("Domain", self._domain)
        table.add_row("Provider", self.config.default_provider)
        table.add_row("Session ID", session.session_id if session else "N/A")
        table.add_row("Skills", ", ".join(self.agent.active_skills) or "(none)")
        table.add_row("Plan Steps", str(len(plan.steps) if plan else 0))
        table.add_row("Tasks Executed", str(len(tasks)))
        self.console.print(table)

    async def _cmd_config(self, arg: str) -> None:
        """Show or set configuration."""
        parts = arg.split(maxsplit=2) if arg else []

        if not arg:
            # Show summary
            table = Table(title="Current Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value")
            table.add_row("Default Provider", self.config.default_provider)
            table.add_row("Anthropic Model", self.config.anthropic.model)
            table.add_row("Anthropic Base URL", self.config.anthropic.base_url or "(default)")
            table.add_row("Max Tokens", str(self.config.anthropic.max_tokens))
            table.add_row("Thinking Budget", str(self.config.anthropic.thinking_budget_tokens))
            table.add_row("Timeout (s)", str(self.config.execution.timeout_seconds))
            table.add_row("R Home", self.config.execution.r_home or "(auto)")
            table.add_row("AFK Mode", str(self.config.afk_mode))
            table.add_row("Debug", str(self.config.debug))
            table.add_row("Data Dir", str(self.config.data_dir))
            table.add_row("Result Dir", str(self.config.result_dir))
            self.console.print(table)
        elif len(parts) >= 2 and parts[0] == "set":
            self._print_system("Config setting via TUI is not persisted. Use .env or settings.yaml for permanent changes.")
        else:
            self._print_system("Usage: /config  or  /config set <key> <value>")

    # ── Plan Commands ──────────────────────────────────────────────────

    async def _cmd_plan(self, arg: str) -> None:
        """Plan management: /plan [show|approve|reject [feedback]]"""
        assert self.agent is not None
        plan = self.agent.get_plan()

        if arg == "show" or not arg:
            if plan is None:
                self._print_system("No active plan. Start an analysis to create one.")
                return
            table = Table(title=f"Plan: {plan.title or 'Untitled'}")
            table.add_column("Step", style="cyan")
            table.add_column("Status")
            table.add_column("Description")
            for step in plan.steps:
                icon = {
                    "pending": "○", "running": "●", "done": "✓",
                    "failed": "✗", "skipped": "⏭", "awaiting_confirmation": "⏸",
                    "interrupted": "⚠",
                }.get(step.status, "?")
                table.add_row(step.step_id, icon, step.description or "")
            self.console.print(table)

        elif arg.startswith("approve"):
            self.agent.set_plan_approved()
            self._print_system(
                "Plan approved. Step-by-step confirmation mode enabled.\n"
                "The agent will ask for confirmation before each step."
            )

        elif arg.startswith("reject"):
            # Extract feedback after "reject"
            feedback = arg[len("reject"):].strip()
            if not feedback:
                self._print_system("Plan rejected. What would you like to change?")
                # Prompt for feedback inline
                feedback = await self._get_user_input("Feedback: ")
                if not feedback:
                    self._print_system("No feedback provided. Plan unchanged.")
                    return
            self._print_system(f"Plan rejected. Revising based on: \"{feedback}\"")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task("[dim]Revising plan...[/dim]", total=None)
                result = await self.agent.recycle_plan(feedback)
                progress.stop()
            if result.content:
                for block in result.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        self._print_agent(block["text"])
            self._print_system("Plan revised. Use /plan show to review, then /plan approve.")
        else:
            self._print_system("Usage: /plan [show|approve|reject <feedback>]")

    # ── Helpers ─────────────────────────────────────────────────────────

    def _rebuild_prompt(self) -> None:
        """Rebuild the agent's system prompt from current state."""
        if self.agent is None or not hasattr(self.agent, '_initialized'):
            return
        session = self.runtime.session.active_session
        domain = session.domain if session else "multi-omics"
        if self.agent._prompt_manager is not None:
            new_prompt = self.agent._rebuild_system_prompt(domain)
            self.runtime.context.set_system_prompt(new_prompt)

    def _print_status_bar(self) -> None:
        """Print a compact status bar at the bottom."""
        if self.agent is None:
            return
        mode = self.agent.mode
        skills = ", ".join(self.agent.active_skills) or "none"
        session = self.runtime.session.active_session
        sid = session.session_id[:12] if session else "N/A"

        bar = (
            f"[dim]Mode:[/dim] [bold]{mode}[/bold]  "
            f"[dim]Skills:[/dim] {skills}  "
            f"[dim]Session:[/dim] {sid}  "
            f"[dim]Ctrl+T: mode | Ctrl+S: save | Ctrl+L: clear | Ctrl+D: quit[/dim]"
        )
        self.console.print(bar, style=STATUS_STYLE)
        self._buffer_add("status_bar", bar)

    @staticmethod
    def _can_pair_tool_results(content: Any, tool_results: list[dict[str, Any]]) -> bool:
        """Check if tool_use blocks in content can be paired with buffered tool_results.

        Old wire sessions lack tool_use_id in TOOL_RESULT events, so pairing is
        impossible. In that case we fall back to stripping tool_use blocks.
        """
        if not isinstance(content, list):
            return True  # String content has no tool_use blocks
        tool_use_ids = {
            b["id"] for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        }
        if not tool_use_ids:
            return True  # No tool_use blocks — no problem
        result_ids = {tr.get("tool_use_id", "") for tr in tool_results}
        # Every tool_use must have a matching tool_result
        return tool_use_ids.issubset(result_ids)

    @staticmethod
    def _strip_tool_use_blocks(content: Any) -> Any:
        """Remove tool_use blocks from content, keeping only text blocks.

        During wire replay of old sessions, tool_use blocks can't be paired with
        tool_results, which causes Anthropic API errors. We preserve text only.
        Returns None if no text blocks remain (caller should skip adding).
        """
        if isinstance(content, list):
            text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
            return text_blocks if text_blocks else None
        return content

    @staticmethod
    def _extract_display_text(content: Any) -> str:
        """Extract readable text from wire content (string or LLM content blocks)."""
        if content is None:
            return ""
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return str(content)

    # ── Output Renderers ────────────────────────────────────────────────

    def _print_user(self, text: str) -> None:
        self.console.print(Panel(text, style=USER_STYLE, title="You", title_align="left"))
        self._buffer_add("user", text)

    def _print_agent(self, text: str) -> None:
        if not text:
            return
        self.console.print(Panel(
            Markdown(text),
            style=AGENT_STYLE,
            title="PassiAgent",
            title_align="left",
        ))
        self._buffer_add("agent", text)

    def _print_tool_call(self, name: str, params_or_str: str | dict | Any) -> None:
        if isinstance(params_or_str, dict):
            params_str = ", ".join(
                f"{k}={repr(v)[:80]}" for k, v in params_or_str.items()
            )
        else:
            params_str = str(params_or_str)[:200]
        self.console.print(f"🔧 {name}({params_str})", style=TOOL_STYLE)
        self._buffer_add("tool_call", name, params_str)

    def _print_tool_result(self, name: str, result: str) -> None:
        summary = str(result)[:200]
        self.console.print(f"  └─ {summary}", style=Style(color="#6B7280", dim=True))
        self._buffer_add("tool_result", summary)

    def _print_system(self, text: str | Markdown) -> None:
        is_md = isinstance(text, Markdown)
        if isinstance(text, str):
            self.console.print(text, style=SYSTEM_STYLE)
        else:
            self.console.print(text)
        self._buffer_add("system", text, is_md)

    def _print_error(self, text: str) -> None:
        self.console.print(f"✗ {text}", style=ERROR_STYLE)
        self._buffer_add("error", text)


# ── Entry Point ─────────────────────────────────────────────────────────

async def run_cli(config: PassiConfig | None = None) -> None:
    """Entry point for 'passi chat' command."""
    if config is None:
        from passi.config import load_config
        config = load_config()
    cli = PassiCLI(config)
    await cli.start()
