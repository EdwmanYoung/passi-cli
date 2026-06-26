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
import logging
from datetime import datetime
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
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

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize and start the interactive CLI."""
        self.console.clear()
        self.console.print(WELCOME_BANNER, style=HEADER_STYLE)

        # Initialize agent
        self.runtime.session.create_session(domain=self._domain)
        self.agent = PassiAgent(self.runtime)
        await self.agent.initialize()

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

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        if self.agent:
            await self.agent.shutdown()
        self.console.print("\n[dim]Session ended.[/dim]")

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
                    await self._handle_command(user_input)
                    continue

                # Chat message
                self._print_user(user_input)
                await self._process_message(user_input)

            except KeyboardInterrupt:
                self._print_system("Interrupted. Type /quit to exit.")
            except EOFError:
                break

        await self._shutdown()

    def _create_input_session(self) -> PromptSession[str]:
        """Create a prompt_toolkit PromptSession with key bindings.

        Follows kimi-cli pattern: all key bindings are registered on a single
        KeyBindings object, no background threads, no stdin conflict.
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
                skills = ",".join(self.agent.active_skills)
                return HTML(f"<rprompt>skills:{skills}</rprompt>")
            return None

        return PromptSession(
            key_bindings=kb,
            style=_PROMPT_STYLE,
            message=_message,
            rprompt=_rprompt,
            complete_while_typing=False,
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
            session = self._create_input_session()
            result = await session.prompt_async()
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
        q_text = question.get("question", "")
        q_context = question.get("context", "")
        q_options = question.get("options") or []

        # Display the question
        if q_context:
            self._print_system(f"[dim]{q_context}[/dim]")
        self.console.print(
            Panel(q_text, style=SYSTEM_STYLE, title="PassiAgent asks", title_align="left")
        )
        if q_options:
            self._print_system(f"Options: {', '.join(q_options)}")

        # Get answer from user
        answer = await self._get_user_input("Your answer: ")
        if not answer.strip():
            answer = "(no answer)"

        self._print_user(f"[response] {answer}")

        # Route answer back to agent for further processing
        await self._process_message(answer)

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

    # ── Output Renderers ────────────────────────────────────────────────

    def _print_user(self, text: str) -> None:
        self.console.print(Panel(text, style=USER_STYLE, title="You", title_align="left"))

    def _print_agent(self, text: str) -> None:
        if not text:
            return
        self.console.print(Panel(
            Markdown(text),
            style=AGENT_STYLE,
            title="PassiAgent",
            title_align="left",
        ))

    def _print_tool_call(self, name: str, params_or_str: str | dict | Any) -> None:
        if isinstance(params_or_str, dict):
            params_str = ", ".join(
                f"{k}={repr(v)[:80]}" for k, v in params_or_str.items()
            )
        else:
            params_str = str(params_or_str)[:200]
        self.console.print(f"🔧 {name}({params_str})", style=TOOL_STYLE)

    def _print_system(self, text: str | Markdown) -> None:
        if isinstance(text, str):
            self.console.print(text, style=SYSTEM_STYLE)
        else:
            self.console.print(text)

    def _print_error(self, text: str) -> None:
        self.console.print(f"✗ {text}", style=ERROR_STYLE)


# ── Entry Point ─────────────────────────────────────────────────────────

async def run_cli(config: PassiConfig | None = None) -> None:
    """Entry point for 'passi chat' command."""
    if config is None:
        from passi.config import load_config
        config = load_config()
    cli = PassiCLI(config)
    await cli.start()
