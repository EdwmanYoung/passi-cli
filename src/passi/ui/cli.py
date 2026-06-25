"""Rich TUI-based interactive CLI for PassiAgent.

Provides a chat-like terminal interface with:
- Rich formatting for messages (user/agent/tool/system)
- Live streaming of agent responses
- Slash commands (/help, /save, /load, /clear, /mode, /export)
- Multi-line input with Alt+Enter
- Progress indication for tool execution
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.style import Style
from rich.text import Text

from passi.config import PassiConfig
from passi.infra.runtime import Runtime
from passi.soul.passi_agent import PassiAgent

logger = logging.getLogger(__name__)

# Brand styles
USER_STYLE = Style(color="#3B82F6", bold=True)
AGENT_STYLE = Style(color="#F8FAFC")
TOOL_STYLE = Style(color="#10B981")
ERROR_STYLE = Style(color="#EF4444")
SYSTEM_STYLE = Style(color="#F59E0B")
HEADER_STYLE = Style(color="#2563EB", bold=True)

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════╗
║     🧬  PassiAgent  v0.1.0                        ║
║     Multi-Omics Bioinformatics Analysis Agent        ║
║                                                      ║
║  Type your analysis request or /help for commands    ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
**Available Commands:**

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/save <name>` | Save current session checkpoint |
| `/clear` | Clear conversation context |
| `/export` | Export session as chat log |
| `/domain <name>` | Switch analysis domain (transcriptomics, genomics, etc.) |
| `/methods <domain>` | List available methods for a domain |
| `/formats <domain>` | List supported data formats for a domain |
| `/mode chat` | Switch to interactive chat mode (default) |
| `/quit` or `/exit` | Exit PassiAgent |

**Tips:**
- Use `Alt+Enter` for multi-line input
- Press `Ctrl+C` to interrupt a running analysis
- Results are saved in `./output/` directory
"""


class PassiCLI:
    """Interactive CLI chat interface for PassiAgent."""

    def __init__(self, config: PassiConfig) -> None:
        self.config = config
        self.console = Console()
        self.runtime = Runtime(config)
        self.agent: PassiAgent | None = None
        self._domain: str = "multi-omics"
        self._running: bool = True

    async def start(self) -> None:
        """Initialize and start the interactive CLI."""
        self.console.clear()
        self.console.print(WELCOME_BANNER, style=HEADER_STYLE)

        # Initialize agent
        self.runtime.session.create_session(domain=self._domain)
        self.agent = PassiAgent(self.runtime)
        await self.agent.initialize()

        self._print_system(f"Session: {self.runtime.session.active_session.session_id}")
        self._print_system(f"Domain: {self._domain} | Provider: {self.config.default_provider}")

        # Main loop
        await self._repl()

    async def _repl(self) -> None:
        """Read-Eval-Print loop."""
        while self._running:
            try:
                user_input = await self._get_input()
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

    async def _get_input(self) -> str:
        """Get user input with prompt styling."""
        loop = asyncio.get_running_loop()
        prompt_text = Text("🧬 > ", style=USER_STYLE)
        return await loop.run_in_executor(
            None,
            lambda: Prompt.ask(prompt_text, console=self.console),
        )

    async def _process_message(self, message: str) -> None:
        """Process a user message through the agent."""
        assert self.agent is not None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Thinking...", total=None)

            try:
                response = await self.agent.chat(message)
            except Exception as e:
                progress.stop()
                self._print_error(f"Agent error: {e}")
                return

            progress.stop()

        # Check for pending question (ask_user tool was triggered)
        pq = response.metadata.get("pending_question") if response.metadata else None
        if pq:
            self._render_ask_user(pq)
            return

        # Render response
        if isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        self._print_agent(block["text"])
                    elif block.get("type") == "tool_use":
                        self._print_tool_call(block.get("name", ""), block.get("input", {}))
        elif isinstance(response.content, str):
            self._print_agent(response.content)

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._print_system(Markdown(HELP_TEXT))
        elif cmd == "/clear":
            if self.agent:
                await self.agent.reset()
            self._print_system("Context cleared.")
        elif cmd == "/save":
            name = arg or f"checkpoint_{datetime.now():%Y%m%d_%H%M%S}"
            state = {"domain": self._domain, "timestamp": datetime.now().isoformat()}
            self.runtime.session.checkpoint(state)
            self._print_system(f"Session checkpoint saved: {name}")
        elif cmd == "/export":
            from passi.wire.persistence import WirePersistence

            session = self.runtime.session.active_session
            if session:
                wire_path = self.runtime.session.get_session_dir() / "wire.jsonl"
                persistence = WirePersistence(wire_path)
                chatlog = persistence.export_chatlog(session.session_id)
                export_dir = self.config.output_dir
                export_dir.mkdir(parents=True, exist_ok=True)
                report_path = export_dir / f"chatlog_{session.session_id}.md"
                report_path.write_text(chatlog, encoding="utf-8")
                self._print_system(f"Chat log exported: {report_path}")
        elif cmd == "/domain":
            if arg:
                self._domain = arg
                self._print_system(f"Domain set to: {self._domain}")
            else:
                self._print_system(f"Current domain: {self._domain}")
        elif cmd == "/methods":
            from passi.knowledge.methods import get_methods_by_domain

            domain = arg or self._domain
            methods = get_methods_by_domain(domain)
            if methods:
                text = "\n".join(f"- **{mid}**: {m['name']} ({m['backend']})" for mid, m in methods.items())
                self._print_system(Markdown(f"**Methods for {domain}:**\n{text}"))
            else:
                self._print_system(f"No methods found for domain: {domain}")
        elif cmd == "/formats":
            from passi.knowledge.formats import get_formats_by_domain

            domain = arg or self._domain
            formats = get_formats_by_domain(domain)
            if formats:
                text = "\n".join(f"- **{f['format']}** ({', '.join(f['suffixes'])}): {f['description']}" for f in formats)
                self._print_system(Markdown(f"**Formats for {domain}:**\n{text}"))
            else:
                self._print_system(f"No formats found for domain: {domain}")
        elif cmd in ("/quit", "/exit"):
            self._running = False
            self._print_system("Goodbye!")
        else:
            self._print_system(f"Unknown command: {cmd}. Type /help for available commands.")

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        if self.agent:
            await self.agent.shutdown()
        self.console.print("\n[dim]Session ended.[/dim]")

    # ── Output helpers ──

    def _render_ask_user(self, pq: dict) -> None:
        """Render a pending question from the ask_user tool."""
        question_text = pq.get("question", "")
        context = pq.get("context", "")
        options = pq.get("options") or []

        # Build the panel body
        body = question_text
        if context:
            body += f"\n\n[dim]{context}[/dim]"

        self.console.print(Panel(
            Markdown(body),
            style=SYSTEM_STYLE,
            title="Agent needs your input",
            title_align="left",
        ))

        if options:
            self.console.print("\n[bold]Options:[/bold]")
            for i, opt in enumerate(options, 1):
                self.console.print(f"  {i}. {opt}")
            self.console.print()

    def _print_user(self, text: str) -> None:
        self.console.print(Panel(text, style=USER_STYLE, title="You", title_align="left"))

    def _print_agent(self, text: str) -> None:
        self.console.print(Panel(
            Markdown(text) if text else "",
            style=AGENT_STYLE,
            title="PassiAgent",
            title_align="left",
        ))

    def _print_tool_call(self, name: str, params: dict[str, Any]) -> None:
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        self.console.print(f"[{TOOL_STYLE.style}]🔧 {name}({params_str})[/]")

    def _print_system(self, text: str | Markdown) -> None:
        if isinstance(text, str):
            self.console.print(f"[{SYSTEM_STYLE.style}]{text}[/]")
        else:
            self.console.print(text)

    def _print_error(self, text: str) -> None:
        self.console.print(f"[{ERROR_STYLE.style}]✗ {text}[/]")
