"""PassiAgent — the main bioinformatics analysis agent.

Implements the Soul protocol with ReAct loop, tool orchestration, and
domain-specific sub-agent delegation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from passi.config import PassiConfig
from passi.infra.context import ContextManager
from passi.infra.hooks import HookManager
from passi.infra.llm_client import LLMClient
from passi.infra.plan import PlanManager, StepStatus
from passi.infra.provenance import ProvenanceTracker
from passi.infra.task_tracker import Task, TaskTracker
from passi.infra.runtime import Runtime
from passi.infra.session import SessionManager
from passi.prompts import PromptManager
from passi.soul.protocol import AgentMessage, AgentStreamEvent, Soul
from passi.tools.registry import ToolRegistry
from passi.wire.protocol import EventType, Wire, WireEvent

logger = logging.getLogger(__name__)


@dataclass
class _ReactEvent:
    """Internal event yielded during ReAct loop execution for real-time streaming."""

    type: str  # "thinking", "text", "tool_call", "tool_result", "error", "done", "pending_question"
    content: str = ""
    tool_name: str = ""
    metadata: dict = field(default_factory=dict)


class PassiAgent(Soul):
    """Main bioinformatics analysis agent.

    Orchestrates tool execution through a ReAct loop with LLM reasoning.
    Delegates complex domain-specific analysis to sub-agents.
    """

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.config: PassiConfig = runtime.config
        self.wire: Wire = Wire()

        # Lazy initialization
        self._llm_client: LLMClient | None = None
        self._tool_registry: ToolRegistry | None = None
        self._provenance: ProvenanceTracker | None = None
        self._plan_manager: PlanManager | None = None
        self._task_tracker: TaskTracker | None = None
        self._prompt_manager: PromptManager | None = None
        self._initialized: bool = False
        self._afk_mode: bool = False
        self._mode: str = "chat"  # "chat", "plan", or "afk"
        self._plan_first: bool = False  # plan-first mode (requires plan before execution)
        self._hook_manager: HookManager | None = None

        # Plan interaction state
        self._plan_qa_active: bool = False  # pre-plan Q&A phase in progress
        self._plan_approved: bool = False  # plan has been explicitly approved
        self._plan_recycle_count: int = 0  # number of plan reject/recycle iterations
        self._step_confirm_mode: bool = False  # step-by-step confirmation active
        self._auto_all: bool = False  # user chose auto-execute all remaining steps

        # Interrupt support
        self._interrupt_event: asyncio.Event = asyncio.Event()
        self._agent_busy: bool = False  # true when agent is processing a message

    async def initialize(self) -> None:
        """Initialize the agent, connecting all services."""
        if self._initialized:
            return

        self._llm_client = self.runtime.get_llm_client()
        self._provenance = ProvenanceTracker(self.config.result_dir)

        # ── Plan manager & Task tracker (must be before _create_tool_registry) ──
        session = self.runtime.session.active_session
        session_dir = (
            self.config.session.sessions_dir / session.session_id
            if session
            else self.config.session.sessions_dir
        )
        self._plan_manager = PlanManager(session_dir)
        self._plan_manager.load_plan()  # Load existing plan if present

        self._task_tracker = TaskTracker(session_dir)
        self._task_tracker.load_tasks()  # Load existing tasks if present

        # Wire events should persist inside the session directory for replay
        self.wire._wire_path = session_dir / "wire.jsonl"

        self._tool_registry = self._create_tool_registry()

        # ── Initialize R environment (opt-in via rpy2_enabled) ──
        exec_cfg = self.config.execution
        if exec_cfg.rpy2_enabled:
            from passi.executors.r_executor import init_rpy2

            r_status = init_rpy2(exec_cfg.r_home, exec_cfg.r_lib_path)
            if r_status["ready"]:
                logger.info(
                    "R environment ready: %s | libs: %s",
                    r_status.get("r_version"),
                    r_status.get("lib_paths"),
                )
            else:
                logger.warning("R environment not available: %s", r_status.get("error"))
                logger.info("R scripts will use Rscript subprocess fallback")
        else:
            logger.info("rpy2 disabled — R tools will use Rscript subprocess if R is on PATH")

        # ── Hook manager ──
        hooks_cfg = self.config.hooks
        hooks_path = hooks_cfg.hooks_file
        self._hook_manager = HookManager(hooks_path, wire=self.wire)
        if hooks_cfg.enabled:
            self.wire.subscribe(self._hook_manager)

        # ── Set up context with templated system prompt ──
        template_dir = self.config.prompt_template_dir or None
        self._prompt_manager = PromptManager(template_dir)
        session = self.runtime.session.active_session
        domain = session.domain if session else "multi-omics"
        result_id = session.result_id if session else ""
        self._afk_mode = getattr(self.config, "afk_mode", False)

        # Derive mode from config or explicit set_mode() call
        if self._afk_mode:
            self._mode = "afk"

        system_prompt = self._rebuild_system_prompt(domain, result_id)
        self.runtime.context.set_system_prompt(system_prompt)
        self.runtime.context.set_tools(
            self._tool_registry.get_schemas(format="anthropic")
        )

        # Wire LLM client to context manager for LLM-based compaction
        self.runtime.context.set_llm_client(self._llm_client)

        # Emit session start
        session = self.runtime.session.active_session
        sid = session.session_id if session else ""
        self.wire.emit(EventType.SESSION_START, session_id=sid)

        # Set hook manager session context
        if self._hook_manager is not None:
            self._hook_manager.set_session_context(sid, domain)

        await self.runtime.initialize()
        self._initialized = True
        logger.info("PassiAgent initialized. Mode: %s", self._mode)

    def set_mode(
        self,
        mode: str = "chat",
        plan_first: bool = False,
        skills: list[str] | None = None,
    ) -> None:
        """Switch agent mode and optionally activate skills.

        Args:
            mode: "chat" (interactive), "plan" (plan-first), or "afk" (autonomous)
            plan_first: In chat/plan mode, require plan creation before execution
            skills: List of skill names to activate (e.g., ["metabolomics", "stats"])
        """
        valid_modes = {"chat", "plan", "afk"}
        if mode not in valid_modes:
            logger.warning("Invalid mode '%s'. Choose from: %s", mode, valid_modes)
            return

        prev_mode = self._mode
        self._mode = mode
        self._afk_mode = (mode == "afk")
        self._plan_first = plan_first or (mode == "plan")

        # Apply skills
        if skills and self._prompt_manager is not None:
            self._prompt_manager.clear_skills()
            for skill in skills:
                self._prompt_manager.load_skill(skill)

        # Rebuild system prompt if already initialized
        if self._initialized and self._prompt_manager is not None:
            session = self.runtime.session.active_session
            domain = session.domain if session else "multi-omics"
            new_prompt = self._rebuild_system_prompt(domain)
            self.runtime.context.set_system_prompt(new_prompt)

        logger.info(
            "Agent mode switched: %s → %s (afk=%s, plan_first=%s, skills=%s)",
            prev_mode, self._mode, self._afk_mode, self._plan_first,
            self._prompt_manager.active_skills if self._prompt_manager else [],
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def plan_first(self) -> bool:
        return self._plan_first

    @property
    def active_skills(self) -> list[str]:
        if self._prompt_manager is not None:
            return self._prompt_manager.active_skills
        return []

    def get_hook_manager(self) -> HookManager | None:
        return self._hook_manager

    def interrupt(self) -> None:
        """Signal the agent to gracefully interrupt the current tool execution.

        Sets the interrupt event that tools check during execution.
        Does NOT kill the process — tools check the flag and return early.
        """
        self._interrupt_event.set()
        logger.info("Interrupt signal sent to agent")

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag before starting a new tool execution."""
        self._interrupt_event.clear()

    def set_plan_approved(self) -> None:
        """Mark the plan as approved and enable step confirmation mode."""
        self._plan_approved = True
        self._step_confirm_mode = True
        self._auto_all = False

        # Emit plan approved event
        session = self.runtime.session.active_session
        sid = session.session_id if session else ""
        plan = self.get_plan()
        self.wire.emit(
            EventType.PLAN_APPROVED,
            {"plan_id": plan.plan_id if plan else "", "title": plan.title if plan else ""},
            sid,
        )

        # Rebuild system prompt to include step confirmation protocol
        session = self.runtime.session.active_session
        domain = session.domain if session else "multi-omics"
        result_id = session.result_id if session else ""
        if self._initialized and self._prompt_manager is not None:
            new_prompt = self._rebuild_system_prompt(domain, result_id)
            self.runtime.context.set_system_prompt(new_prompt)

        logger.info("Plan approved, step confirmation mode enabled")

    async def recycle_plan(self, feedback: str) -> AgentMessage:
        """Handle plan rejection by feeding feedback back to the agent.

        The agent will call create_plan again with the user's feedback incorporated.
        """
        if not self._initialized:
            await self.initialize()

        self._plan_recycle_count += 1

        # Emit plan recycled event
        session = self.runtime.session.active_session
        sid = session.session_id if session else ""
        plan = self.get_plan()
        self.wire.emit(
            EventType.PLAN_RECYCLED,
            {
                "version": self._plan_recycle_count,
                "feedback": feedback,
                "previous_plan_id": plan.plan_id if plan else "",
            },
            sid,
        )

        if self._plan_recycle_count > self.config.plan.max_recycles:
            return AgentMessage(
                role="agent",
                content=[
                    {"type": "text", "text": (
                        f"Plan has been rejected {self._plan_recycle_count} times "
                        f"(max: {self.config.plan.max_recycles}). "
                        "Consider switching to chat mode for more flexible interaction. "
                        "Use /mode chat to switch."
                    )}
                ],
            )

        # Build recycle directive
        recycle_msg = (
            f"[Plan Rejection - Revision {self._plan_recycle_count}]\n"
            f"The user rejected your plan with this feedback:\n\n"
            f"\"{feedback}\"\n\n"
            f"Please revise the plan based on this feedback. Revise ONLY the parts "
            f"they mentioned. Do NOT change other aspects of the plan unless the "
            f"feedback implies a broader change. Call create_plan again."
        )

        self.runtime.context.add_message("user", recycle_msg)

        # Run one ReAct iteration to let agent call create_plan
        assert self._llm_client is not None
        context = self.runtime.context.get_full_context()
        response = await self._llm_client.chat(
            messages=context["messages"],
            tools=context.get("tools"),
            system=context["system"],
            max_tokens=self.config.get_llm_config().tool_call_max_tokens,
        )

        text_parts = [b.get("text", "") for b in response["content"] if b.get("type") == "text"]
        agent_text = " ".join(text_parts)

        if agent_text:
            self.runtime.context.add_message("assistant", agent_text)

        # Handle tool calls from response (agent may call create_plan)
        tool_calls = response.get("tool_calls") or []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_input = tc.get("input", {})
            assert self._tool_registry is not None
            result = await self._tool_registry.execute(tool_name, tool_input)

        return AgentMessage(
            role="agent",
            content=[{"type": "text", "text": agent_text or "Plan revised."}],
        )

    @property
    def plan_approved(self) -> bool:
        return self._plan_approved

    @property
    def agent_busy(self) -> bool:
        return self._agent_busy

    def _rebuild_system_prompt(self, domain: str, result_id: str = "") -> str:
        """Rebuild system prompt from current mode/skills state."""
        if self._prompt_manager is None:
            return ""
        return self._prompt_manager.build_system_prompt(
            domain=domain,
            plan_enabled=True,
            data_check_enabled=self.config.enable_data_format_check,
            afk_mode=self._afk_mode,
            plan_first=self._plan_first,
            plan_qa=self._plan_qa_active,
            step_confirm=self._step_confirm_mode,
            result_id=result_id,
        )

    async def chat(
        self,
        user_message: str,
        attachments: list[str] | None = None,
    ) -> AgentMessage:
        """Send a message and get a complete response via ReAct loop."""
        if not self._initialized:
            await self.initialize()

        assert self._llm_client is not None
        assert self._tool_registry is not None
        assert self._provenance is not None

        session = self.runtime.session.active_session
        sid = session.session_id if session else ""

        # Emit user message
        self.wire.emit(EventType.USER_MESSAGE, {"content": user_message}, sid)
        self.runtime.context.add_message("user", user_message)

        self._agent_busy = True
        try:
            final_result: AgentMessage | None = None
            async for event in self._run_react_loop_stream(user_message, attachments):
                if event.type == "done":
                    final_result = event.metadata.get("agent_message")
            if final_result is None:
                return AgentMessage(role="agent", content=[{"type": "text", "text": ""}])
            return final_result
        finally:
            self._agent_busy = False
            self.runtime.session.touch()

    async def _run_react_loop_stream(
        self,
        user_message: str,
        attachments: list[str] | None = None,
    ) -> AsyncIterator[_ReactEvent]:
        """Run the ReAct loop, yielding events at each step for real-time streaming.

        The caller (chat() or chat_stream()) handles user message setup and
        the _agent_busy flag. This generator only manages the ReAct loop itself.
        """
        session = self.runtime.session.active_session
        sid = session.session_id if session else ""

        max_iterations = 20
        final_content: list[dict] = []
        pending_question: dict[str, Any] | None = None

        for iteration in range(max_iterations):
            if self._interrupt_event.is_set():
                break

            yield _ReactEvent(type="thinking", content="Analyzing...")

            context = self.runtime.context.get_full_context()
            response = await self._llm_client.chat(
                messages=context["messages"],
                tools=context.get("tools"),
                system=context["system"],
                max_tokens=self.config.get_llm_config().tool_call_max_tokens,
            )

            # Track token usage
            usage = response.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            if input_tokens > 0:
                self.runtime.context.update_api_tokens(input_tokens)

            # Handle text response
            text_parts = [b.get("text", "") for b in response["content"] if b.get("type") == "text"]
            agent_text = " ".join(text_parts)
            if agent_text:
                final_content.append({"type": "text", "text": agent_text})
                yield _ReactEvent(type="text", content=agent_text)

            # Handle tool calls
            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                if agent_text:
                    self.runtime.context.add_message("assistant", agent_text)
                break

            # Store assistant message with tool_use blocks
            assistant_blocks: list[dict[str, Any]] = []
            for b in response["content"]:
                if b.get("type") == "text" and b.get("text"):
                    assistant_blocks.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "tool_use":
                    assistant_blocks.append({
                        "type": "tool_use",
                        "id": b.get("id", ""),
                        "name": b.get("name", ""),
                        "input": b.get("input", {}),
                    })
            self.runtime.context.add_message("assistant", assistant_blocks)

            # Execute tools and collect results
            tool_results: list[dict[str, Any]] = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc.get("input", {})

                # ── Task tracking ──
                plan_step_id = ""
                if self._plan_manager is not None:
                    current_step = self._plan_manager.get_current_step()
                    if current_step is not None:
                        plan_step_id = current_step.step_id

                task = None
                if self._task_tracker is not None:
                    task = self._task_tracker.create_task(
                        tool_name, tool_input, step_id=plan_step_id
                    )

                # Emit tool call (include tool_use id for wire replay pairing)
                self.wire.emit(
                    EventType.TOOL_CALL,
                    {"name": tool_name, "params": tool_input, "id": tc.get("id", "")},
                    sid,
                )

                yield _ReactEvent(
                    type="tool_call",
                    content=json.dumps(tool_input, ensure_ascii=False, default=str),
                    tool_name=tool_name,
                )

                # Clear interrupt flag before execution
                self.clear_interrupt()

                # Execute tool
                start = time.perf_counter()
                try:
                    try:
                        result = await self._tool_registry.execute(
                            tool_name, tool_input, interrupt_event=self._interrupt_event
                        )
                    except TypeError:
                        result = await self._tool_registry.execute(tool_name, tool_input)
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    yield _ReactEvent(
                        type="error",
                        content=str(exc),
                        tool_name=tool_name,
                    )
                    result = {"success": False, "error": str(exc)}
                else:
                    elapsed_ms = (time.perf_counter() - start) * 1000

                # Check if interrupted
                if result.get("interrupted"):
                    self.wire.emit(
                        EventType.TOOL_INTERRUPTED,
                        {"name": tool_name, "run_dir": result.get("run_dir", "")},
                        sid,
                    )
                    if self._plan_manager is not None:
                        current = self._plan_manager.get_current_step()
                        if current is not None:
                            self._plan_manager.update_step_status(
                                step_id=current.step_id,
                                status=StepStatus.INTERRUPTED,
                                error_message="Interrupted by user",
                            )
                    pending_question = {
                        "question": (
                            f"Tool '{tool_name}' was interrupted. "
                            "What would you like to do?"
                        ),
                        "context": f"Step was interrupted after {elapsed_ms:.0f}ms",
                        "options": [
                            "Resume from interrupted step",
                            "Skip this step",
                            "Modify plan",
                            "Abort analysis",
                        ],
                    }
                    yield _ReactEvent(
                        type="pending_question",
                        content=pending_question["question"],
                        metadata={
                            "context": pending_question["context"],
                            "options": pending_question["options"],
                        },
                    )
                    break

                # Record provenance
                input_files: list[str] = []
                output_files: list[str] = []
                if tool_name in ("run_python", "run_r"):
                    input_files = tool_input.get("input_files", [])
                    run_dir = result.get("run_dir", "")
                    if run_dir:
                        rd = Path(run_dir)
                        if rd.exists():
                            known = {"script.py", "script.R", "stdout.log", "stderr.log", "run_metadata.json"}
                            output_files = sorted(
                                str(p) for p in rd.iterdir()
                                if p.is_file() and p.name not in known
                            )

                record = self._provenance.record_step(
                    tool_name=tool_name,
                    tool_params=tool_input,
                    input_files=input_files,
                    output_files=output_files,
                    exit_code=0 if result.get("success") else 1,
                    error_message=result.get("error", ""),
                    duration_ms=elapsed_ms,
                    session_id=sid,
                )

                # ── Task tracking: complete task ──
                if self._task_tracker is not None and task is not None:
                    self._task_tracker.complete_task(
                        task.task_id,
                        success=result.get("success", False),
                        result_summary=str(result.get("result", ""))[:200],
                        error=result.get("error", ""),
                        provenance_step_id=record.step_id,
                    )

                # Emit tool result (include tool_use_id for wire replay pairing)
                self.wire.emit(
                    EventType.TOOL_RESULT,
                    {"name": tool_name, "result": result, "tool_use_id": tc.get("id", "")},
                    sid,
                )

                result_summary = str(result.get("result", ""))[:200] or str(result.get("stdout", ""))[:200]
                yield _ReactEvent(
                    type="tool_result",
                    content=result_summary,
                    tool_name=tool_name,
                    metadata={"success": result.get("success", False)},
                )

                # Notify hook manager on tool errors
                if not result.get("success") and self._hook_manager is not None:
                    error_msg = result.get("error", "") or result.get("stderr", "")
                    self._hook_manager.notify_error(tool_name, str(error_msg)[:500])

                # Emit plan-related events
                self._emit_plan_event(tool_name, tool_input, result, sid)

                # Collect tool result — batched into one message later
                result_text = json.dumps(result, ensure_ascii=False, default=str)
                final_content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tool_name,
                    "input": tool_input,
                })
                tool_results.append({
                    "tool_use_id": tc.get("id", ""),
                    "content": result_text,
                })

                # Check if this tool wants to pause for user input
                if result.get("__ask_user__"):
                    if self._afk_mode:
                        logger.info(
                            "AFK mode: agent called ask_user. Question: %s",
                            str(result.get("question", ""))[:200],
                        )
                        if tool_results:
                            tool_results[-1]["content"] = json.dumps(
                                {
                                    "success": True,
                                    "message": (
                                        "AFK autonomous mode active. Your ask_user call has been intercepted. "
                                        "Please make your best-guess decision based on observed data patterns "
                                        "and bioinformatics best practices. Proceed with the analysis immediately. "
                                        "Do NOT call ask_user again."
                                    ),
                                    "auto_decision": True,
                                },
                                ensure_ascii=False,
                                default=str,
                            )
                    else:
                        pending_question = {
                            "question": result["question"],
                            "context": result.get("context", ""),
                            "options": result.get("options"),
                        }
                        # Remove ask_user's tool_use block from final_content —
                        # its "result" is a question, not computational output
                        if final_content and final_content[-1].get("type") == "tool_use":
                            final_content.pop()
                        yield _ReactEvent(
                            type="pending_question",
                            content=pending_question["question"],
                            metadata={
                                "context": pending_question.get("context", ""),
                                "options": pending_question.get("options"),
                            },
                        )
                        break

            # Batch all tool results from this iteration into one message
            if tool_results:
                self.runtime.context.add_message("tool_results", tool_results)

            # Break ReAct loop if ask_user was triggered
            if pending_question is not None:
                break

            # Check for context compaction
            if self.runtime.context.needs_compaction():
                await self.runtime.context.compact()

        # Build final message
        content = final_content if len(final_content) > 1 else final_content[0] if final_content else {"type": "text", "text": ""}
        metadata: dict[str, Any] = {}
        if pending_question is not None:
            metadata["pending_question"] = pending_question
        agent_msg = AgentMessage(
            role="agent",
            content=content if isinstance(content, list) else [content],
            metadata=metadata,
        )

        # Emit agent message
        self.wire.emit(
            EventType.AGENT_MESSAGE,
            {"content": agent_msg.content},
            sid,
        )

        yield _ReactEvent(
            type="done",
            content="",
            metadata={"agent_message": agent_msg},
        )

    async def chat_stream(
        self,
        user_message: str,
        attachments: list[str] | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Stream agent response with incremental events in real-time."""
        if not self._initialized:
            await self.initialize()

        assert self._llm_client is not None
        assert self._tool_registry is not None
        assert self._provenance is not None

        session = self.runtime.session.active_session
        sid = session.session_id if session else ""

        # Emit user message
        self.wire.emit(EventType.USER_MESSAGE, {"content": user_message}, sid)
        self.runtime.context.add_message("user", user_message)

        self._agent_busy = True
        try:
            yield AgentStreamEvent(type="thinking", content="Analyzing request...")

            async for event in self._run_react_loop_stream(user_message, attachments):
                yield AgentStreamEvent(
                    type=event.type,
                    content=event.content,
                    tool_name=event.tool_name,
                    metadata=event.metadata,
                )
        finally:
            self._agent_busy = False
            self.runtime.session.touch()

    async def execute_tool(self, tool_name: str, params: dict) -> AgentMessage:
        """Execute a tool directly."""
        if not self._initialized:
            await self.initialize()

        assert self._tool_registry is not None
        result = await self._tool_registry.execute(tool_name, params)
        return AgentMessage(
            role="tool",
            content=json.dumps(result, ensure_ascii=False, default=str),
            name=tool_name,
        )

    async def reset(self) -> None:
        """Reset conversation context."""
        self.runtime.context.clear()
        logger.info("Agent context reset.")

    async def shutdown(self) -> None:
        """Clean shutdown."""
        session = self.runtime.session.active_session
        if session:
            self.wire.emit(EventType.SESSION_END, session_id=session.session_id)
        await self.runtime.shutdown()

    def get_plan(self):
        """Get the current analysis plan, if any."""
        from passi.infra.plan import AnalysisPlan
        if self._plan_manager is None:
            return None
        return self._plan_manager.get_plan()

    def get_tasks(self) -> list[Task]:
        """Get all recorded tasks from the current session."""
        if self._task_tracker is None:
            return []
        return self._task_tracker.get_tasks()

    def get_task(self, task_id: str) -> Task | None:
        """Get a specific task by ID."""
        if self._task_tracker is None:
            return None
        return self._task_tracker.get_task(task_id)

    def _create_tool_registry(self) -> ToolRegistry:
        """Create and populate the tool registry with all available tools."""
        registry = ToolRegistry()

        # System / plan tools
        from passi.tools.ask_user_tool import AskUserTool
        from passi.tools.system_tools import (
            CreatePlanTool,
            GetPlanTool,
            UpdatePlanStatusTool,
        )

        registry.register(CreatePlanTool(self._plan_manager), category="system")
        registry.register(UpdatePlanStatusTool(self._plan_manager), category="system")
        registry.register(GetPlanTool(self._plan_manager), category="system")
        registry.register(AskUserTool(), category="system")

        # I/O tools
        from passi.tools.io_tools import (
            ExportResultsTool,
            ParseOmicsDataTool,
            ReadFileTool,
            WriteFileTool,
        )

        registry.register(ReadFileTool(), category="io")
        registry.register(WriteFileTool(), category="io")
        registry.register(ParseOmicsDataTool(), category="io")
        registry.register(ExportResultsTool(), category="io")

        exec_cfg = self.config.execution

        def _get_session_id() -> str:
            s = self.runtime.session.active_session
            return s.session_id if s else "default"

        def _get_result_id() -> str:
            s = self.runtime.session.active_session
            return s.result_id if s else "result_default"

        def _get_step_name() -> str:
            """Get the current plan step name for result directory structure."""
            if self._plan_manager is not None:
                step = self._plan_manager.get_current_step()
                if step is not None:
                    return step.step_id
            return ""

        result_id = _get_result_id()
        runs_base = self.config.result_dir / result_id

        # Execution tools
        from passi.tools.exec_tools import RunPythonTool, RunRTool

        run_python = RunPythonTool(
            runs_base=runs_base,
            session_id_provider=_get_session_id,
            step_name_provider=_get_step_name,
        )
        run_python.python_path = exec_cfg.python_path or "python"
        registry.register(run_python, category="exec")

        run_r = RunRTool(
            runs_base=runs_base,
            session_id_provider=_get_session_id,
            step_name_provider=_get_step_name,
        )
        run_r.r_home = exec_cfg.r_home or ""
        run_r.r_lib_path = exec_cfg.r_lib_path or ""
        run_r.r_path = exec_cfg.rscript_binary
        registry.register(run_r, category="exec")

        # QC tools
        from passi.tools.qc_tools import QcReportTool

        registry.register(QcReportTool(), category="qc")

        # Genomics tools
        from passi.tools.genomics_tools import (
            GwasAnalysisTool,
            ManhattanPlotTool,
            VcfStatsTool,
        )

        registry.register(VcfStatsTool(), category="genomics")
        registry.register(GwasAnalysisTool(), category="genomics")
        registry.register(ManhattanPlotTool(), category="genomics")

        # Epigenetics tools
        from passi.tools.epigenetics_tools import MethylationAnalysisTool, PeakQcTool

        registry.register(PeakQcTool(), category="epigenetics")
        registry.register(MethylationAnalysisTool(), category="epigenetics")

        # Transcriptomics tools
        from passi.tools.transcriptomics_tools import DifferentialAnalysisTool

        de_tool = DifferentialAnalysisTool(
            r_home=exec_cfg.r_home or "",
            r_lib_path=exec_cfg.r_lib_path or "",
            r_path=exec_cfg.rscript_binary,
        )
        registry.register(de_tool, category="transcriptomics")

        # Enrichment analysis tools
        from passi.tools.enrichment_tools import EnrichmentTool

        enrich_tool = EnrichmentTool(
            r_home=exec_cfg.r_home or "",
            r_lib_path=exec_cfg.r_lib_path or "",
            r_path=exec_cfg.rscript_binary,
        )
        registry.register(enrich_tool, category="transcriptomics")

        # Clinical / biostatistics tools
        from passi.tools.clinical_tools import SurvivalAnalysisTool

        surv_tool = SurvivalAnalysisTool(
            r_home=exec_cfg.r_home or "",
            r_lib_path=exec_cfg.r_lib_path or "",
            r_path=exec_cfg.rscript_binary,
        )
        registry.register(surv_tool, category="clinical")

        return registry

    def _emit_plan_event(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: dict[str, Any],
        session_id: str,
    ) -> None:
        """Emit plan-related wire events based on tool execution results."""
        if tool_name == "create_plan" and result.get("success"):
            self.wire.emit(
                EventType.PLAN_CREATED,
                {
                    "plan_id": result.get("plan_id"),
                    "title": result.get("title"),
                    "steps_count": result.get("steps_count"),
                },
                session_id,
            )
        elif tool_name == "update_plan_status" and result.get("success"):
            status = tool_input.get("status", "")
            step_id = tool_input.get("step_id", "")
            if status == "running":
                self.wire.emit(
                    EventType.PLAN_STEP_START,
                    {"step_id": step_id},
                    session_id,
                )
            elif status == "done":
                self.wire.emit(
                    EventType.PLAN_STEP_COMPLETE,
                    {"step_id": step_id},
                    session_id,
                )
            elif status == "failed":
                self.wire.emit(
                    EventType.PLAN_STEP_FAILED,
                    {
                        "step_id": step_id,
                        "error": tool_input.get("error_message", ""),
                    },
                    session_id,
                )

        # Auto-sync code execution results with plan step status
        if tool_name in ("run_python", "run_r") and self._plan_manager is not None:
            current = self._plan_manager.get_current_step()
            if current is not None:
                if result.get("success"):
                    self._plan_manager.update_step_status(
                        step_id=current.step_id,
                        status=StepStatus.DONE,
                        output_summary=str(result.get("stdout", ""))[:200] or str(result.get("run_dir", "")),
                    )
                    self.wire.emit(
                        EventType.PLAN_STEP_COMPLETE,
                        {"step_id": current.step_id},
                        session_id,
                    )
                else:
                    error_msg = result.get("error") or result.get("stderr", "")
                    self._plan_manager.update_step_status(
                        step_id=current.step_id,
                        status=StepStatus.FAILED,
                        error_message=str(error_msg)[:500],
                    )
                    self.wire.emit(
                        EventType.PLAN_STEP_FAILED,
                        {
                            "step_id": current.step_id,
                            "error": str(error_msg)[:200],
                        },
                        session_id,
                    )
