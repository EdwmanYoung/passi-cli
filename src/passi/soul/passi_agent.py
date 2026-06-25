"""PassiAgent — the main bioinformatics analysis agent.

Implements the Soul protocol with ReAct loop, tool orchestration, and
domain-specific sub-agent delegation.
"""

from __future__ import annotations

import json
import logging
import time
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

    async def initialize(self) -> None:
        """Initialize the agent, connecting all services."""
        if self._initialized:
            return

        self._llm_client = self.runtime.get_llm_client()
        self._provenance = ProvenanceTracker(self.config.output_dir)

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

        self._tool_registry = self._create_tool_registry()

        # ── Initialize R environment ──
        exec_cfg = self.config.execution
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
        self._afk_mode = getattr(self.config, "afk_mode", False)

        # Derive mode from config or explicit set_mode() call
        if self._afk_mode:
            self._mode = "afk"

        system_prompt = self._rebuild_system_prompt(domain)
        self.runtime.context.set_system_prompt(system_prompt)
        self.runtime.context.set_tools(
            self._tool_registry.get_schemas(format="anthropic")
        )

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

    def _rebuild_system_prompt(self, domain: str) -> str:
        """Rebuild system prompt from current mode/skills state."""
        if self._prompt_manager is None:
            return ""
        return self._prompt_manager.build_system_prompt(
            domain=domain,
            plan_enabled=True,
            data_check_enabled=self.config.enable_data_format_check,
            afk_mode=self._afk_mode,
            plan_first=self._plan_first,
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

        # ReAct loop
        max_iterations = 20
        final_content: list[dict] = []
        pending_question: dict[str, Any] | None = None

        for iteration in range(max_iterations):
            context = self.runtime.context.get_full_context()
            response = await self._llm_client.chat(
                messages=context["messages"],
                tools=context.get("tools"),
                system=context["system"],
                max_tokens=self.config.get_llm_config().tool_call_max_tokens,
            )

            # Handle text response
            text_parts = [b.get("text", "") for b in response["content"] if b.get("type") == "text"]
            agent_text = " ".join(text_parts)
            if agent_text:
                final_content.append({"type": "text", "text": agent_text})

            # Handle tool calls
            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                # No more tools — agent is done; store final assistant text
                if agent_text:
                    self.runtime.context.add_message("assistant", agent_text)
                break

            # Store assistant message with tool_use blocks first (Anthropic format requirement)
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

            # Execute tools and collect results, then batch into one message
            tool_results: list[dict[str, Any]] = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc.get("input", {})

                # ── Task tracking: create task ──
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

                # Emit tool call
                self.wire.emit(
                    EventType.TOOL_CALL,
                    {"name": tool_name, "params": tool_input},
                    sid,
                )

                # Execute tool
                start = time.perf_counter()
                result = await self._tool_registry.execute(tool_name, tool_input)
                elapsed_ms = (time.perf_counter() - start) * 1000

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

                # Emit tool result
                self.wire.emit(
                    EventType.TOOL_RESULT,
                    {"name": tool_name, "result": result},
                    sid,
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
                        break  # break from tool execution loop

            # Batch all tool results from this iteration into one message
            if tool_results:
                self.runtime.context.add_message("tool_results", tool_results)

            # Break ReAct loop if ask_user was triggered
            if pending_question is not None:
                break

            # Check for context compaction
            if self.runtime.context.needs_compaction():
                self.runtime.context.compact()

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

        self.runtime.session.touch()
        return agent_msg

    async def chat_stream(
        self,
        user_message: str,
        attachments: list[str] | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Stream agent response with incremental events."""
        if not self._initialized:
            await self.initialize()

        yield AgentStreamEvent(type="thinking", content="Analyzing request...")
        result = await self.chat(user_message, attachments)

        # Replay the result as stream events
        if isinstance(result.content, list):
            for block in result.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        yield AgentStreamEvent(type="text", content=block["text"])
                    elif block.get("type") == "tool_use":
                        yield AgentStreamEvent(
                            type="tool_call",
                            content=str(block.get("input", "")),
                            tool_name=block.get("name", ""),
                        )
        elif isinstance(result.content, str):
            yield AgentStreamEvent(type="text", content=result.content)

        yield AgentStreamEvent(type="done", content="")

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
        runs_base = self.config.output_dir / "runs"

        def _get_session_id() -> str:
            s = self.runtime.session.active_session
            return s.session_id if s else "default"

        # Execution tools
        from passi.tools.exec_tools import RunPythonTool, RunRTool

        run_python = RunPythonTool(runs_base=runs_base, session_id_provider=_get_session_id)
        run_python.python_path = exec_cfg.python_path or "python"
        registry.register(run_python, category="exec")

        run_r = RunRTool(runs_base=runs_base, session_id_provider=_get_session_id)
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
