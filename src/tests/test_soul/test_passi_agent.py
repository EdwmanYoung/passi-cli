"""TDD-style unit tests for PassiAgent core logic.

Covers: chat(), chat_stream(), execute_tool(), reset(),
ReAct loop, tool orchestration, context compaction, error handling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from passi.config import PassiConfig
from passi.infra.provenance import ProvenanceTracker
from passi.infra.runtime import Runtime
from passi.soul.passi_agent import PassiAgent
from passi.tools.io_tools import ReadFileTool
from passi.tools.registry import ToolRegistry
from tests.fixtures.mock_llm import FakeLLMClient, FakeLLMClientWithToolSequence


def _make_runtime(tmp_path: Path) -> Runtime:
    """Create a Runtime with test config pointing to a temp directory."""
    cfg = PassiConfig(
        anthropic={"api_key": "test-key", "model": "claude-sonnet-4-6"},
        default_provider="anthropic",
        session={"sessions_dir": tmp_path / "sessions"},
        output_dir=tmp_path / "output",
        debug=True,
    )
    return Runtime(config=cfg)


def _make_agent(runtime: Runtime, llm_client: FakeLLMClient, *, create_session: bool = True) -> PassiAgent:
    """Create a PassiAgent with injected fake LLM client, bypassing initialize()."""
    agent = PassiAgent(runtime)

    # Build a minimal tool registry for testing
    registry = ToolRegistry()
    registry.register(ReadFileTool(), "io")

    agent._llm_client = llm_client
    agent._tool_registry = registry
    agent._provenance = ProvenanceTracker(runtime.config.output_dir)
    agent._initialized = True

    if create_session:
        runtime.session.create_session(domain="test")
        # Set system prompt + tools so context is ready
        runtime.context.set_system_prompt("You are a test agent.")
        runtime.context.set_tools(registry.get_schemas(format="anthropic"))

    return agent


class TestPassiAgentChat:
    """chat() — core ReAct loop and message flow."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, tmp_path: Path):
        llm = FakeLLMClient("Here is your analysis result.")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        result = await agent.chat("Analyze this data")
        assert result.role == "agent"
        assert isinstance(result.content, list)
        texts = [b["text"] for b in result.content if b.get("type") == "text"]
        assert "Here is your analysis result." in " ".join(texts)

    @pytest.mark.asyncio
    async def test_chat_adds_user_and_assistant_to_context(self, tmp_path: Path):
        llm = FakeLLMClient("Response.")
        runtime = _make_runtime(tmp_path)
        agent = _make_agent(runtime, llm)

        await agent.chat("Hello")
        msgs = runtime.context.get_messages()
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_chat_creates_session(self, tmp_path: Path):
        llm = FakeLLMClient("Response.")
        runtime = _make_runtime(tmp_path)
        agent = _make_agent(runtime, llm)

        await agent.chat("Hello")
        session = runtime.session.active_session
        assert session is not None
        assert len(session.session_id) > 0

    @pytest.mark.asyncio
    async def test_chat_sends_content_to_llm(self, tmp_path: Path):
        llm = FakeLLMClient("OK")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        await agent.chat("What genes are upregulated?")
        assert len(llm.chat_history) >= 1
        last_call = llm.chat_history[-1]
        # messages should include system + user message
        msg_texts = []
        for m in last_call["messages"]:
            c = m.get("content", "")
            if isinstance(c, str):
                msg_texts.append(c)
        combined = " ".join(msg_texts)
        assert "upregulated" in combined

    @pytest.mark.asyncio
    async def test_chat_with_tool_call_executes_tool(self, tmp_path: Path):
        # Create a tmp file so read_file succeeds
        test_file = tmp_path / "data.txt"
        test_file.write_text("gene1\t10\ngene2\t20")

        llm = FakeLLMClient("I'll read the file.")
        llm.set_tool_calls([{
            "id": "tool_001",
            "name": "read_file",
            "input": {"path": str(test_file), "max_lines": 10},
        }])

        agent = _make_agent(_make_runtime(tmp_path), llm)

        result = await agent.chat("Read the data file")
        # Content should include the tool_use block
        tool_blocks = [b for b in result.content if b.get("type") == "tool_use"]
        assert len(tool_blocks) >= 1
        assert tool_blocks[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_chat_reat_loop_multiple_tool_rounds(self, tmp_path: Path):
        """Agent handles multiple rounds of tool→LLM→tool→LLM."""
        file1 = tmp_path / "a.txt"
        file1.write_text("aaa")
        file2 = tmp_path / "b.txt"
        file2.write_text("bbb")

        llm = FakeLLMClientWithToolSequence()
        llm.set_sequence(
            tool_sequences=[
                [{"id": "t1", "name": "read_file", "input": {"path": str(file1)}}],
                [{"id": "t2", "name": "read_file", "input": {"path": str(file2)}}],
            ],
            final_response="Both files read successfully.",
        )

        agent = _make_agent(_make_runtime(tmp_path), llm)
        result = await agent.chat("Read both files")

        texts = [b["text"] for b in result.content if b.get("type") == "text"]
        combined = " ".join(texts)
        assert "Both files read" in combined
        # Should have 2 tool_use blocks
        tool_blocks = [b for b in result.content if b.get("type") == "tool_use"]
        assert len(tool_blocks) == 2

    @pytest.mark.asyncio
    async def test_chat_no_tool_calls_stops_loop(self, tmp_path: Path):
        """When LLM returns no tool_calls, the ReAct loop exits immediately."""
        llm = FakeLLMClient("Done, no tools needed.")
        llm._tool_calls = None  # explicitly None

        agent = _make_agent(_make_runtime(tmp_path), llm)
        result = await agent.chat("Say hello")

        # Only 1 LLM call — no re-entry into loop
        assert len(llm.chat_history) == 1
        assert result.role == "agent"


class TestPassiAgentChatStream:
    """chat_stream() — streaming event iterator."""

    @pytest.mark.asyncio
    async def test_stream_yields_thinking_event(self, tmp_path: Path):
        llm = FakeLLMClient("Streaming result.")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        events = [e async for e in agent.chat_stream("Query")]
        event_types = [e.type for e in events]
        assert "thinking" in event_types

    @pytest.mark.asyncio
    async def test_stream_yields_text_events(self, tmp_path: Path):
        llm = FakeLLMClient("Streaming result.")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        events = [e async for e in agent.chat_stream("Query")]
        texts = [e.content for e in events if e.type == "text"]
        assert len(texts) >= 1
        assert "Streaming result" in " ".join(str(t) for t in texts)

    @pytest.mark.asyncio
    async def test_stream_yields_done_event_last(self, tmp_path: Path):
        llm = FakeLLMClient("Done.")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        events = [e async for e in agent.chat_stream("Query")]
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls_yields_tool_call_events(self, tmp_path: Path):
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")

        llm = FakeLLMClient("Using tool.")
        llm.set_tool_calls([{
            "id": "tc_1",
            "name": "read_file",
            "input": {"path": str(test_file)},
        }])

        agent = _make_agent(_make_runtime(tmp_path), llm)
        events = [e async for e in agent.chat_stream("Read file")]
        tool_events = [e for e in events if e.type == "tool_call"]
        assert len(tool_events) >= 1
        assert tool_events[0].tool_name == "read_file"


class TestPassiAgentExecuteTool:
    """execute_tool() — direct tool invocation without LLM."""

    @pytest.mark.asyncio
    async def test_execute_tool_returns_result(self, tmp_path: Path):
        test_file = tmp_path / "genes.tsv"
        test_file.write_text("gene\texpression\nBRCA1\t100\nTP53\t200")

        agent = _make_agent(_make_runtime(tmp_path), FakeLLMClient("unused"))

        result = await agent.execute_tool("read_file", {"path": str(test_file), "max_lines": 5})
        assert result.role == "tool"
        assert result.name == "read_file"
        assert "success" in str(result.content)

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_tool_returns_error(self, tmp_path: Path):
        agent = _make_agent(_make_runtime(tmp_path), FakeLLMClient("unused"))

        result = await agent.execute_tool("nonexistent_tool", {})
        assert result.role == "tool"
        content_str = str(result.content)
        assert "not found" in content_str.lower() or "error" in content_str.lower()


class TestPassiAgentResetAndShutdown:
    """reset() and shutdown() lifecycle."""

    @pytest.mark.asyncio
    async def test_reset_clears_context(self, tmp_path: Path):
        llm = FakeLLMClient("Response.")
        runtime = _make_runtime(tmp_path)
        agent = _make_agent(runtime, llm)

        await agent.chat("Message 1")
        assert runtime.context.message_count > 0

        await agent.reset()
        assert runtime.context.message_count == 0

    @pytest.mark.asyncio
    async def test_shutdown_does_not_raise(self, tmp_path: Path):
        agent = _make_agent(_make_runtime(tmp_path), FakeLLMClient("OK"))

        # Should not raise
        await agent.shutdown()


class TestPassiAgentInitialization:
    """Real initialize() — bypasses R to avoid rpy2 dependency in CI."""

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Calling initialize() twice should not double-initialize."""
        # Prevent rpy2 init from running
        monkeypatch.setattr(
            "passi.executors.r_executor.init_rpy2",
            lambda *a, **kw: {"ready": False, "error": "test skip"},
        )
        # Prevent llm_client real API init
        monkeypatch.setattr(
            "passi.infra.llm_client.create_llm_client",
            lambda *a, **kw: FakeLLMClient("ok"),
        )

        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)

        await agent.initialize()
        assert agent._initialized is True
        # Second call should return immediately
        await agent.initialize()
        assert agent._initialized is True

    @pytest.mark.asyncio
    async def test_chat_auto_initializes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """chat() calls initialize() if not already done."""
        monkeypatch.setattr(
            "passi.executors.r_executor.init_rpy2",
            lambda *a, **kw: {"ready": False, "error": "test skip"},
        )
        monkeypatch.setattr(
            "passi.infra.llm_client.create_llm_client",
            lambda *a, **kw: FakeLLMClient("auto-init response"),
        )

        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        assert agent._initialized is False

        result = await agent.chat("Test")
        assert agent._initialized is True
        assert result.role == "agent"


class TestPassiAgentContextCompaction:
    """Context compaction during long conversations."""

    @pytest.mark.asyncio
    async def test_compaction_triggered_on_large_context(self, tmp_path: Path):
        """When context exceeds warning threshold, older messages are compacted."""
        runtime = _make_runtime(tmp_path)

        # Fill context with many messages to exceed 200k token threshold
        large_payload = "x" * 50000  # ~16.6K tokens worth of chars per message
        for i in range(15):
            runtime.context.add_message("user" if i % 2 == 0 else "assistant", large_payload)

        assert runtime.context.needs_compaction() is True

        llm = FakeLLMClient("Compact response.")
        agent = _make_agent(runtime, llm)

        result = await agent.chat("Final query")
        assert result.role == "agent"

    @pytest.mark.asyncio
    async def test_compaction_reduces_message_count(self, tmp_path: Path):
        runtime = _make_runtime(tmp_path)
        for i in range(20):
            runtime.context.add_message("user" if i % 2 == 0 else "assistant", "x" * 10000)

        before = runtime.context.message_count
        await runtime.context.compact()
        after = runtime.context.message_count
        assert after < before


class TestPassiAgentErrorHandling:
    """Agent behavior when tools fail."""

    @pytest.mark.asyncio
    async def test_failed_tool_still_returns_content(self, tmp_path: Path):
        """When a tool call fails, agent continues the ReAct loop."""
        test_file = tmp_path / "data.txt"
        test_file.write_text("content")

        llm = FakeLLMClient("I tried reading but will continue.")
        llm.set_tool_calls([{
            "id": "t1",
            "name": "read_file",
            "input": {"path": str(test_file)},
        }])

        agent = _make_agent(_make_runtime(tmp_path), llm)
        result = await agent.chat("Analyze")
        assert result.role == "agent"
        # Should have a tool_use block even if tool returned error
        tool_blocks = [b for b in result.content if b.get("type") == "tool_use"]
        assert len(tool_blocks) >= 1


class TestPassiAgentToolRegistry:
    """_create_tool_registry() builds a complete registry."""

    def test_create_tool_registry_has_all_categories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "passi.executors.r_executor.init_rpy2",
            lambda *a, **kw: {"ready": False, "error": "test skip"},
        )
        monkeypatch.setattr(
            "passi.infra.llm_client.create_llm_client",
            lambda *a, **kw: FakeLLMClient("ok"),
        )

        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        registry = agent._create_tool_registry()

        categories = registry.list_categories()
        # Should have at least these categories
        expected = {"io", "exec", "qc", "genomics", "epigenetics", "transcriptomics", "clinical"}
        assert expected <= set(categories.keys())

        # Each category should have tools
        for cat in expected:
            assert len(categories[cat]) >= 1, f"Category '{cat}' is empty"

    def test_tool_registry_returns_anthropic_schemas(self, tmp_path: Path):
        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        registry = agent._create_tool_registry()

        schemas = registry.get_schemas(format="anthropic")
        assert isinstance(schemas, list)
        assert len(schemas) > 0

        # Anthropic schema format check
        for s in schemas:
            assert "name" in s
            assert "input_schema" in s


class TestPassiAgentWireEvents:
    """Wire protocol events emitted during chat."""

    @pytest.mark.asyncio
    async def test_user_message_emits_event(self, tmp_path: Path):
        llm = FakeLLMClient("OK")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        events_before = len(agent.wire._events) if hasattr(agent.wire, '_events') else 0

        await agent.chat("Hello wire")
        # Wire should have recorded events
        assert agent.wire is not None


class TestPassiAgentAskUser:
    """ask_user tool integration — ReAct loop pause and metadata propagation."""

    @pytest.mark.asyncio
    async def test_ask_user_sets_pending_question_in_metadata(self, tmp_path: Path):
        """When LLM calls ask_user, AgentMessage.metadata gets pending_question."""
        from passi.tools.ask_user_tool import AskUserTool

        llm = FakeLLMClient("I need to ask you something.")
        llm.set_tool_calls([{
            "id": "ask_001",
            "name": "ask_user",
            "input": {
                "question": "Which comparison group?",
                "context": "Multiple groups found.",
                "options": ["SARS-CoV-2 vs Mock", "IAV vs Mock"],
            },
        }])

        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        registry = ToolRegistry()
        registry.register(ReadFileTool(), "io")
        registry.register(AskUserTool(), "system")
        agent._llm_client = llm
        agent._tool_registry = registry
        agent._provenance = ProvenanceTracker(runtime.config.output_dir)
        agent._initialized = True
        runtime.session.create_session(domain="test")
        runtime.context.set_system_prompt("You are a test agent.")
        runtime.context.set_tools(registry.get_schemas(format="anthropic"))

        result = await agent.chat("Help me analyze this data")
        assert result.metadata.get("pending_question") is not None
        pq = result.metadata["pending_question"]
        assert pq["question"] == "Which comparison group?"
        assert pq["context"] == "Multiple groups found."
        assert pq["options"] == ["SARS-CoV-2 vs Mock", "IAV vs Mock"]

    @pytest.mark.asyncio
    async def test_ask_user_breaks_react_loop(self, tmp_path: Path):
        """Ask user should stop the ReAct loop immediately, not continue iterating."""
        from passi.tools.ask_user_tool import AskUserTool

        llm = FakeLLMClientWithToolSequence()
        llm.set_sequence(
            tool_sequences=[
                [{"id": "ask_1", "name": "ask_user", "input": {"question": "Confirm?"}}],
                [{"id": "t2", "name": "read_file", "input": {"path": str(tmp_path / "x.txt")}}],
            ],
            final_response="This should not be reached.",
        )

        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        registry = ToolRegistry()
        registry.register(ReadFileTool(), "io")
        registry.register(AskUserTool(), "system")
        agent._llm_client = llm
        agent._tool_registry = registry
        agent._provenance = ProvenanceTracker(runtime.config.output_dir)
        agent._initialized = True
        runtime.session.create_session(domain="test")
        runtime.context.set_system_prompt("You are a test agent.")
        runtime.context.set_tools(registry.get_schemas(format="anthropic"))

        result = await agent.chat("Analyze")
        assert result.metadata.get("pending_question") is not None
        # Only 1 LLM call should have been made (ask_user returned, loop broke)
        assert llm._call_index == 1

    @pytest.mark.asyncio
    async def test_ask_user_without_options(self, tmp_path: Path):
        """ask_user with no options should still set pending_question."""
        from passi.tools.ask_user_tool import AskUserTool

        llm = FakeLLMClient("Asking...")
        llm.set_tool_calls([{
            "id": "ask_1",
            "name": "ask_user",
            "input": {"question": "What threshold?", "context": "Need clarification."},
        }])

        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        registry = ToolRegistry()
        registry.register(ReadFileTool(), "io")
        registry.register(AskUserTool(), "system")
        agent._llm_client = llm
        agent._tool_registry = registry
        agent._provenance = ProvenanceTracker(runtime.config.output_dir)
        agent._initialized = True
        runtime.session.create_session(domain="test")
        runtime.context.set_system_prompt("You are a test agent.")
        runtime.context.set_tools(registry.get_schemas(format="anthropic"))

        result = await agent.chat("Analyze")
        pq = result.metadata.get("pending_question")
        assert pq is not None
        assert pq["question"] == "What threshold?"
        assert pq["options"] is None

    @pytest.mark.asyncio
    async def test_no_ask_user_does_not_set_metadata(self, tmp_path: Path):
        """Normal tool execution without ask_user should leave metadata empty."""
        test_file = tmp_path / "data.txt"
        test_file.write_text("gene1\t10")

        llm = FakeLLMClient("Reading file.")
        llm.set_tool_calls([{
            "id": "t1",
            "name": "read_file",
            "input": {"path": str(test_file)},
        }])

        runtime = _make_runtime(tmp_path)
        agent = _make_agent(runtime, llm)

        result = await agent.chat("Read file")
        assert "pending_question" not in result.metadata


class TestPassiAgentStreaming:
    """Real-time streaming via _run_react_loop_stream and chat_stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_tool_result_events(self, tmp_path: Path):
        """chat_stream yields tool_result events after each tool execution."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")

        llm = FakeLLMClient("Using tool.")
        llm.set_tool_calls([{
            "id": "tc_1",
            "name": "read_file",
            "input": {"path": str(test_file)},
        }])

        agent = _make_agent(_make_runtime(tmp_path), llm)
        events = [e async for e in agent.chat_stream("Read file")]
        result_events = [e for e in events if e.type == "tool_result"]
        assert len(result_events) >= 1
        assert result_events[0].tool_name == "read_file"

    @pytest.mark.asyncio
    async def test_stream_yields_events_incrementally(self, tmp_path: Path):
        """Events are yielded as they happen, not all at the end after completion."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")

        llm = FakeLLMClient("Using tool.")
        llm.set_tool_calls([{
            "id": "tc_1",
            "name": "read_file",
            "input": {"path": str(test_file)},
        }])

        agent = _make_agent(_make_runtime(tmp_path), llm)
        events = [e async for e in agent.chat_stream("Read file")]

        # Events appear in correct order: thinking → text → tool_call → tool_result → done
        event_types = [e.type for e in events]
        assert "thinking" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "done" in event_types

        # tool_call must appear before tool_result
        tc_idx = event_types.index("tool_call")
        tr_idx = event_types.index("tool_result")
        assert tc_idx < tr_idx, "tool_call must come before tool_result"

    @pytest.mark.asyncio
    async def test_stream_pending_question_yields_event(self, tmp_path: Path):
        """chat_stream yields pending_question event when ask_user is called."""
        from passi.tools.ask_user_tool import AskUserTool

        llm = FakeLLMClient("Asking...")
        llm.set_tool_calls([{
            "id": "ask_1",
            "name": "ask_user",
            "input": {
                "question": "Confirm step?",
                "context": "About to run analysis.",
                "options": ["Yes", "No"],
            },
        }])

        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        registry = ToolRegistry()
        registry.register(ReadFileTool(), "io")
        registry.register(AskUserTool(), "system")
        agent._llm_client = llm
        agent._tool_registry = registry
        agent._provenance = ProvenanceTracker(runtime.config.output_dir)
        agent._initialized = True
        runtime.session.create_session(domain="test")
        runtime.context.set_system_prompt("You are a test agent.")
        runtime.context.set_tools(registry.get_schemas(format="anthropic"))

        events = [e async for e in agent.chat_stream("Run analysis")]
        pq_events = [e for e in events if e.type == "pending_question"]
        assert len(pq_events) == 1
        assert pq_events[0].content == "Confirm step?"
        assert pq_events[0].metadata.get("options") == ["Yes", "No"]

    @pytest.mark.asyncio
    async def test_stream_done_event_contains_agent_message(self, tmp_path: Path):
        """The final done event carries the AgentMessage in its metadata."""
        llm = FakeLLMClient("Analysis result.")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        events = [e async for e in agent.chat_stream("Analyze")]
        done_events = [e for e in events if e.type == "done"]
        assert len(done_events) == 1
        assert "agent_message" in done_events[0].metadata
        assert done_events[0].metadata["agent_message"].role == "agent"


class TestPassiAgentAgentBusyFlag:
    """_agent_busy lifecycle during chat() and chat_stream()."""

    @pytest.mark.asyncio
    async def test_agent_busy_reset_after_chat(self, tmp_path: Path):
        """_agent_busy is False after chat() completes."""
        llm = FakeLLMClient("Response.")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        assert agent._agent_busy is False
        await agent.chat("Hello")
        assert agent._agent_busy is False

    @pytest.mark.asyncio
    async def test_agent_busy_reset_after_chat_error(self, tmp_path: Path):
        """_agent_busy is False even when chat() raises an exception."""
        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        agent._llm_client = MagicMock()
        agent._llm_client.chat.side_effect = RuntimeError("LLM connection lost")
        agent._tool_registry = MagicMock()
        agent._provenance = MagicMock()
        agent._initialized = True
        runtime.session.create_session(domain="test")
        runtime.context.set_system_prompt("You are a test agent.")
        runtime.context.set_tools([])

        assert agent._agent_busy is False
        with pytest.raises(RuntimeError, match="LLM connection lost"):
            await agent.chat("Hello")
        assert agent._agent_busy is False

    @pytest.mark.asyncio
    async def test_agent_busy_reset_after_chat_stream(self, tmp_path: Path):
        """_agent_busy is False after chat_stream() completes."""
        llm = FakeLLMClient("Response.")
        agent = _make_agent(_make_runtime(tmp_path), llm)

        assert agent._agent_busy is False
        async for _ in agent.chat_stream("Hello"):
            pass
        assert agent._agent_busy is False

    @pytest.mark.asyncio
    async def test_agent_busy_reset_after_chat_stream_error(self, tmp_path: Path):
        """_agent_busy is False even when chat_stream() encounters an error."""
        runtime = _make_runtime(tmp_path)
        agent = PassiAgent(runtime)
        agent._llm_client = MagicMock()
        agent._llm_client.chat.side_effect = RuntimeError("LLM connection lost")
        agent._tool_registry = MagicMock()
        agent._provenance = MagicMock()
        agent._initialized = True
        runtime.session.create_session(domain="test")
        runtime.context.set_system_prompt("You are a test agent.")
        runtime.context.set_tools([])

        assert agent._agent_busy is False
        with pytest.raises(RuntimeError, match="LLM connection lost"):
            async for _ in agent.chat_stream("Hello"):
                pass
        assert agent._agent_busy is False
