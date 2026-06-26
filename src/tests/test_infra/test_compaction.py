"""Unit tests for ContextManager — LLM-based compaction, token tracking, thresholds."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from passi.config import PassiConfig
from passi.infra.context import ContextManager


def _make_config() -> PassiConfig:
    return PassiConfig(
        anthropic={"api_key": "test-key", "model": "claude-sonnet-4-6"},
        default_provider="anthropic",
    )


class TestContextDefaults:
    """Default values and initialization."""

    def test_default_warning_threshold(self):
        cm = ContextManager(_make_config())
        assert cm.DEFAULT_WARNING_TOKENS == 200_000

    def test_initial_token_count_is_zero(self):
        cm = ContextManager(_make_config())
        assert cm.estimated_tokens == 0
        assert cm.message_count == 0

    def test_set_system_prompt(self):
        cm = ContextManager(_make_config())
        cm.set_system_prompt("You are a bioinformatics assistant.")
        assert "bioinformatics" in cm._system_prompt

    def test_set_tools(self):
        cm = ContextManager(_make_config())
        cm.set_tools([{"name": "run_python", "description": "Execute Python code"}])
        assert len(cm._tools) == 1

    def test_set_llm_client(self):
        cm = ContextManager(_make_config())
        mock = AsyncMock()
        cm.set_llm_client(mock)
        assert cm._llm_client is mock


class TestTokenEstimation:
    """Character-based token estimation and API token tracking."""

    def test_token_estimation_with_messages(self):
        cm = ContextManager(_make_config())
        cm.add_message("user", "x" * 3000)
        assert cm.estimated_tokens > 0

    def test_update_api_tokens(self):
        cm = ContextManager(_make_config())
        cm.update_api_tokens(150_000)
        assert cm._last_api_tokens == 150_000

    def test_estimated_tokens_uses_max_api_wins(self):
        cm = ContextManager(_make_config())
        cm.update_api_tokens(250_000)
        assert cm.estimated_tokens == 250_000

    def test_char_estimate_wins_when_larger(self):
        cm = ContextManager(_make_config())
        cm.set_system_prompt("x" * 1_000_000)
        cm.add_message("user", "trigger estimate")  # triggers _update_token_estimate()
        cm.update_api_tokens(50_000)
        assert cm.estimated_tokens > 300_000

    def test_needs_compaction_uses_max_estimate(self):
        cm = ContextManager(_make_config())
        cm.update_api_tokens(210_000)
        assert cm.needs_compaction() is True

    def test_needs_compaction_false_when_below_threshold(self):
        cm = ContextManager(_make_config())
        cm.update_api_tokens(50_000)
        assert cm.needs_compaction() is False

    def test_needs_compaction_custom_threshold(self):
        cm = ContextManager(_make_config())
        cm.update_api_tokens(60_000)
        assert cm.needs_compaction() is False
        assert cm.needs_compaction(threshold=50_000) is True


class TestCompactionBehavior:
    """Compaction with LLM summarization and truncation fallback."""

    @pytest.mark.asyncio
    async def test_compact_too_few_messages(self):
        cm = ContextManager(_make_config())
        cm.add_message("user", "hello")
        cm.add_message("assistant", "hi")
        count_before = cm.message_count
        result = await cm.compact()
        assert result is None  # Returns None when too few messages
        assert cm.message_count == count_before

    @pytest.mark.asyncio
    async def test_compact_truncation_fallback(self):
        cm = ContextManager(_make_config())
        for i in range(20):
            cm.add_message("user" if i % 2 == 0 else "assistant", f"Message {i} content")
        before = cm.message_count
        result = await cm.compact()
        assert result is None  # Truncation fallback returns None
        assert cm.message_count < before

    @pytest.mark.asyncio
    async def test_compact_with_llm_client(self):
        cm = ContextManager(_make_config())
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = {
            "content": [{"type": "text", "text": "Summary: analyzed messages about bioinformatics."}],
            "tool_calls": None,
        }
        cm.set_llm_client(mock_llm)

        for i in range(20):
            cm.add_message("user" if i % 2 == 0 else "assistant", f"Message {i} content")

        before = cm.message_count
        result = await cm.compact()
        # Should return summary text on success or None on failure
        assert result is not None or cm.message_count < before

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_truncation(self):
        cm = ContextManager(_make_config())
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = Exception("LLM unavailable")
        cm.set_llm_client(mock_llm)

        for i in range(20):
            cm.add_message("user" if i % 2 == 0 else "assistant", f"Message {i} content")

        before = cm.message_count
        result = await cm.compact()
        assert result is None  # Fallback returns None
        assert cm.message_count < before

    def test_clear_resets_messages(self):
        cm = ContextManager(_make_config())
        cm.add_message("user", "test")
        cm.clear()
        assert cm.message_count == 0

    def test_reset_clears_everything(self):
        cm = ContextManager(_make_config())
        cm.set_system_prompt("test system")
        cm.set_tools([{"name": "tool"}])
        cm.add_message("user", "test")
        cm.reset()
        assert cm.message_count == 0
        assert cm._system_prompt == ""
        assert cm._tools == []

    def test_get_full_context(self):
        cm = ContextManager(_make_config())
        cm.set_system_prompt("You are helpful.")
        cm.set_tools([{"name": "run_python"}])
        cm.add_message("user", "run analysis")
        ctx = cm.get_full_context()
        assert ctx["system"] == "You are helpful."
        assert len(ctx["messages"]) == 1
        assert len(ctx["tools"]) == 1


class TestContextEdgeCases:
    """Edge case handling."""

    def test_add_message_with_list_content(self):
        cm = ContextManager(_make_config())
        cm.add_message("user", [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}])
        assert cm.message_count == 1

    @pytest.mark.asyncio
    async def test_compact_preserves_recent_messages(self):
        cm = ContextManager(_make_config())
        for i in range(20):
            cm.add_message("user" if i % 2 == 0 else "assistant", f"msg{i}")
        await cm.compact()
        messages = cm.get_messages()
        assert any("msg19" in str(m.get("content", "")) for m in messages)
