"""TDD-style unit tests for ContextManager."""

from __future__ import annotations

import pytest

from passi.config import PassiConfig
from passi.infra.context import ContextManager


class TestContextManager:
    """Unit tests for ContextManager — messages, compaction, clear."""

    @pytest.fixture
    def ctx(self) -> ContextManager:
        return ContextManager(PassiConfig())

    def test_add_message_appends_to_list(self, ctx: ContextManager):
        # Act
        ctx.add_message("user", "Hello")

        # Assert
        assert ctx.message_count == 1
        assert ctx.get_messages()[0]["role"] == "user"

    def test_set_system_prompt_included_in_full_context(self, ctx: ContextManager):
        # Act
        ctx.set_system_prompt("You are a bioinformatics assistant.")
        full = ctx.get_full_context()

        # Assert
        assert "bioinformatics" in full["system"]

    def test_set_tools_included_in_full_context(self, ctx: ContextManager):
        # Act
        tools = [{"name": "tool1"}, {"name": "tool2"}]
        ctx.set_tools(tools)
        full = ctx.get_full_context()

        # Assert
        assert len(full["tools"]) == 2

    def test_get_messages_excludes_compacted(self, ctx: ContextManager):
        # Arrange
        for i in range(10):
            ctx.add_message("user", f"msg {i}")

        # Act — no compaction yet, should return all
        assert len(ctx.get_messages()) == 10

    def test_clear_removes_messages_preserves_system(self, ctx: ContextManager):
        # Arrange
        ctx.set_system_prompt("keep me")
        ctx.add_message("user", "remove me")

        # Act
        ctx.clear()

        # Assert
        assert ctx.message_count == 0
        assert "keep me" in ctx.get_full_context()["system"]

    def test_reset_removes_everything(self, ctx: ContextManager):
        # Arrange
        ctx.set_system_prompt("prompt")
        ctx.set_tools([{"name": "t"}])
        ctx.add_message("user", "msg")

        # Act
        ctx.reset()

        # Assert
        assert ctx.message_count == 0
        assert ctx.get_full_context()["system"] == ""
        assert ctx.get_full_context()["tools"] == []

    def test_needs_compaction_with_small_context_returns_false(self, ctx: ContextManager):
        # Arrange — one message is well under any threshold
        ctx.add_message("user", "tiny")

        # Act
        needs = ctx.needs_compaction()

        # Assert
        assert needs is False

    def test_estimated_tokens_increases_with_messages(self, ctx: ContextManager):
        # Arrange
        before = ctx.estimated_tokens
        ctx.add_message("user", "A" * 3000)  # ~1000 estimated tokens

        # Act
        after = ctx.estimated_tokens

        # Assert
        assert after > before
