"""Unit tests for AskUserTool — marker, params validation, execution."""

from __future__ import annotations

import pytest

from passi.tools.ask_user_tool import AskUserParams, AskUserTool


class TestAskUserTool:
    """AskUserTool unit tests."""

    def test_name_and_description(self):
        tool = AskUserTool()
        assert tool.name == "ask_user"
        assert "clarifying question" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_execute_returns_ask_user_marker(self):
        tool = AskUserTool()
        result = await tool.execute(AskUserParams(
            question="Which comparison group?",
            context="Multiple groups detected in the data.",
            options=["SARS-CoV-2 vs Mock", "IAV vs Mock"],
        ))
        assert result["success"] is True
        assert result["__ask_user__"] is True
        assert result["question"] == "Which comparison group?"
        assert result["context"] == "Multiple groups detected in the data."
        assert result["options"] == ["SARS-CoV-2 vs Mock", "IAV vs Mock"]

    @pytest.mark.asyncio
    async def test_execute_without_context_or_options(self):
        tool = AskUserTool()
        result = await tool.execute(AskUserParams(
            question="What significance threshold?",
        ))
        assert result["success"] is True
        assert result["__ask_user__"] is True
        assert result["context"] == ""
        assert result["options"] is None

    @pytest.mark.asyncio
    async def test_execute_with_empty_options_list(self):
        tool = AskUserTool()
        result = await tool.execute(AskUserParams(
            question="Confirm analysis method?",
            options=[],
        ))
        assert result["__ask_user__"] is True
        assert result["options"] == []

    def test_params_model_validation(self):
        """Params should require question field."""
        with pytest.raises(Exception):
            AskUserParams()  # missing required 'question'

        params = AskUserParams(question="Test?")
        assert params.question == "Test?"
        assert params.context == ""
        assert params.options is None

    def test_params_model_with_all_fields(self):
        params = AskUserParams(
            question="Which method?",
            context="DESeq2 vs edgeR trade-off.",
            options=["DESeq2", "edgeR"],
        )
        assert params.question == "Which method?"
        assert len(params.options) == 2
