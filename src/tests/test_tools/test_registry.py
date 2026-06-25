"""TDD-style unit tests for ToolRegistry."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from passi.tools.base import CallableTool
from passi.tools.registry import ToolRegistry


class _EchoParams(BaseModel):
    message: str = Field(default="hello")


class _EchoTool(CallableTool[_EchoParams]):
    name = "echo"
    description = "Echoes the input message"
    params_model = _EchoParams

    async def execute(self, params: _EchoParams, **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "result": params.message}


class _FailingTool(CallableTool[_EchoParams]):
    name = "failing"
    description = "Always fails"
    params_model = _EchoParams

    async def execute(self, params: _EchoParams, **kwargs: Any) -> dict[str, Any]:
        msg = "Deliberate failure"
        raise RuntimeError(msg)


class TestToolRegistry:
    """Tests for ToolRegistry — register, get, execute, schemas."""

    def test_register_adds_tool(self, tool_registry: ToolRegistry):
        # Arrange
        tool = _EchoTool()

        # Act
        tool_registry.register(tool, category="test")

        # Assert
        assert tool_registry.get("echo") is tool
        assert "echo" in tool_registry.list_tools()
        assert "echo" in tool_registry.list_tools(category="test")

    def test_get_unknown_tool_returns_none(self, tool_registry: ToolRegistry):
        # Act
        result = tool_registry.get("nonexistent")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_valid_tool_returns_success(self, tool_registry: ToolRegistry):
        # Arrange
        tool_registry.register(_EchoTool(), category="test")

        # Act
        result = await tool_registry.execute("echo", {"message": "hi"})

        # Assert
        assert result["success"] is True
        assert result["result"] == "hi"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self, tool_registry: ToolRegistry):
        # Act
        result = await tool_registry.execute("ghost", {})

        # Assert
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_exception_returns_error(self, tool_registry: ToolRegistry):
        # Arrange
        tool_registry.register(_FailingTool())

        # Act
        result = await tool_registry.execute("failing", {})

        # Assert
        assert result["success"] is False
        assert "Deliberate failure" in result["error"]

    def test_get_schemas_openai_returns_function_format(self, tool_registry: ToolRegistry):
        # Arrange
        tool_registry.register(_EchoTool())

        # Act
        schemas = tool_registry.get_schemas(format="openai")

        # Assert
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"

    def test_list_categories_returns_organized_tools(self, tool_registry: ToolRegistry):
        # Arrange
        tool_registry.register(_EchoTool(), category="io")
        tool_registry.register(_FailingTool(), category="test")

        # Act
        categories = tool_registry.list_categories()

        # Assert
        assert "io" in categories
        assert "test" in categories
        assert "echo" in categories["io"]
