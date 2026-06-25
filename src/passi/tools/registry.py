"""Tool registry — central discovery, registration, and execution of tools.

Inspired by Kimi CLI's KimiToolset pattern.
"""

from __future__ import annotations

import logging
from typing import Any

from passi.tools.base import CallableTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all PassiAgent tools.

    Provides tool discovery, registration, and schema export for LLM integration.
    """

    def __init__(self) -> None:
        self._tools: dict[str, CallableTool] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: CallableTool, category: str = "general") -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting.", tool.name)
        self._tools[tool.name] = tool
        if category not in self._categories:
            self._categories[category] = []
        if tool.name not in self._categories[category]:
            self._categories[category].append(tool.name)
        logger.debug("Registered tool: %s [%s]", tool.name, category)

    def register_all(self, tools: list[tuple[CallableTool, str]]) -> None:
        """Register multiple tools at once."""
        for tool, category in tools:
            self.register(tool, category)

    def get(self, name: str) -> CallableTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: str | None = None) -> list[str]:
        """List tool names, optionally filtered by category."""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())

    def list_categories(self) -> dict[str, list[str]]:
        """Get all categories and their tools."""
        return dict(self._categories)

    def get_schemas(self, format: str = "openai") -> list[dict[str, Any]]:
        """Get all tool schemas for LLM integration.

        Args:
            format: 'openai' or 'anthropic'
        """
        if format == "anthropic":
            return [t.to_anthropic_schema() for t in self._tools.values()]
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(
        self, tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool by name with raw parameters.

        Parameters are validated through the tool's Pydantic model.
        """
        tool = self.get(tool_name)
        if tool is None:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}",
            }
        try:
            validated = tool.validate_params(params)
            result = await tool.execute(validated)
            result.setdefault("success", True)
            return result
        except Exception as e:
            logger.exception("Tool '%s' execution failed", tool_name)
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }

    def execute_sync(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Synchronous execution helper for non-async contexts."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.execute(tool_name, params))
        else:
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(
                self.execute(tool_name, params), loop
            )
            return future.result()
