"""Fixtures for tool-layer tests."""

from __future__ import annotations

import pytest

from passi.tools.registry import ToolRegistry


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """A fresh, empty ToolRegistry for isolated tool tests."""
    return ToolRegistry()
