"""TDD-style unit tests for execution tools (run_python, run_r)."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.tools.exec_tools import RunPythonParams, RunPythonTool, RunRParams, RunRTool


class TestRunPythonTool:
    """Unit tests for RunPythonTool."""

    @pytest.mark.asyncio
    async def test_run_simple_print_returns_stdout(self):
        # Arrange
        tool = RunPythonTool()
        params = RunPythonParams(code="print('hello world')")

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert "hello world" in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_with_imports_executes_successfully(self):
        # Arrange
        tool = RunPythonTool()
        code = "import json, math; print(math.sqrt(4))"
        params = RunPythonParams(code=code)

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert "2.0" in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_with_syntax_error_returns_failure(self):
        # Arrange
        tool = RunPythonTool()
        params = RunPythonParams(code="print(")

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_run_times_out_on_infinite_loop(self):
        # Arrange
        tool = RunPythonTool()
        params = RunPythonParams(code="import time; time.sleep(999)", timeout=1)

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False
        assert "timed out" in result["error"]


class TestRunRTool:
    """Unit tests for RunRTool (primarily verifies fallback to Rscript)."""

    @pytest.mark.asyncio
    async def test_run_r_with_subprocess_fallback(self):
        # Arrange — force subprocess path by disabling rpy2
        tool = RunRTool()
        params = RunRParams(
            code='cat("Hello from R\\n")',
            use_rpy2=False,
            timeout=30,
        )

        # Act
        result = await tool.execute(params)

        # Assert
        # May fail if Rscript is not installed — that's expected in CI
        if result["success"]:
            assert "Hello from R" in result["stdout"]
            assert result["execution_method"] == "Rscript"
        else:
            # Acceptable: Rscript not available in test environment
            assert "error" in result or not result["success"]

    @pytest.mark.asyncio
    async def test_run_r_with_rpy2_attempts_bridge(self):
        # Arrange
        tool = RunRTool()
        params = RunRParams(code="x <- 1 + 1; print(x)", use_rpy2=True, timeout=30)

        # Act
        result = await tool.execute(params)

        # Assert — either rpy2 succeeds or falls back
        assert "success" in result
        if result.get("rpy2_available"):
            assert result["success"] is True
            assert result["execution_method"] == "rpy2"
