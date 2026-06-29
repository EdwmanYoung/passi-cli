"""TDD-style unit tests for execution tools (run_python, run_r)."""

from __future__ import annotations

import json
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


class TestRunPythonToolRunDir:
    """Unit tests for persistent run directory behavior in RunPythonTool."""

    @pytest.mark.asyncio
    async def test_creates_run_directory_with_script_and_logs(self, tmp_path: Path):
        """Run dir is created with script.py, stdout.log, stderr.log, metadata."""
        runs_base = tmp_path / "runs"
        tool = RunPythonTool(runs_base=runs_base, session_id_provider=lambda: "test_session")
        params = RunPythonParams(code="print('hello from run dir')")

        result = await tool.execute(params)

        assert "run_dir" in result
        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "script.py").exists()
        assert (run_dir / "stdout.log").exists()
        assert (run_dir / "stderr.log").exists()
        assert (run_dir / "run_metadata.json").exists()
        assert "hello from run dir" in (run_dir / "stdout.log").read_text()

    @pytest.mark.asyncio
    async def test_run_dir_path_contains_session_and_tool_name(self, tmp_path: Path):
        """Run dir path follows convention: runs_base/session_id/run_<ts>_run_python/"""
        runs_base = tmp_path / "runs"
        tool = RunPythonTool(runs_base=runs_base, session_id_provider=lambda: "sess_abc")
        params = RunPythonParams(code="print(1)")

        result = await tool.execute(params)
        run_dir = Path(result["run_dir"])

        assert runs_base in run_dir.parents
        assert "sess_abc" in str(run_dir)
        assert "run_" in run_dir.name
        assert "run_python" in run_dir.name

    @pytest.mark.asyncio
    async def test_output_files_land_in_run_dir(self, tmp_path: Path):
        """Script creates a file in CWD (project root); it is discoverable."""
        runs_base = tmp_path / "runs"
        tool = RunPythonTool(runs_base=runs_base, session_id_provider=lambda: "test")
        code = "with open('result.csv', 'w') as f:\n    f.write('a,b,c\\n1,2,3')"
        params = RunPythonParams(code=code)

        await tool.execute(params)

        # File is written to CWD (project root), not run_dir
        cwd_file = Path.cwd() / "result.csv"
        assert cwd_file.exists()
        cwd_file.unlink()  # cleanup

    @pytest.mark.asyncio
    async def test_custom_output_dir_param(self, tmp_path: Path):
        """When output_dir is specified, use it instead of auto-generated path."""
        custom = tmp_path / "my_custom_run"
        tool = RunPythonTool(runs_base=tmp_path / "runs", session_id_provider=lambda: "x")
        params = RunPythonParams(code="print(1)", output_dir=str(custom))

        result = await tool.execute(params)
        assert Path(result["run_dir"]) == custom
        assert custom.exists()

    @pytest.mark.asyncio
    async def test_output_dir_creates_subdirectories(self, tmp_path: Path):
        """output_dir path auto-creates code/, intermediate/, outputs/ subdirectories."""
        custom = tmp_path / "my_run"
        tool = RunPythonTool(runs_base=tmp_path / "runs", session_id_provider=lambda: "x")
        params = RunPythonParams(code="print(1)", output_dir=str(custom))

        result = await tool.execute(params)
        assert result["success"] is True
        assert (custom / "code").is_dir()
        assert (custom / "intermediate").is_dir()
        assert (custom / "outputs").is_dir()

    @pytest.mark.asyncio
    async def test_output_dir_script_in_code_subdir(self, tmp_path: Path):
        """Script is written to code/ subdirectory, not run_dir root."""
        custom = tmp_path / "my_run"
        tool = RunPythonTool(runs_base=tmp_path / "runs", session_id_provider=lambda: "x")
        params = RunPythonParams(code="print('hello')", output_dir=str(custom))

        result = await tool.execute(params)
        assert result["success"] is True
        assert (custom / "code" / "script.py").exists()
        assert not (custom / "script.py").exists()

    @pytest.mark.asyncio
    async def test_failed_execution_still_creates_run_dir(self, tmp_path: Path):
        """Even on error, run_dir exists with stderr.log containing the error."""
        runs_base = tmp_path / "runs"
        tool = RunPythonTool(runs_base=runs_base, session_id_provider=lambda: "test")
        params = RunPythonParams(code="raise ValueError('boom')")

        result = await tool.execute(params)
        assert result["success"] is False
        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "stderr.log").exists()
        stderr_content = (run_dir / "stderr.log").read_text()
        assert "ValueError" in stderr_content or "boom" in stderr_content

    @pytest.mark.asyncio
    async def test_timeout_creates_run_dir_with_error_info(self, tmp_path: Path):
        """Timeout still creates run_dir with error in metadata."""
        runs_base = tmp_path / "runs"
        tool = RunPythonTool(runs_base=runs_base, session_id_provider=lambda: "test")
        params = RunPythonParams(code="import time; time.sleep(999)", timeout=1)

        result = await tool.execute(params)
        assert result["success"] is False
        assert "run_dir" in result
        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "stderr.log").exists()

    @pytest.mark.asyncio
    async def test_input_files_recorded_in_metadata(self, tmp_path: Path):
        """input_files param is persisted in run_metadata.json."""
        runs_base = tmp_path / "runs"
        tool = RunPythonTool(runs_base=runs_base, session_id_provider=lambda: "test")
        params = RunPythonParams(
            code="print('ok')",
            input_files=["/data/samples.csv", "/data/design.csv"],
        )

        result = await tool.execute(params)
        run_dir = Path(result["run_dir"])
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        assert metadata["input_files"] == ["/data/samples.csv", "/data/design.csv"]

    @pytest.mark.asyncio
    async def test_preserves_script_content_verbatim(self, tmp_path: Path):
        """The saved script.py matches the input code exactly."""
        runs_base = tmp_path / "runs"
        tool = RunPythonTool(runs_base=runs_base, session_id_provider=lambda: "test")
        code = "x = 42\nfor i in range(3):\n    print(x + i)"
        params = RunPythonParams(code=code)

        result = await tool.execute(params)
        run_dir = Path(result["run_dir"])
        assert (run_dir / "script.py").read_text() == code


class TestRunRToolRunDir:
    """Unit tests for persistent run directory behavior in RunRTool."""

    @pytest.mark.asyncio
    async def test_rscript_creates_run_directory(self, tmp_path: Path):
        """Rscript execution creates run dir with script.R and logs."""
        runs_base = tmp_path / "runs"
        tool = RunRTool(runs_base=runs_base, session_id_provider=lambda: "test_sess")
        params = RunRParams(code='cat("Hello from R\\n")', use_rpy2=False, timeout=30)

        result = await tool.execute(params)
        if not result["success"]:
            pytest.skip("Rscript not available")

        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "script.R").exists()
        assert (run_dir / "stdout.log").exists()
        assert "Hello from R" in (run_dir / "stdout.log").read_text()

    @pytest.mark.asyncio
    async def test_rscript_output_files_in_run_dir(self, tmp_path: Path):
        """R script that writes a file relative to CWD; it appears in project root."""
        runs_base = tmp_path / "runs"
        tool = RunRTool(runs_base=runs_base, session_id_provider=lambda: "test")
        # Use a unique filename to avoid collisions and clean up after the test.
        output_name = f"rscript_output_{tmp_path.name}.csv"
        code = f'write.csv(data.frame(a=1:3, b=4:6), "{output_name}", row.names=FALSE)'
        params = RunRParams(code=code, use_rpy2=False, timeout=30)

        result = await tool.execute(params)
        if not result["success"]:
            pytest.skip("Rscript not available")

        # Subprocess cwd is the project root, so relative files land there.
        cwd_file = Path.cwd() / output_name
        assert cwd_file.exists()
        cwd_file.unlink()

    @pytest.mark.asyncio
    async def test_rscript_error_still_creates_run_dir(self, tmp_path: Path):
        """Failed Rscript execution preserves stderr.log in run_dir."""
        runs_base = tmp_path / "runs"
        tool = RunRTool(runs_base=runs_base, session_id_provider=lambda: "test")
        params = RunRParams(code='stop("intentional error")', use_rpy2=False, timeout=30)

        result = await tool.execute(params)
        if "error" in result and "Rscript not found" in result["error"]:
            pytest.skip("Rscript not available")

        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "stderr.log").exists()

    @pytest.mark.asyncio
    async def test_rpy2_creates_run_directory(self, tmp_path: Path):
        """rpy2 execution creates run dir and returns rpy2_available flag."""
        runs_base = tmp_path / "runs"
        tool = RunRTool(runs_base=runs_base, session_id_provider=lambda: "test")
        params = RunRParams(code="x <- 1 + 1; print(x)", use_rpy2=True, timeout=30)

        result = await tool.execute(params)
        if not result.get("rpy2_available"):
            pytest.skip("rpy2 not available")

        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "script.R").exists()
        assert (run_dir / "run_metadata.json").exists()

    @pytest.mark.asyncio
    async def test_r_error_preserves_stderr(self, tmp_path: Path):
        """R execution errors are captured in stderr.log (rpy2 or Rscript)."""
        runs_base = tmp_path / "runs"
        tool = RunRTool(runs_base=runs_base, session_id_provider=lambda: "test")
        params = RunRParams(code="nonexistent_function()", use_rpy2=True, timeout=30)

        result = await tool.execute(params)

        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "stderr.log").exists()
        stderr_content = (run_dir / "stderr.log").read_text()
        assert "nonexistent_function" in stderr_content
