"""Code execution tools — run Python and R code in sandboxed environments.

Each execution creates a persistent run directory preserving the script, full
stdout/stderr logs, metadata, and any output files the code produces.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from passi.tools.base import CallableTool

# ── Run directory helper ─────────────────────────────────────────────────


def _finalize_run_dir(
    run_dir: Path,
    tool_name: str,
    input_files: list[str],
    exit_code: int,
    duration_ms: float,
    execution_method: str,
    error: str = "",
) -> list[str]:
    """Scan run_dir for output files and write run_metadata.json.

    Returns list of output file paths (relative to run_dir) discovered after
    execution, excluding the tool's own artifacts.
    """
    known_files = {"script.py", "script.R", "stdout.log", "stderr.log", "run_metadata.json"}
    output_files = sorted(
        str(p) for p in run_dir.iterdir()
        if p.is_file() and p.name not in known_files
    )
    metadata = {
        "tool": tool_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "input_files": input_files,
        "output_files": output_files,
        "execution_method": execution_method,
        "error": error,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return output_files


# ── Python Execution ─────────────────────────────────────────────────────


class RunPythonParams(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=120, description="Execution timeout in seconds")
    packages: list[str] | None = Field(default=None, description="Extra packages to ensure are installed")
    input_files: list[str] = Field(default_factory=list, description="Input file paths the code depends on")
    output_dir: str = Field(default="", description="Custom run directory; auto-generated if empty")


class RunPythonTool(CallableTool[RunPythonParams]):
    name = "run_python"
    description = (
        "Execute Python code in a sandboxed subprocess. Use for data analysis, "
        "statistical computations, and visualization. Returns stdout, stderr, and exit code. "
        "A run directory is created with script, logs, and any output files preserved for inspection."
    )
    params_model = RunPythonParams

    def __init__(
        self,
        runs_base: Path | None = None,
        session_id_provider: Callable[[], str] | None = None,
    ) -> None:
        self.runs_base = runs_base or Path("output") / "runs"
        self._session_id_provider = session_id_provider or (lambda: "default")

    async def execute(self, params: RunPythonParams, **kwargs: Any) -> dict[str, Any]:
        # Determine run directory
        if params.output_dir:
            run_dir = Path(params.output_dir)
        else:
            sid = self._session_id_provider()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            run_dir = self.runs_base / sid / f"run_{ts}_{self.name}"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Write script to run dir (preserved, not deleted)
        script_path = run_dir / "script.py"
        script_path.write_text(params.code, encoding="utf-8")

        try:
            start = time.perf_counter()
            result = subprocess.run(
                ["python", str(script_path.resolve())],
                capture_output=True,
                text=True,
                timeout=params.timeout,
                cwd=str(run_dir),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Write full logs to run dir
            (run_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

            # Discover output files produced by the script
            output_files = _finalize_run_dir(
                run_dir=run_dir,
                tool_name=self.name,
                input_files=params.input_files,
                exit_code=result.returncode,
                duration_ms=round(elapsed_ms, 1),
                execution_method="subprocess",
            )

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[-5000:],
                "stderr": result.stderr[-5000:],
                "duration_ms": round(elapsed_ms, 1),
                "run_dir": str(run_dir),
                "output_files": output_files,
            }
        except subprocess.TimeoutExpired:
            (run_dir / "stderr.log").write_text(
                f"Code execution timed out after {params.timeout}s", encoding="utf-8"
            )
            _finalize_run_dir(
                run_dir=run_dir,
                tool_name=self.name,
                input_files=params.input_files,
                exit_code=-1,
                duration_ms=params.timeout * 1000,
                execution_method="subprocess",
                error=f"Code execution timed out after {params.timeout}s",
            )
            return {
                "success": False,
                "error": f"Code execution timed out after {params.timeout}s",
                "run_dir": str(run_dir),
            }


# ── R Execution ──────────────────────────────────────────────────────────


class RunRParams(BaseModel):
    code: str = Field(..., description="R code to execute")
    timeout: int = Field(default=300, description="Execution timeout in seconds")
    use_rpy2: bool = Field(default=True, description="Use rpy2 bridge if available, else Rscript subprocess")
    input_files: list[str] = Field(default_factory=list, description="Input file paths the code depends on")
    output_dir: str = Field(default="", description="Custom run directory; auto-generated if empty")


class RunRTool(CallableTool[RunRParams]):
    name = "run_r"
    description = (
        "Execute R code. Uses rpy2 bridge for tight Python integration when available, "
        "falls back to Rscript subprocess. Supports Bioconductor and CRAN packages. "
        "Returns stdout, stderr, and exit code. "
        "A run directory is created with script, logs, and any output files preserved for inspection."
    )
    params_model = RunRParams

    # Configurable at registration time — set from ExecutionConfig
    r_home: str = ""
    r_lib_path: str = ""
    r_path: str = "Rscript"

    def __init__(
        self,
        runs_base: Path | None = None,
        session_id_provider: Callable[[], str] | None = None,
    ) -> None:
        self.runs_base = runs_base or Path("output") / "runs"
        self._session_id_provider = session_id_provider or (lambda: "default")

    async def execute(self, params: RunRParams, **kwargs: Any) -> dict[str, Any]:
        # Determine run directory (shared by both execution paths)
        if params.output_dir:
            run_dir = Path(params.output_dir)
        else:
            sid = self._session_id_provider()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            run_dir = self.runs_base / sid / f"run_{ts}_{self.name}"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Write script to run dir (preserved, not deleted)
        (run_dir / "script.R").write_text(params.code, encoding="utf-8")

        if params.use_rpy2:
            rpy2_result = await self._execute_via_rpy2(params, run_dir)
            if rpy2_result.get("rpy2_available") and rpy2_result.get("success"):
                return rpy2_result

        # Fallback to Rscript subprocess
        return await self._execute_via_subprocess(params, run_dir)

    async def _execute_via_rpy2(self, params: RunRParams, run_dir: Path) -> dict[str, Any]:
        from passi.executors.r_executor import init_rpy2

        status = init_rpy2(self.r_home, self.r_lib_path)
        if not status["ready"]:
            return {
                "success": False,
                "rpy2_available": False,
                "error": status.get("error", "rpy2 not available"),
                "run_dir": str(run_dir),
            }

        try:
            import rpy2.robjects as ro
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects import numpy2ri, pandas2ri

            start = time.perf_counter()
            with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                ro.r(params.code)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # rpy2 does not natively capture R console output; the execution
            # succeeded so we note that. Detailed output capture requires
            # sink() in the user's R code or Rscript fallback.
            (run_dir / "stdout.log").write_text("", encoding="utf-8")
            (run_dir / "stderr.log").write_text("", encoding="utf-8")

            output_files = _finalize_run_dir(
                run_dir=run_dir,
                tool_name=self.name,
                input_files=params.input_files,
                exit_code=0,
                duration_ms=round(elapsed_ms, 1),
                execution_method="rpy2",
            )

            return {
                "success": True,
                "rpy2_available": True,
                "duration_ms": round(elapsed_ms, 1),
                "execution_method": "rpy2",
                "run_dir": str(run_dir),
                "stdout": "",
                "stderr": "",
                "output_files": output_files,
            }
        except Exception as e:
            (run_dir / "stderr.log").write_text(str(e), encoding="utf-8")
            _finalize_run_dir(
                run_dir=run_dir,
                tool_name=self.name,
                input_files=params.input_files,
                exit_code=1,
                duration_ms=0,
                execution_method="rpy2",
                error=str(e),
            )
            return {
                "success": False,
                "rpy2_available": True,
                "error": str(e),
                "execution_method": "rpy2",
                "run_dir": str(run_dir),
            }

    async def _execute_via_subprocess(self, params: RunRParams, run_dir: Path) -> dict[str, Any]:
        rscript = self.r_path or "Rscript"
        if self.r_home and not os.path.isabs(rscript):
            home = Path(self.r_home)
            for subpath in ("bin/Rscript.exe", "bin/Rscript"):
                exe = home / subpath
                if exe.exists():
                    rscript = str(exe)
                    break

        script_path = run_dir / "script.R"

        try:
            start = time.perf_counter()
            result = subprocess.run(
                [rscript, "--no-save", str(script_path.resolve())],
                capture_output=True,
                text=True,
                timeout=params.timeout,
                cwd=str(run_dir),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Write full logs to run dir
            (run_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

            output_files = _finalize_run_dir(
                run_dir=run_dir,
                tool_name=self.name,
                input_files=params.input_files,
                exit_code=result.returncode,
                duration_ms=round(elapsed_ms, 1),
                execution_method="Rscript",
            )

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[-5000:],
                "stderr": result.stderr[-5000:],
                "duration_ms": round(elapsed_ms, 1),
                "execution_method": "Rscript",
                "run_dir": str(run_dir),
                "output_files": output_files,
            }
        except subprocess.TimeoutExpired:
            (run_dir / "stderr.log").write_text(
                f"R execution timed out after {params.timeout}s", encoding="utf-8"
            )
            _finalize_run_dir(
                run_dir=run_dir,
                tool_name=self.name,
                input_files=params.input_files,
                exit_code=-1,
                duration_ms=params.timeout * 1000,
                execution_method="Rscript",
                error=f"R execution timed out after {params.timeout}s",
            )
            return {
                "success": False,
                "error": f"R execution timed out after {params.timeout}s",
                "run_dir": str(run_dir),
            }
        except FileNotFoundError:
            (run_dir / "stderr.log").write_text(f"Rscript not found: {rscript}", encoding="utf-8")
            _finalize_run_dir(
                run_dir=run_dir,
                tool_name=self.name,
                input_files=params.input_files,
                exit_code=1,
                duration_ms=0,
                execution_method="Rscript",
                error=f"Rscript not found: {rscript}",
            )
            return {
                "success": False,
                "error": f"Rscript not found: {rscript}",
                "run_dir": str(run_dir),
            }
