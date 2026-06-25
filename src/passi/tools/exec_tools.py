"""Code execution tools — run Python and R code in sandboxed environments."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from passi.tools.base import CallableTool


class RunPythonParams(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=120, description="Execution timeout in seconds")
    packages: list[str] | None = Field(default=None, description="Extra packages to ensure are installed")


class RunPythonTool(CallableTool[RunPythonParams]):
    name = "run_python"
    description = (
        "Execute Python code in a sandboxed subprocess. Use for data analysis, "
        "statistical computations, and visualization. Returns stdout, stderr, and exit code."
    )
    params_model = RunPythonParams

    async def execute(self, params: RunPythonParams, **kwargs: Any) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(params.code)
            script_path = f.name

        try:
            start = time.perf_counter()
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=params.timeout,
                cwd=str(Path.cwd()),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[-5000:],  # truncate
                "stderr": result.stderr[-5000:],
                "duration_ms": round(elapsed_ms, 1),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Code execution timed out after {params.timeout}s",
            }
        finally:
            Path(script_path).unlink(missing_ok=True)


class RunRParams(BaseModel):
    code: str = Field(..., description="R code to execute")
    timeout: int = Field(default=300, description="Execution timeout in seconds")
    use_rpy2: bool = Field(default=True, description="Use rpy2 bridge if available, else Rscript subprocess")


class RunRTool(CallableTool[RunRParams]):
    name = "run_r"
    description = (
        "Execute R code. Uses rpy2 bridge for tight Python integration when available, "
        "falls back to Rscript subprocess. Supports Bioconductor and CRAN packages. "
        "Returns stdout, stderr, and exit code."
    )
    params_model = RunRParams

    # Configurable at registration time — set from ExecutionConfig
    r_home: str = ""
    r_lib_path: str = ""
    r_path: str = "Rscript"

    async def execute(self, params: RunRParams, **kwargs: Any) -> dict[str, Any]:
        if params.use_rpy2:
            rpy2_result = await self._execute_via_rpy2(params)
            if rpy2_result.get("rpy2_available") and rpy2_result.get("success"):
                return rpy2_result

        # Fallback to Rscript subprocess
        return await self._execute_via_subprocess(params)

    async def _execute_via_rpy2(self, params: RunRParams) -> dict[str, Any]:
        from passi.executors.r_executor import init_rpy2

        status = init_rpy2(self.r_home, self.r_lib_path)
        if not status["ready"]:
            return {"success": False, "rpy2_available": False, "error": status.get("error", "rpy2 not available")}

        try:
            import rpy2.robjects as ro
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects import numpy2ri, pandas2ri

            start = time.perf_counter()
            with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                ro.r(params.code)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return {
                "success": True,
                "rpy2_available": True,
                "duration_ms": round(elapsed_ms, 1),
                "execution_method": "rpy2",
            }
        except Exception as e:
            return {
                "success": False,
                "rpy2_available": True,
                "error": str(e),
                "execution_method": "rpy2",
            }

    async def _execute_via_subprocess(self, params: RunRParams) -> dict[str, Any]:
        rscript = self.r_path or "Rscript"
        # If r_home is configured, prefer Rscript from there
        if self.r_home and not os.path.isabs(rscript):
            home = Path(self.r_home)
            for subpath in ("bin/Rscript.exe", "bin/Rscript"):
                exe = home / subpath
                if exe.exists():
                    rscript = str(exe)
                    break

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".R", delete=False, encoding="utf-8"
        ) as f:
            f.write(params.code)
            script_path = f.name

        try:
            start = time.perf_counter()
            result = subprocess.run(
                [rscript, "--no-save", script_path],
                capture_output=True,
                text=True,
                timeout=params.timeout,
                cwd=str(Path.cwd()),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[-5000:],
                "stderr": result.stderr[-5000:],
                "duration_ms": round(elapsed_ms, 1),
                "execution_method": "Rscript",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"R execution timed out after {params.timeout}s"}
        except FileNotFoundError:
            return {"success": False, "error": f"Rscript not found: {rscript}"}
        finally:
            Path(script_path).unlink(missing_ok=True)
