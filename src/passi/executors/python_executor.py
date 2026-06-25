"""Python code executor with sandboxing."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PythonExecutor:
    """Execute Python code in a subprocess sandbox."""

    def __init__(self, python_path: str = "python", timeout: int = 300) -> None:
        self.python_path = python_path
        self.timeout = timeout

    def execute(
        self,
        code: str,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute Python code and return stdout, stderr, and exit code."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            start = time.perf_counter()
            result = subprocess.run(
                [self.python_path, "-u", script_path],
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                cwd=str(Path.cwd()),
                env=env,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": round(elapsed_ms, 1),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout or self.timeout}s",
            }
        finally:
            Path(script_path).unlink(missing_ok=True)
