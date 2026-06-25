"""Docker sandbox executor for isolated analysis execution."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Optional Docker-based sandbox for reproducible analysis."""

    def __init__(
        self,
        image: str = "digitagent/bioinfo:latest",
        work_dir: str | None = None,
    ) -> None:
        self.image = image
        self.work_dir = work_dir or str(Path.cwd())
        self._docker_available: bool | None = None

    def is_available(self) -> bool:
        """Check if Docker is available on the system."""
        if self._docker_available is None:
            self._docker_available = shutil.which("docker") is not None
            if not self._docker_available:
                logger.warning("Docker not available, sandbox disabled.")
        return self._docker_available

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 600,
        mounts: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Execute code in a Docker container.

        Args:
            code: Code to execute
            language: 'python' or 'r'
            timeout: Execution timeout in seconds
            mounts: List of (host_path, container_path) bind mounts

        Returns:
            Result dict with stdout, stderr, exit_code
        """
        if not self.is_available():
            return {"success": False, "error": "Docker is not available"}

        cmd = ["docker", "run", "--rm", "--workdir", "/workspace"]

        # Bind mount the current working directory
        cmd.extend(["-v", f"{self.work_dir}:/workspace"])

        # Additional mounts
        if mounts:
            for host_path, container_path in mounts:
                cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Memory and CPU limits
        cmd.extend(["--memory", "16g", "--cpus", "8"])

        # Entry point
        if language == "python":
            cmd.extend([self.image, "python", "-c", code])
        elif language == "r":
            cmd.extend([self.image, "Rscript", "-e", code])
        else:
            return {"success": False, "error": f"Unsupported language: {language}"}

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[-10000:],
                "stderr": result.stderr[-5000:],
                "execution_method": "docker",
                "image": self.image,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Docker execution timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}
