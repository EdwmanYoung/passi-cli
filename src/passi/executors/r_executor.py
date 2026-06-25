"""R code executor — rpy2 bridge with Rscript subprocess fallback.

rpy2 requires R_HOME to be set **before** any ``import rpy2.robjects``.
This module provides ``init_rpy2()`` to configure the environment and
attempt the rpy2 import once; subsequent calls reuse the cached result.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── rpy2 singleton state ──
_rpy2_ready: bool | None = None  # None = not yet tried
_rpy2_status: dict[str, Any] = {}


def _ensure_make_on_path() -> None:
    """Add ``make`` to PATH if not already present.

    rpy2 calls ``R CMD config`` which internally uses ``make`` to resolve
    build flags.  On Windows without RTools, ``make`` is often missing.
    Common install locations are scanned automatically.
    """
    import shutil

    if shutil.which("make"):
        return

    candidates = [
        "C:/Program Files (x86)/GnuWin32/bin",
        "C:/GnuWin32/bin",
        "C:/rtools44/usr/bin",
        "C:/rtools43/usr/bin",
        "C:/rtools42/usr/bin",
        "C:/ProgramData/chocolatey/bin",
    ]
    for d in candidates:
        if Path(d, "make.exe").exists() or Path(d, "make").exists():
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            logger.info("make found at %s", d)
            return


def init_rpy2(r_home: str = "", r_lib_path: str = "") -> dict[str, Any]:
    """Configure R environment and attempt rpy2 import.  Idempotent — only
    runs the import attempt once; returns the cached status on subsequent calls.

    Parameters
    ----------
    r_home:
        Path to R home (e.g. ``D:/project/R``). If empty, uses the
        ``R_HOME`` / ``PASSI_R_HOME`` env vars.
    r_lib_path:
        Custom R library path. Set as ``R_LIBS_USER``.  If empty, uses the
        value already in the environment.

    Returns
    -------
    dict with keys: ``ready`` (bool), ``r_home``, ``r_version``, ``lib_paths``,
    ``error`` (if any).
    """
    global _rpy2_ready, _rpy2_status

    if _rpy2_ready is not None:
        return _rpy2_status

    # ── Resolve R_HOME ──
    if not r_home:
        r_home = os.environ.get("PASSI_R_HOME", os.environ.get("R_HOME", ""))

    if r_home:
        r_home_path = Path(r_home)
        if not r_home_path.exists():
            _rpy2_ready = False
            _rpy2_status = {"ready": False, "r_home": r_home, "error": f"R_HOME does not exist: {r_home}"}
            return _rpy2_status
    else:
        _rpy2_ready = False
        _rpy2_status = {"ready": False, "r_home": "", "error": "R_HOME not set — configure PASSI_EXECUTION__R_HOME"}
        return _rpy2_status

    os.environ["R_HOME"] = str(r_home_path)
    logger.info("R_HOME = %s", r_home_path)

    # Add bin/<arch> to PATH so R.dll is discoverable on Windows
    for arch_dir in (r_home_path / "bin" / "x64", r_home_path / "bin" / "i386", r_home_path / "bin"):
        if arch_dir.exists():
            os.environ["PATH"] = str(arch_dir) + os.pathsep + os.environ.get("PATH", "")

    # Ensure 'make' is available (rpy2 needs R CMD config which calls make)
    # Common install locations on Windows: GnuWin32, RTools, chocolatey
    _ensure_make_on_path()

    # ── R library path ──
    if r_lib_path:
        lib = Path(r_lib_path)
        lib.mkdir(parents=True, exist_ok=True)
        os.environ["R_LIBS_USER"] = str(lib.resolve())

    # ── Attempt rpy2 import ──
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri  # noqa: F401

        r_ver = ro.r("R.version$version.string")[0]
        lib_paths = list(ro.r(".libPaths()"))

        _rpy2_ready = True
        _rpy2_status = {
            "ready": True,
            "r_home": str(r_home_path),
            "r_version": r_ver,
            "lib_paths": lib_paths,
            "r_libs_user": os.environ.get("R_LIBS_USER", ""),
        }
        logger.info("rpy2 ready — %s", r_ver)
        return _rpy2_status

    except ImportError as e:
        _rpy2_ready = False
        _rpy2_status = {"ready": False, "r_home": str(r_home_path), "error": f"rpy2 not installed: {e}"}
        return _rpy2_status
    except Exception as e:
        _rpy2_ready = False
        _rpy2_status = {"ready": False, "r_home": str(r_home_path), "error": str(e)}
        logger.warning("rpy2 init failed: %s", e)
        return _rpy2_status


def get_r_status() -> dict[str, Any]:
    """Return the cached R environment status (does not trigger init)."""
    if _rpy2_ready is None:
        return {"ready": False, "r_home": "", "error": "R environment not yet initialized"}
    return _rpy2_status


class RExecutor:
    """Execute R code via rpy2 (primary) or Rscript subprocess (fallback)."""

    def __init__(
        self,
        r_path: str = "Rscript",
        r_home: str = "",
        r_lib_path: str = "",
        use_rpy2: bool = True,
        timeout: int = 300,
    ) -> None:
        self.r_path = r_path
        self.r_home = r_home
        self.r_lib_path = r_lib_path
        self.use_rpy2 = use_rpy2
        self.timeout = timeout

    def execute(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute R code.  Tries rpy2 first, falls back to Rscript."""
        if self.use_rpy2:
            status = init_rpy2(self.r_home, self.r_lib_path)
            if status["ready"]:
                result = self._execute_rpy2(code)
                if result.get("success"):
                    return result

        return self._execute_subprocess(code, timeout)

    def _execute_rpy2(self, code: str) -> dict[str, Any]:
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects import numpy2ri, pandas2ri

            start = time.perf_counter()
            with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                ro.r(code)
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

    def _execute_subprocess(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        rscript = self.r_path or "Rscript"
        # Resolve from r_home if it's a bare name
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
            f.write(code)
            script_path = f.name

        try:
            start = time.perf_counter()
            result = subprocess.run(
                [rscript, "--no-save", script_path],
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
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
            return {"success": False, "error": "R execution timed out"}
        except FileNotFoundError:
            return {"success": False, "error": f"Rscript not found: {rscript}"}
        finally:
            Path(script_path).unlink(missing_ok=True)
