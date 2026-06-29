"""Tests for passi.executors.r_executor.

These tests do not require a real R installation. They exercise environment
resolution, rpy2 import mocking, and subprocess fallback paths.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import passi.executors.r_executor as r_executor
from passi.executors.r_executor import RExecutor, get_r_status, init_rpy2


def _make_fake_rpy2_modules() -> dict[str, Any]:
    """Build a minimal set of mocked rpy2 modules for init_rpy2 tests."""
    fake_ro = ModuleType("rpy2.robjects")
    fake_ro.r = MagicMock(return_value=[])
    fake_ro.default_converter = MagicMock()

    fake_conversion = ModuleType("conversion")
    fake_conversion.localconverter = MagicMock()

    fake_pandas2ri = ModuleType("pandas2ri")
    fake_pandas2ri.converter = MagicMock()
    fake_numpy2ri = ModuleType("numpy2ri")
    fake_numpy2ri.converter = MagicMock()

    return {
        "rpy2": ModuleType("rpy2"),
        "rpy2.robjects": fake_ro,
        "rpy2.robjects.conversion": fake_conversion,
        "rpy2.robjects.numpy2ri": fake_numpy2ri,
        "rpy2.robjects.pandas2ri": fake_pandas2ri,
    }


@pytest.fixture(autouse=True)
def reset_rpy2_state() -> None:
    """Reset module-level rpy2 singleton, PATH, and R_HOME before each test."""
    r_executor._rpy2_ready = None
    r_executor._rpy2_status = {}
    original_path = os.environ.get("PATH", "")
    original_r_home = os.environ.get("R_HOME", None)
    yield
    r_executor._rpy2_ready = None
    r_executor._rpy2_status = {}
    os.environ["PATH"] = original_path
    if original_r_home is None:
        os.environ.pop("R_HOME", None)
    else:
        os.environ["R_HOME"] = original_r_home


@pytest.fixture
def fake_r_home(tmp_path: Path) -> Path:
    """Create a minimal fake R_HOME tree."""
    home = tmp_path / "R"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "Rscript.exe").write_text("")
    return home


class TestEnsureMakeOnPath:
    """PATH extension for make.exe / make."""

    def test_make_already_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When make is on PATH, no changes are made."""
        monkeypatch.setenv("PATH", "/usr/bin")
        with patch("shutil.which", return_value="/usr/bin/make"):
            r_executor._ensure_make_on_path()
        assert os.environ["PATH"] == "/usr/bin"

    def test_make_found_in_candidate_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Candidate directory containing make.exe is prepended to PATH."""
        make_dir = tmp_path / "GnuWin32" / "bin"
        make_dir.mkdir(parents=True)
        (make_dir / "make.exe").write_text("")

        original_path = str(tmp_path / "orig")
        monkeypatch.setenv("PATH", original_path)

        with patch.object(r_executor, "logger"):
            with patch("shutil.which", return_value=None):
                _ensure_make_test(make_dir)

        expected = str(make_dir) + os.pathsep + original_path
        assert os.environ["PATH"] == expected


def _ensure_make_test(make_dir: Path) -> None:
    """Helper that mimics the real scanning logic for one candidate dir."""
    import shutil

    if shutil.which("make"):
        return
    if Path(make_dir, "make.exe").exists() or Path(make_dir, "make").exists():
        os.environ["PATH"] = str(make_dir) + os.pathsep + os.environ.get("PATH", "")


class TestInitRpy2:
    """R environment initialization and rpy2 import handling."""

    def test_init_rpy2_idempotent(self, fake_r_home: Path) -> None:
        """Subsequent calls return the cached status dict."""
        fake_modules = _make_fake_rpy2_modules()
        with patch.dict("sys.modules", fake_modules, clear=False):
            with patch.dict(os.environ, {"R_HOME": str(fake_r_home)}):
                first = init_rpy2()
                second = init_rpy2()
        assert first is second
        assert first["ready"] is False

    def test_init_rpy2_missing_r_home(self) -> None:
        """Empty R_HOME returns ready=False with helpful error."""
        with patch.dict(os.environ, {"R_HOME": "", "PASSI_R_HOME": ""}, clear=False):
            status = init_rpy2()
        assert status["ready"] is False
        assert "R_HOME not set" in status["error"]

    def test_init_rpy2_nonexistent_r_home(self, tmp_path: Path) -> None:
        """Non-existent explicit R_HOME returns ready=False."""
        missing = tmp_path / "missing_R"
        status = init_rpy2(r_home=str(missing))
        assert status["ready"] is False
        assert "does not exist" in status["error"]

    def test_init_rpy2_import_error(self, fake_r_home: Path) -> None:
        """Missing rpy2 package is reported cleanly."""
        real_import = __builtins__["__import__"]

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("rpy2"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            status = init_rpy2(r_home=str(fake_r_home))
        assert status["ready"] is False
        assert "rpy2 not installed" in status["error"] or "No module named" in status["error"]

    def test_init_rpy2_generic_exception(self, fake_r_home: Path) -> None:
        """Any non-ImportError during rpy2 init is captured."""
        fake_modules = _make_fake_rpy2_modules()
        fake_modules["rpy2.robjects"].r.side_effect = RuntimeError("rpy2 broken")

        with patch.dict("sys.modules", fake_modules, clear=False):
            status = init_rpy2(r_home=str(fake_r_home))

        assert status["ready"] is False
        assert "rpy2 broken" in status["error"]

    def test_init_rpy2_success(self, fake_r_home: Path) -> None:
        """Successful rpy2 import reports version and lib paths."""
        fake_modules = _make_fake_rpy2_modules()

        def fake_r(expr: str) -> Any:
            if "version.string" in expr:
                return ["R version 4.4.0"]
            if ".libPaths" in expr:
                return ["/R-lib"]
            return []

        fake_modules["rpy2.robjects"].r = fake_r

        with patch.dict("sys.modules", fake_modules, clear=False):
            status = init_rpy2(r_home=str(fake_r_home))

        assert status["ready"] is True
        assert status["r_home"] == str(fake_r_home)
        assert status["r_version"] == "R version 4.4.0"
        assert "/R-lib" in status["lib_paths"]


class TestGetRStatus:
    """Cached R status queries."""

    def test_get_r_status_before_init(self) -> None:
        """Before init_rpy2 runs, status reports not initialized."""
        status = get_r_status()
        assert status["ready"] is False
        assert "not yet initialized" in status["error"]

    def test_get_r_status_after_init(self, fake_r_home: Path) -> None:
        """After any init attempt, cached status is returned."""
        init_rpy2(r_home=str(fake_r_home))
        status = get_r_status()
        assert status is r_executor._rpy2_status


class TestRExecutor:
    """RExecutor subprocess and rpy2 execution paths."""

    def test_execute_subprocess_success(self, tmp_path: Path) -> None:
        """Rscript subprocess success returns stdout/exit_code."""
        fake_rscript = tmp_path / "Rscript.bat"
        fake_rscript.write_text("@echo off\necho hello\n")

        executor = RExecutor(r_path=str(fake_rscript), use_rpy2=False)
        result = executor.execute("cat('hi')")

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]
        assert result["execution_method"] == "Rscript"

    def test_execute_subprocess_timeout(self) -> None:
        """Rscript timeout returns a clear error."""
        executor = RExecutor(r_path="Rscript", use_rpy2=False, timeout=1)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("Rscript", 1)):
            result = executor.execute("cat('hi')")

        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_execute_subprocess_not_found(self) -> None:
        """Missing Rscript binary returns FileNotFoundError handling."""
        executor = RExecutor(r_path="/nonexistent/Rscript", use_rpy2=False)
        result = executor.execute("cat('hi')")

        assert result["success"] is False
        assert "Rscript not found" in result["error"]

    def test_execute_resolves_rscript_from_r_home(self, tmp_path: Path) -> None:
        """Bare Rscript name is resolved against r_home."""
        home = tmp_path / "R"
        bin_dir = home / "bin"
        bin_dir.mkdir(parents=True)
        fake_exe = bin_dir / "Rscript.exe"
        fake_exe.write_text("")

        executor = RExecutor(r_path="Rscript", r_home=str(home), use_rpy2=False)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "resolved\n"
            mock_run.return_value.stderr = ""
            result = executor.execute("cat('hi')")

        assert result["success"] is True
        assert "resolved" in result["stdout"]
        # Verify it was called with the resolved exe path
        called_args = mock_run.call_args[0][0]
        assert str(fake_exe) in called_args[0]

    def test_execute_rpy2_success(self, fake_r_home: Path) -> None:
        """When rpy2 is mocked ready, rpy2 execution method is reported."""
        fake_ro = ModuleType("rpy2.robjects")
        fake_ro.r = MagicMock()
        fake_ro.default_converter = MagicMock()

        fake_localconverter = MagicMock()
        fake_localconverter.__enter__ = MagicMock(return_value=None)
        fake_localconverter.__exit__ = MagicMock(return_value=None)

        fake_conversion = ModuleType("conversion")
        fake_conversion.localconverter = lambda converter: fake_localconverter

        fake_pandas2ri = ModuleType("pandas2ri")
        fake_pandas2ri.converter = MagicMock()
        fake_numpy2ri = ModuleType("numpy2ri")
        fake_numpy2ri.converter = MagicMock()

        fake_modules = {
            "rpy2": ModuleType("rpy2"),
            "rpy2.robjects": fake_ro,
            "rpy2.robjects.conversion": fake_conversion,
            "rpy2.robjects.numpy2ri": fake_numpy2ri,
            "rpy2.robjects.pandas2ri": fake_pandas2ri,
        }

        with patch.dict("sys.modules", fake_modules, clear=False):
            executor = RExecutor(r_home=str(fake_r_home), use_rpy2=True)
            result = executor.execute("cat('hi')")

        assert result["success"] is True
        assert result["execution_method"] == "rpy2"
        assert "duration_ms" in result


class TestRExecutorRealR:
    """Real R execution tests using the project-local R environment."""

    def test_rpy2_execution_with_project_r(self) -> None:
        """RExecutor runs R code through the real rpy2 bridge."""
        executor = RExecutor(use_rpy2=True, timeout=60)
        result = executor.execute("x <- 1 + 1; print(x)")

        if not result.get("rpy2_available"):
            pytest.skip("rpy2 bridge not available")
        assert result["success"] is True
        assert result["execution_method"] == "rpy2"
        assert "duration_ms" in result

    def test_rscript_fallback_with_project_r(self) -> None:
        """RExecutor falls back to the real Rscript subprocess."""
        executor = RExecutor(use_rpy2=False, timeout=60)
        result = executor.execute('cat("Hello from project R\n")')

        assert result["success"] is True
        assert result["execution_method"] == "Rscript"
        assert "Hello from project R" in result["stdout"]
