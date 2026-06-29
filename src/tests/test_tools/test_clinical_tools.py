"""TDD-style unit tests for SurvivalAnalysisTool."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from passi.tools.clinical_tools import SurvivalAnalysisParams, SurvivalAnalysisTool


@pytest.fixture
def survival_csv(tmp_path: Path) -> Path:
    """Create synthetic survival data."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "OS.time": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
            "OS.event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "group": (["treatment"] * 5) + (["control"] * 5),
            "age": [55, 62, 48, 71, 59, 54, 67, 45, 73, 58],
            "stage": [1, 2, 1, 3, 2, 1, 3, 2, 3, 1],
        }
    )
    path = tmp_path / "survival_data.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def competing_risk_csv(tmp_path: Path) -> Path:
    """Create synthetic competing risks data."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "time": [5, 10, 15, 20, 25, 30, 35, 40],
            "status": [0, 1, 2, 1, 0, 2, 1, 0],
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    path = tmp_path / "cr_data.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


class TestSurvivalAnalysisTool:
    """Unit tests for SurvivalAnalysisTool (KM, Cox, competing risks)."""

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self, survival_csv):
        tool = SurvivalAnalysisTool()
        params = SurvivalAnalysisParams(
            data_path=str(survival_csv),
            time_col="OS.time",
            event_col="OS.event",
            method="bad_method",
        )

        result = await tool.execute(params)
        assert result["success"] is False
        assert "Unknown method" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self):
        tool = SurvivalAnalysisTool()
        params = SurvivalAnalysisParams(
            data_path="/nonexistent/data.tsv",
            time_col="OS.time",
            event_col="OS.event",
        )

        result = await tool.execute(params)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_params_validation(self):
        params = SurvivalAnalysisParams(
            data_path="/data/survival.tsv",
            time_col="OS.time",
            event_col="OS.event",
            method="cox",
            group_col="treatment",
            covariates="age, stage",
        )
        assert params.method == "cox"
        assert params.covariates == "age, stage"

    @pytest.mark.asyncio
    async def test_km_script_template(self):
        """KM script renders correctly."""
        from passi.tools.clinical_tools import KM_SURVFIT_SCRIPT

        script = KM_SURVFIT_SCRIPT.format(
            data_path="/data/surv.tsv",
            time_col="OS.time",
            event_col="OS.event",
            group_col="group",
            output_json="/tmp/km.json",
        )
        assert "Surv(" in script
        assert "survfit" in script
        assert "survdiff" in script
        assert "/data/surv.tsv" in script
        assert "group" in script

    @pytest.mark.asyncio
    async def test_cox_script_template(self):
        """Cox script renders with covariates."""
        from passi.tools.clinical_tools import COXPH_SCRIPT

        script = COXPH_SCRIPT.format(
            data_path="/data/surv.tsv",
            time_col="time",
            event_col="status",
            group_col="group",
            covariates="age, stage, group",
            output_json="/tmp/cox.json",
        )
        assert "coxph" in script
        assert "Surv(" in script
        assert "age" in script
        assert "stage" in script
        assert "cox.zph" in script

    @pytest.mark.asyncio
    async def test_competing_risks_script_template(self):
        """Competing risks script uses mstate."""
        from passi.tools.clinical_tools import COMPETING_RISK_SCRIPT

        script = COMPETING_RISK_SCRIPT.format(
            data_path="/data/cr.tsv",
            time_col="time",
            event_col="status",
            output_json="/tmp/cr.json",
        )
        assert "mstate" in script
        assert "survfit" in script


class TestSurvivalAnalysisExecution:
    """Mocked execution paths for rpy2 and Rscript fallbacks."""

    @pytest.fixture
    def survival_csv(self, tmp_path: Path) -> Path:
        import pandas as pd

        df = pd.DataFrame(
            {
                "OS.time": [12, 24, 36, 48, 60],
                "OS.event": [1, 0, 1, 0, 1],
                "group": ["A", "A", "B", "B", "B"],
            }
        )
        path = tmp_path / "survival.tsv"
        df.to_csv(path, sep="\t", index=False)
        return path

    def _make_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _fake_rpy2_modules(self) -> dict[str, Any]:
        return {
            "rpy2": ModuleType("rpy2"),
            "rpy2.robjects": ModuleType("rpy2.robjects"),
            "rpy2.robjects.conversion": ModuleType("conversion"),
            "rpy2.robjects.numpy2ri": ModuleType("numpy2ri"),
            "rpy2.robjects.pandas2ri": ModuleType("pandas2ri"),
        }

    @pytest.mark.asyncio
    async def test_km_rpy2_path(self, survival_csv: Path, tmp_path: Path) -> None:
        tool = SurvivalAnalysisTool()
        output_json = tmp_path / "survival_km_result.json"
        self._make_json(output_json, {"method": "km", "n": 5, "groups": 2, "logrank_p": 0.12})

        fake_modules = self._fake_rpy2_modules()
        fake_modules["rpy2.robjects"].r = MagicMock()
        fake_modules["rpy2.robjects"].default_converter = MagicMock()
        fake_modules["rpy2.robjects.conversion"].localconverter = lambda c: MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
        fake_modules["rpy2.robjects.numpy2ri"].converter = MagicMock()
        fake_modules["rpy2.robjects.pandas2ri"].converter = MagicMock()

        with patch("passi.executors.r_executor.init_rpy2", return_value={"ready": True}):
            with patch.dict("sys.modules", fake_modules, clear=False):
                params = SurvivalAnalysisParams(
                    data_path=str(survival_csv),
                    time_col="OS.time",
                    event_col="OS.event",
                    group_col="group",
                    method="km",
                    output_dir=str(tmp_path),
                )
                result = await tool.execute(params)

        assert result["success"] is True
        assert result["method"] == "km"
        assert result["n"] == 5

    @pytest.mark.asyncio
    async def test_cox_rscript_fallback(self, survival_csv: Path, tmp_path: Path) -> None:
        tool = SurvivalAnalysisTool()
        output_json = tmp_path / "survival_cox_result.json"
        self._make_json(output_json, {"method": "cox", "n": 5, "events": 3, "concordance": 0.75})

        class FakeResult:
            returncode = 0
            stdout = "COXPH_DONE|n=5|events=3|concordance=0.7500|ph_p=0.1200\n"
            stderr = ""

        with patch("passi.executors.r_executor.init_rpy2", return_value={"ready": False}):
            with patch("subprocess.run", return_value=FakeResult()):
                params = SurvivalAnalysisParams(
                    data_path=str(survival_csv),
                    time_col="OS.time",
                    event_col="OS.event",
                    covariates="group",
                    method="cox",
                    output_dir=str(tmp_path),
                )
                result = await tool.execute(params)

        assert result["success"] is True
        assert result["method"] == "cox"
        assert result["concordance"] == 0.75

    @pytest.mark.asyncio
    async def test_competing_risks_rscript(self, competing_risk_csv: Path, tmp_path: Path) -> None:
        tool = SurvivalAnalysisTool()
        output_json = tmp_path / "survival_competing_risks_result.json"
        self._make_json(
            output_json,
            {"method": "competing_risks", "n_total": 8, "events_primary": 3, "events_competing": 2},
        )

        class FakeResult:
            returncode = 0
            stdout = "CR_DONE|n=8|events_type1=3|events_type2=2\n"
            stderr = ""

        with patch("passi.executors.r_executor.init_rpy2", return_value={"ready": False}):
            with patch("subprocess.run", return_value=FakeResult()):
                params = SurvivalAnalysisParams(
                    data_path=str(competing_risk_csv),
                    time_col="time",
                    event_col="status",
                    method="competing_risks",
                    output_dir=str(tmp_path),
                )
                result = await tool.execute(params)

        assert result["success"] is True
        assert result["events_primary"] == 3
        assert result["events_competing"] == 2


class TestSurvivalAnalysisRealR:
    """Real Rscript execution tests using the project-local R environment."""

    @pytest.fixture
    def survival_data(self, tmp_path: Path) -> Path:
        """Create a small survival dataset with treatment/control groups."""
        import pandas as pd

        df = pd.DataFrame({
            "time": [12, 18, 24, 30, 36, 15, 20, 28, 40, 50, 60, 72],
            "event": [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "group": (["treatment"] * 6) + (["control"] * 6),
            "age": [55, 62, 48, 71, 59, 54, 67, 45, 73, 58, 65, 52],
        })
        path = tmp_path / "survival.tsv"
        df.to_csv(path, sep="\t", index=False)
        return path

    @pytest.fixture
    def competing_risk_data(self, tmp_path: Path) -> Path:
        """Create a small competing risks dataset."""
        import pandas as pd

        df = pd.DataFrame({
            "time": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "status": [0, 1, 2, 1, 0, 2, 1, 0, 1, 2],
            "group": (["A"] * 5) + (["B"] * 5),
        })
        path = tmp_path / "competing.tsv"
        df.to_csv(path, sep="\t", index=False)
        return path

    @pytest.mark.asyncio
    async def test_km_real_r(self, survival_data: Path, tmp_path: Path) -> None:
        tool = SurvivalAnalysisTool()
        params = SurvivalAnalysisParams(
            data_path=str(survival_data),
            time_col="time",
            event_col="event",
            group_col="group",
            method="km",
            output_dir=str(tmp_path / "result"),
        )
        result = await tool.execute(params)

        assert result["success"] is True, result.get("error", "")
        assert result["method"] == "km"
        assert result["groups"] == 2
        assert "logrank_p" in result
        assert "details" in result

    @pytest.mark.asyncio
    async def test_cox_real_r(self, survival_data: Path, tmp_path: Path) -> None:
        tool = SurvivalAnalysisTool()
        params = SurvivalAnalysisParams(
            data_path=str(survival_data),
            time_col="time",
            event_col="event",
            covariates="group, age",
            method="cox",
            output_dir=str(tmp_path / "result"),
        )
        result = await tool.execute(params)

        assert result["success"] is True, result.get("error", "")
        assert result["method"] == "cox"
        assert "concordance" in result
        assert result["n"] == 12
        assert "details" in result

    @pytest.mark.asyncio
    async def test_competing_risks_real_r(self, competing_risk_data: Path, tmp_path: Path) -> None:
        tool = SurvivalAnalysisTool()
        params = SurvivalAnalysisParams(
            data_path=str(competing_risk_data),
            time_col="time",
            event_col="status",
            method="competing_risks",
            output_dir=str(tmp_path / "result"),
        )
        result = await tool.execute(params)

        assert result["success"] is True, result.get("error", "")
        assert result["method"] == "competing_risks"
        assert result["n_total"] == 10
        assert "events_primary" in result
        assert "events_competing" in result
        assert "details" in result


class TestSurvivalAnalysisParseOutput:
    """Direct tests for SurvivalAnalysisTool._parse_output."""

    @pytest.fixture
    def tool(self) -> SurvivalAnalysisTool:
        return SurvivalAnalysisTool()

    def test_km_done_line_parsed(self, tool: SurvivalAnalysisTool, tmp_path: Path) -> None:
        result = tool._parse_output(
            "km",
            tmp_path / "missing.json",
            tmp_path / "data.tsv",
            stdout="KM_DONE|groups=2|logrank_p=0.034000\n",
        )
        assert result["success"] is True
        assert result["groups"] == 2
        assert result["logrank_p"] == 0.034

    def test_cox_ph_warning(self, tool: SurvivalAnalysisTool, tmp_path: Path) -> None:
        result = tool._parse_output(
            "cox",
            tmp_path / "missing.json",
            tmp_path / "data.tsv",
            stdout="COXPH_DONE|n=100|events=40|concordance=0.6500|ph_p=0.0100\n",
        )
        assert result["ph_warning"] == "Proportional hazards assumption may be violated (p < 0.05)"

    def test_json_details_promoted(self, tool: SurvivalAnalysisTool, tmp_path: Path) -> None:
        json_path = tmp_path / "survival_km_result.json"
        json_path.write_text(
            json.dumps({"n": 50, "events": 20, "logrank_p": 0.04, "groups": 2}),
            encoding="utf-8",
        )
        result = tool._parse_output("km", json_path, tmp_path / "data.tsv")
        assert result["n"] == 50
        assert result["events"] == 20
        assert "details" in result

    def test_warning_when_no_output(self, tool: SurvivalAnalysisTool, tmp_path: Path) -> None:
        result = tool._parse_output("km", tmp_path / "missing.json", tmp_path / "data.tsv")
        assert result["success"] is True
        assert "warning" in result
        assert "No output data produced" in result["warning"]
