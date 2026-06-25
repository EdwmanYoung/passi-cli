"""TDD-style unit tests for SurvivalAnalysisTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.tools.clinical_tools import SurvivalAnalysisParams, SurvivalAnalysisTool


class TestSurvivalAnalysisTool:
    """Unit tests for SurvivalAnalysisTool (KM, Cox, competing risks)."""

    @pytest.fixture
    def survival_csv(self, tmp_path: Path) -> Path:
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
    def competing_risk_csv(self, tmp_path: Path) -> Path:
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
