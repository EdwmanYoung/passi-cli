"""TDD-style unit tests for QcReportTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.tools.qc_tools import QcReportParams, QcReportTool


class TestQcReportTool:
    """Unit tests for QcReportTool — missing values, PCA, recommendations."""

    @pytest.fixture
    def count_csv(self, tmp_path: Path) -> Path:
        """Create a synthetic count matrix with some missing values."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "gene": [f"GENE{i}" for i in range(100)],
                "sample_A": [100] * 50 + [50] * 50,
                "sample_B": [90] * 50 + [None] * 50,  # 50% missing in B
                "sample_C": [80] * 100,
                "sample_D": [70] * 100,
            }
        )
        path = tmp_path / "test_counts.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.mark.asyncio
    async def test_qc_detects_missing_values(self, count_csv: Path):
        # Arrange
        tool = QcReportTool()
        params = QcReportParams(data_path=str(count_csv))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["metrics"]["missing"]["total"] == 50  # 50 NAs in sample_B
        assert result["metrics"]["missing"]["percent"] > 0

    @pytest.mark.asyncio
    async def test_qc_reports_shape(self, count_csv: Path):
        # Arrange
        tool = QcReportTool()
        params = QcReportParams(data_path=str(count_csv))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["metrics"]["shape"]["rows"] == 100
        assert result["metrics"]["shape"]["columns"] == 5  # gene + 4 samples

    @pytest.mark.asyncio
    async def test_qc_nonexistent_file_returns_error(self, tmp_path: Path):
        # Arrange
        tool = QcReportTool()
        params = QcReportParams(data_path=str(tmp_path / "ghost.csv"))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_qc_generates_recommendations_for_high_missing(self, tmp_path: Path):
        # Arrange — high missing rate
        import pandas as pd

        df = pd.DataFrame(
            {
                "gene": [f"G{i}" for i in range(20)],
                "S1": [1] * 10 + [None] * 10,
                "S2": [2] * 5 + [None] * 15,
            }
        )
        path = tmp_path / "high_missing.csv"
        df.to_csv(path, index=False)

        tool = QcReportTool()
        params = QcReportParams(data_path=str(path))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert any("missing" in r.lower() for r in result["recommendations"])

    @pytest.mark.asyncio
    async def test_qc_includes_group_info(self, count_csv: Path, tmp_path: Path):
        # Arrange
        import pandas as pd

        # Add a group column
        df = pd.read_csv(count_csv)
        df["group"] = ["control"] * 50 + ["treatment"] * 50
        path = tmp_path / "counts_with_group.csv"
        df.to_csv(path, index=False)

        tool = QcReportTool()
        params = QcReportParams(data_path=str(path), group_col="group")

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert "groups" in result["metrics"]
        assert "control" in result["metrics"]["groups"]
