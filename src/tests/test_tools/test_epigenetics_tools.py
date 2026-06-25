"""TDD-style unit tests for epigenetics tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from passi.tools.epigenetics_tools import (
    MethylationAnalysisParams,
    MethylationAnalysisTool,
    PeakQcParams,
    PeakQcTool,
)


class TestPeakQcTool:
    """Unit tests for PeakQcTool."""

    @pytest.fixture
    def narrowpeak_file(self, tmp_path: Path) -> Path:
        """Create a minimal narrowPeak file."""
        np.random.seed(42)
        lines = []
        chroms = ["chr1", "chr2", "chr3", "chr4"]
        for i in range(500):
            chrom = chroms[i % 4]
            start = np.random.randint(1, 10000000)
            width = np.random.choice([150, 200, 300, 500, 800, 1200])
            end = start + width
            name = f"peak_{i}"
            score = int(np.random.randint(0, 1000))
            signal = np.random.exponential(100)
            pval = np.random.uniform(1e-10, 1e-2)
            qval = pval * np.random.uniform(1, 5)
            lines.append(
                f"{chrom}\t{start}\t{end}\t{name}\t{score}\t.\t{signal:.2f}\t"
                f"{-np.log10(pval):.2f}\t{-np.log10(qval):.2f}\t{width}"
            )
        path = tmp_path / "peaks.narrowPeak"
        path.write_text("\n".join(lines))
        return path

    @pytest.mark.asyncio
    async def test_peak_qc_with_narrowpeak(self, narrowpeak_file):
        tool = PeakQcTool()
        params = PeakQcParams(peak_path=str(narrowpeak_file))

        result = await tool.execute(params)
        assert result["success"] is True
        stats = result["stats"]
        assert stats["total_peaks"] == 500
        assert "width" in stats
        assert "signal" in stats
        assert stats["width"]["median"] > 0
        assert "significant_peaks" in stats.get("q_value", {})

    @pytest.mark.asyncio
    async def test_peak_qc_missing_file_returns_error(self):
        tool = PeakQcTool()
        params = PeakQcParams(peak_path="/nonexistent/peaks.narrowPeak")

        result = await tool.execute(params)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_peak_qc_generates_recommendations(self, narrowpeak_file):
        tool = PeakQcTool()
        params = PeakQcParams(peak_path=str(narrowpeak_file))

        result = await tool.execute(params)
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_peak_qc_params_validation(self):
        params = PeakQcParams(peak_path="/data/peaks.narrowPeak", align_path="/data/align.bam")
        assert params.peak_path == "/data/peaks.narrowPeak"
        assert params.align_path == "/data/align.bam"


class TestMethylationAnalysisTool:
    """Unit tests for MethylationAnalysisTool."""

    @pytest.fixture
    def beta_matrix(self, tmp_path: Path) -> Path:
        """Create a synthetic beta-value matrix."""
        import pandas as pd

        np.random.seed(42)
        n_cpgs = 200
        n_samples = 6

        # Generate realistic beta values (bimodal: low and high methylation)
        beta_data = {}
        for i in range(n_samples):
            # Mix of low, medium, high methylation CpGs
            low = np.random.beta(2, 20, n_cpgs // 3)
            mid = np.random.beta(10, 10, n_cpgs // 3)
            high = np.random.beta(20, 2, n_cpgs - 2 * (n_cpgs // 3))
            beta_data[f"sample_{i}"] = np.concatenate([low, mid, high])

        df = pd.DataFrame(beta_data)
        df.index = [f"cg{idx:06d}" for idx in range(n_cpgs)]
        df.index.name = "cpg_id"

        path = tmp_path / "beta_matrix.tsv"
        df.to_csv(path, sep="\t")
        return path

    @pytest.fixture
    def beta_with_metadata(self, tmp_path: Path) -> tuple[Path, Path]:
        """Beta matrix + metadata for group comparison."""
        import pandas as pd

        n_cpgs = 100
        n_samples = 8

        beta_data = {}
        for i in range(n_samples):
            beta_data[f"S{i}"] = np.random.beta(2, 5, n_cpgs)

        df = pd.DataFrame(beta_data)
        df.index = [f"cg{idx:06d}" for idx in range(n_cpgs)]
        df.index.name = "cpg_id"

        beta_path = tmp_path / "beta.tsv"
        df.to_csv(beta_path, sep="\t")

        meta = pd.DataFrame(
            {"group": (["cancer"] * 4) + (["normal"] * 4)},
            index=[f"S{i}" for i in range(n_samples)],
        )
        meta.index.name = "sample_id"
        meta_path = tmp_path / "meta.tsv"
        meta.to_csv(meta_path, sep="\t")

        return beta_path, meta_path

    @pytest.mark.asyncio
    async def test_methylation_with_beta_matrix(self, beta_matrix):
        tool = MethylationAnalysisTool()
        params = MethylationAnalysisParams(data_path=str(beta_matrix))

        result = await tool.execute(params)
        assert result["success"] is True
        stats = result["stats"]
        assert stats["format"] == "beta_matrix"
        assert stats["n_cpgs"] == 200
        assert stats["n_samples"] == 6
        assert "distribution" in stats
        assert 0 < stats["global"]["mean_beta"] < 1

    @pytest.mark.asyncio
    async def test_methylation_with_group_comparison(self, beta_with_metadata):
        beta_path, meta_path = beta_with_metadata
        tool = MethylationAnalysisTool()
        params = MethylationAnalysisParams(
            data_path=str(beta_path),
            metadata_path=str(meta_path),
            group_col="group",
        )

        result = await tool.execute(params)
        assert result["success"] is True
        stats = result["stats"]
        assert "differential_methylation" in stats
        dm = stats["differential_methylation"]
        assert "total_dm" in dm
        assert dm["total_dm"] >= 0

    @pytest.mark.asyncio
    async def test_methylation_missing_file_returns_error(self):
        tool = MethylationAnalysisTool()
        params = MethylationAnalysisParams(data_path="/nonexistent/beta.tsv")

        result = await tool.execute(params)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_methylation_params_validation(self):
        params = MethylationAnalysisParams(
            data_path="/data/beta.tsv",
            metadata_path="/data/meta.tsv",
            group_col="condition",
        )
        assert params.data_path == "/data/beta.tsv"
        assert params.group_col == "condition"
