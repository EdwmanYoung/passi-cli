"""TDD-style unit tests for transcriptomics analysis tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.tools.transcriptomics_tools import DifferentialAnalysisParams, DifferentialAnalysisTool


class TestDifferentialAnalysisTool:
    """Unit tests for DifferentialAnalysisTool (DESeq2/edgeR/limma)."""

    @pytest.fixture
    def rna_seq_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create synthetic RNA-seq count matrix + metadata."""
        import numpy as np
        import pandas as pd

        n_genes, n_samples = 200, 8
        np.random.seed(42)

        # Count matrix: genes × samples
        baseline = np.random.poisson(100, size=(n_genes, n_samples)).astype(float)
        # Add DE signal: first 50 genes, last 4 samples
        baseline[:50, 4:] += np.random.poisson(80, size=(50, 4))
        baseline[100:130, 4:] -= np.random.poisson(40, size=(30, 4))

        genes = [f"ENSG{i:05d}" for i in range(n_genes)]
        samples = [f"sample_{i}" for i in range(n_samples)]
        counts_df = pd.DataFrame(baseline, index=genes, columns=samples)

        # Metadata
        metadata = pd.DataFrame(
            {
                "condition": (["control"] * 4) + (["treatment"] * 4),
                "batch": ["A", "B", "A", "B"] * 2,
            },
            index=samples,
        )

        counts_path = tmp_path / "counts.tsv"
        meta_path = tmp_path / "metadata.tsv"
        counts_df.to_csv(counts_path, sep="\t")
        metadata.to_csv(meta_path, sep="\t")

        return counts_path, meta_path

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self, rna_seq_data):
        # Arrange
        counts, meta = rna_seq_data
        tool = DifferentialAnalysisTool()
        params = DifferentialAnalysisParams(
            counts_path=str(counts),
            metadata_path=str(meta),
            group_col="condition",
            method="unknown_method",
        )

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False
        assert "Unknown method" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_counts_file_returns_error(self, rna_seq_data):
        # Arrange
        _, meta = rna_seq_data
        tool = DifferentialAnalysisTool()
        params = DifferentialAnalysisParams(
            counts_path="/nonexistent/counts.tsv",
            metadata_path=str(meta),
            group_col="condition",
        )

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_params_model_validates(self):
        """Ensure the Pydantic model accepts valid parameters."""
        params = DifferentialAnalysisParams(
            counts_path="/data/counts.tsv",
            metadata_path="/data/meta.tsv",
            group_col="condition",
            method="deseq2",
            alpha=0.01,
        )
        assert params.alpha == 0.01
        assert params.method == "deseq2"

    @pytest.mark.asyncio
    async def test_deseq2_script_is_generated(self, rna_seq_data):
        """Verify R script template renders without KeyError."""
        counts, meta = rna_seq_data
        from passi.tools.transcriptomics_tools import METHOD_SCRIPTS

        script = METHOD_SCRIPTS["deseq2"].format(
            counts_path=str(counts).replace("\\", "/"),
            metadata_path=str(meta).replace("\\", "/"),
            group_col="condition",
            alpha=0.05,
            output_path="/tmp/test_out.tsv",
        )
        assert "DESeq2" in script
        assert "DESeqDataSetFromMatrix" in script
        assert "/tmp/test_out.tsv" in script

    @pytest.mark.asyncio
    async def test_all_methods_have_scripts(self):
        """All three methods have R script templates."""
        from passi.tools.transcriptomics_tools import METHOD_SCRIPTS

        for method in ("deseq2", "edger", "limma"):
            assert method in METHOD_SCRIPTS
            assert "read.table" in METHOD_SCRIPTS[method]
