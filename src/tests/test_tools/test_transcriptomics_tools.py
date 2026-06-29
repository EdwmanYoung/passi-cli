"""TDD-style unit tests for transcriptomics analysis tools."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from passi.tools.transcriptomics_tools import DifferentialAnalysisParams, DifferentialAnalysisTool


class TestDifferentialAnalysisTool:
    """Unit tests for DifferentialAnalysisTool (DESeq2/edgeR/limma)."""

    @pytest.fixture
    def rna_seq_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create synthetic RNA-seq count matrix + metadata."""
        import numpy as np

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


class TestDifferentialAnalysisExecution:
    """Mocked execution paths for rpy2 and Rscript fallbacks."""

    @pytest.fixture
    def rna_seq_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create synthetic RNA-seq count matrix + metadata."""
        import numpy as np

        n_genes, n_samples = 50, 8
        np.random.seed(42)
        baseline = np.random.poisson(100, size=(n_genes, n_samples)).astype(float)
        genes = [f"ENSG{i:05d}" for i in range(n_genes)]
        samples = [f"sample_{i}" for i in range(n_samples)]
        counts_df = pd.DataFrame(baseline, index=genes, columns=samples)
        metadata = pd.DataFrame(
            {"condition": (["control"] * 4) + (["treatment"] * 4)},
            index=samples,
        )
        counts_path = tmp_path / "counts.tsv"
        meta_path = tmp_path / "metadata.tsv"
        counts_df.to_csv(counts_path, sep="\t")
        metadata.to_csv(meta_path, sep="\t")
        return counts_path, meta_path

    def _make_output_file(self, output_path: Path, method: str) -> None:
        """Create a fake DE result file with method-specific columns."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if method == "deseq2":
            df = pd.DataFrame(
                {
                    "gene": ["G1", "G2", "G3"],
                    "baseMean": [100.0, 200.0, 300.0],
                    "log2FoldChange": [1.5, -2.0, 0.5],
                    "lfcSE": [0.1, 0.2, 0.3],
                    "stat": [15.0, -10.0, 1.0],
                    "pvalue": [1e-10, 1e-8, 0.1],
                    "padj": [1e-9, 1e-7, 0.2],
                }
            )
        elif method == "edger":
            df = pd.DataFrame(
                {
                    "gene": ["G1", "G2", "G3"],
                    "logFC": [1.5, -2.0, 0.5],
                    "logCPM": [2.0, 3.0, 4.0],
                    "F": [10.0, 20.0, 1.0],
                    "PValue": [1e-10, 1e-8, 0.1],
                    "FDR": [1e-9, 1e-7, 0.2],
                }
            )
        else:  # limma
            df = pd.DataFrame(
                {
                    "gene": ["G1", "G2", "G3"],
                    "logFC": [1.5, -2.0, 0.5],
                    "AveExpr": [5.0, 6.0, 7.0],
                    "t": [10.0, -8.0, 1.0],
                    "P.Value": [1e-10, 1e-8, 0.1],
                    "adj.P.Val": [1e-9, 1e-7, 0.2],
                }
            )
        df.to_csv(output_path, sep="\t", index=False)

    @pytest.mark.asyncio
    async def test_rpy2_execution_path(self, rna_seq_data, tmp_path: Path):
        """When rpy2 is ready, tool executes via rpy2 and parses output."""
        counts, meta = rna_seq_data
        tool = DifferentialAnalysisTool()
        output_path = tmp_path / "result" / "de_results_deseq2_counts.tsv"
        self._make_output_file(output_path, "deseq2")

        fake_modules = {
            "rpy2": ModuleType("rpy2"),
            "rpy2.robjects": ModuleType("rpy2.robjects"),
            "rpy2.robjects.conversion": ModuleType("conversion"),
            "rpy2.robjects.numpy2ri": ModuleType("numpy2ri"),
            "rpy2.robjects.pandas2ri": ModuleType("pandas2ri"),
        }
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
                params = DifferentialAnalysisParams(
                    counts_path=str(counts),
                    metadata_path=str(meta),
                    group_col="condition",
                    method="deseq2",
                    output_dir=str(tmp_path / "result"),
                )
                result = await tool.execute(params)

        assert result["success"] is True
        assert result["method"] == "deseq2"
        assert result["up_regulated"] == 1
        assert result["down_regulated"] == 1
        assert "top_genes" in result

    @pytest.mark.asyncio
    async def test_rscript_fallback_execution(self, rna_seq_data, tmp_path: Path):
        """When rpy2 is unavailable, tool falls back to Rscript subprocess."""
        counts, meta = rna_seq_data
        tool = DifferentialAnalysisTool()
        output_path = tmp_path / "result" / "de_results_edger_counts.tsv"
        self._make_output_file(output_path, "edger")

        class FakeResult:
            returncode = 0
            stdout = "edgeR_DONE|up=10|down=5|total_genes=50\n"
            stderr = ""

        with patch("passi.executors.r_executor.init_rpy2", return_value={"ready": False}):
            with patch("subprocess.run", return_value=FakeResult()):
                params = DifferentialAnalysisParams(
                    counts_path=str(counts),
                    metadata_path=str(meta),
                    group_col="condition",
                    method="edger",
                    output_dir=str(tmp_path / "result"),
                )
                result = await tool.execute(params)

        assert result["success"] is True
        assert result["method"] == "edger"
        assert result["up_regulated"] == 1
        assert result["down_regulated"] == 1

    @pytest.mark.asyncio
    async def test_rscript_not_found(self, rna_seq_data, tmp_path: Path):
        """Missing Rscript returns a clear error."""
        counts, meta = rna_seq_data
        tool = DifferentialAnalysisTool(r_path="/missing/Rscript")

        with patch("passi.executors.r_executor.init_rpy2", return_value={"ready": False}):
            params = DifferentialAnalysisParams(
                counts_path=str(counts),
                metadata_path=str(meta),
                group_col="condition",
                method="limma",
                output_dir=str(tmp_path / "result"),
            )
            result = await tool.execute(params)

        assert result["success"] is False
        assert "Rscript not found" in result["error"]


@pytest.fixture
def de_counts(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny RNA-seq count matrix + metadata for real R tests."""
    import numpy as np

    np.random.seed(42)
    n_genes, n_samples = 100, 8
    samples = [f"S{i}" for i in range(1, n_samples + 1)]
    genes = [f"GENE{i}" for i in range(1, n_genes + 1)]

    # Baseline counts with a reasonable mean
    counts = np.random.poisson(200, size=(n_genes, n_samples))
    # First 20 genes are up-regulated in treatment (last 4 samples)
    counts[:20, 4:] += np.random.poisson(100, size=(20, 4))
    # Next 20 genes are down-regulated in treatment
    counts[20:40, 4:] = np.maximum(counts[20:40, 4:] - np.random.poisson(80, size=(20, 4)), 0)

    counts_df = pd.DataFrame(counts, index=genes, columns=samples)
    metadata = pd.DataFrame(
        {"condition": (["control"] * 4) + (["treatment"] * 4)},
        index=samples,
    )

    counts_path = tmp_path / "counts.tsv"
    meta_path = tmp_path / "metadata.tsv"
    counts_df.to_csv(counts_path, sep="\t")
    metadata.to_csv(meta_path, sep="\t")
    return counts_path, meta_path


class TestDifferentialAnalysisRealR:
    """Real Rscript execution tests using the project-local R environment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["deseq2", "edger", "limma"])
    async def test_real_rscript_execution(self, de_counts, tmp_path: Path, method: str) -> None:
        """Each DE method runs through the real Rscript subprocess path."""
        counts, meta = de_counts
        tool = DifferentialAnalysisTool()
        params = DifferentialAnalysisParams(
            counts_path=str(counts),
            metadata_path=str(meta),
            group_col="condition",
            method=method,
            output_dir=str(tmp_path / "result"),
            alpha=0.05,
        )

        result = await tool.execute(params)

        assert result["success"] is True, f"{method} failed: {result.get('error', '')}"
        assert result["method"] == method
        assert result["total_genes"] > 0
        assert "output_file" in result
        assert Path(result["output_file"]).exists()
        # Up/down counts are available for all three methods
        assert isinstance(result.get("up_regulated"), int)
        assert isinstance(result.get("down_regulated"), int)
        # The injected signal should produce at least some significant genes.
        assert result["up_regulated"] + result["down_regulated"] > 0


class TestDifferentialAnalysisParseOutput:
    """Direct tests for DifferentialAnalysisTool._parse_output."""

    @pytest.fixture
    def tool(self) -> DifferentialAnalysisTool:
        return DifferentialAnalysisTool()

    def test_missing_output_file(self, tool: DifferentialAnalysisTool, tmp_path: Path) -> None:
        """Missing TSV returns an error with stderr context."""
        missing = tmp_path / "missing.tsv"
        result = tool._parse_output("deseq2", missing, stdout="", stderr="err log")
        assert result["success"] is False
        assert "not created" in result["error"]
        assert result["stderr"] == "err log"

    def test_parse_deseq2_output(self, tool: DifferentialAnalysisTool, tmp_path: Path) -> None:
        """DESeq2 columns are mapped to up/down/significant counts."""
        path = tmp_path / "de.tsv"
        df = pd.DataFrame(
            {
                "gene": ["A", "B", "C", "D"],
                "log2FoldChange": [2.0, -1.5, 0.5, 3.0],
                "padj": [0.01, 0.02, 0.5, 0.001],
            }
        )
        df.to_csv(path, sep="\t", index=False)

        result = tool._parse_output("deseq2", path)
        assert result["success"] is True
        assert result["up_regulated"] == 2
        assert result["down_regulated"] == 1
        assert result["significant"] == 3

    def test_parse_limma_output(self, tool: DifferentialAnalysisTool, tmp_path: Path) -> None:
        """limma columns are mapped correctly."""
        path = tmp_path / "limma.tsv"
        df = pd.DataFrame(
            {
                "gene": ["A", "B", "C"],
                "logFC": [1.0, -2.0, 0.1],
                "adj.P.Val": [0.01, 0.04, 0.2],
            }
        )
        df.to_csv(path, sep="\t", index=False)

        result = tool._parse_output("limma", path)
        assert result["success"] is True
        assert result["up_regulated"] == 1
        assert result["down_regulated"] == 1

    def test_truncates_large_outputs(self, tool: DifferentialAnalysisTool, tmp_path: Path) -> None:
        """Outputs over 5000 rows are truncated to 5000."""
        path = tmp_path / "big.tsv"
        df = pd.DataFrame(
            {
                "gene": [f"G{i}" for i in range(6000)],
                "log2FoldChange": [1.0] * 6000,
                "padj": [0.01] * 6000,
            }
        )
        df.to_csv(path, sep="\t", index=False)

        result = tool._parse_output("deseq2", path)
        assert result["success"] is True
        assert result["total_genes"] == 5000
