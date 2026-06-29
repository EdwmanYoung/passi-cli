"""TDD-style unit tests for EnrichmentTool (fgsea + ORA)."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from passi.tools.enrichment_tools import EnrichmentParams, EnrichmentTool


class TestEnrichmentTool:
    """Unit tests for EnrichmentTool — fgsea and ORA."""

    @pytest.fixture
    def ranked_genes(self, tmp_path: Path) -> Path:
        """Create a synthetic ranked gene list for fgsea."""
        import numpy as np

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "gene": [f"GENE{i}" for i in range(500)],
                "rank_val": np.random.normal(0, 1, 500),
            }
        )
        # Make some genes significantly changed
        df.loc[:49, "rank_val"] += 3
        df = df.sort_values("rank_val", ascending=False)
        path = tmp_path / "ranked.tsv"
        df.to_csv(path, sep="\t", index=False, header=False)
        return path

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self, ranked_genes):
        tool = EnrichmentTool()
        params = EnrichmentParams(method="bad_method", ranked_path=str(ranked_genes))

        result = await tool.execute(params)
        assert result["success"] is False
        assert "Unknown method" in result["error"]

    @pytest.mark.asyncio
    async def test_fgsea_missing_ranked_file_returns_error(self):
        tool = EnrichmentTool()
        params = EnrichmentParams(method="fgsea", ranked_path="/nonexistent/ranked.tsv")

        result = await tool.execute(params)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_ora_missing_de_file_returns_error(self):
        tool = EnrichmentTool()
        params = EnrichmentParams(method="ora", de_path="/nonexistent/de.tsv")

        result = await tool.execute(params)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_params_validation(self):
        params = EnrichmentParams(
            method="fgsea",
            ranked_path="/data/ranked.tsv",
            gmt_path="/data/hallmark.gmt",
            min_size=10,
            max_size=1000,
            alpha=0.01,
        )
        assert params.method == "fgsea"
        assert params.min_size == 10
        assert params.alpha == 0.01

    @pytest.mark.asyncio
    async def test_fgsea_script_template(self):
        """fgsea R script renders correctly."""
        from passi.tools.enrichment_tools import FGSEA_SCRIPT

        script = FGSEA_SCRIPT.format(
            ranked_path="/data/ranked.tsv",
            gmt_path="/data/pathways.gmt",
            min_size=10,
            max_size=500,
            n_perm=1000,
            alpha=0.05,
            output_path="/tmp/fgsea_out.tsv",
            output_json="/tmp/fgsea.json",
        )
        assert "fgsea" in script
        assert "ranks" in script
        assert "gmtPathways" in script
        assert "/data/pathways.gmt" in script

    @pytest.mark.asyncio
    async def test_ora_script_template(self):
        """ORA R script renders correctly."""
        from passi.tools.enrichment_tools import ORA_SCRIPT

        script = ORA_SCRIPT.format(
            de_path="/data/de_results.tsv",
            gene_col="gene_id",
            p_col="padj",
            gene_id_type="ENSEMBL",
            ontologies="BP,MF",
            do_kegg="true",
            alpha=0.05,
            output_go="/tmp/go.tsv",
            output_kegg="/tmp/kegg.tsv",
            output_json="/tmp/ora.json",
        )
        assert "enrichGO" in script
        assert "enrichKEGG" in script
        assert "gene_id" in script
        assert "padj" in script


class TestEnrichmentExecution:
    """Mocked execution paths for rpy2 and Rscript fallbacks."""

    @pytest.fixture
    def ranked_genes(self, tmp_path: Path) -> Path:
        """Create a small ranked gene list."""
        df = pd.DataFrame(
            {
                "gene": [f"GENE{i}" for i in range(50)],
                "rank_val": [float(i) for i in range(50)],
            }
        )
        path = tmp_path / "ranked.tsv"
        df.to_csv(path, sep="\t", index=False, header=False)
        return path

    @pytest.fixture
    def de_results(self, tmp_path: Path) -> Path:
        """Create a small DE result table for ORA."""
        df = pd.DataFrame(
            {
                "gene": [f"GENE{i}" for i in range(100)],
                "padj": [0.001] * 20 + [0.5] * 80,
            }
        )
        path = tmp_path / "de.tsv"
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
    async def test_fgsea_rpy2_path(self, ranked_genes: Path, tmp_path: Path) -> None:
        tool = EnrichmentTool()
        output_json = tmp_path / "fgsea_summary.json"
        self._make_json(
            output_json,
            {
                "method": "fgsea",
                "total_pathways": 10,
                "up_regulated": 3,
                "down_regulated": 2,
            },
        )

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
                params = EnrichmentParams(
                    method="fgsea",
                    ranked_path=str(ranked_genes),
                    output_dir=str(tmp_path),
                )
                result = await tool.execute(params)

        assert result["success"] is True
        assert result["method"] == "fgsea"
        assert result["total_pathways"] == 10

    @pytest.mark.asyncio
    async def test_ora_rscript_fallback(self, de_results: Path, tmp_path: Path) -> None:
        tool = EnrichmentTool()
        output_json = tmp_path / "ora_summary.json"
        self._make_json(
            output_json,
            {
                "method": "ora",
                "n_significant_genes": 20,
                "go_terms": 5,
                "kegg_pathways": 3,
            },
        )

        class FakeResult:
            returncode = 0
            stdout = "ORA_DONE|go_terms=5|kegg_pathways=3\n"
            stderr = ""

        with patch("passi.executors.r_executor.init_rpy2", return_value={"ready": False}):
            with patch("subprocess.run", return_value=FakeResult()):
                params = EnrichmentParams(
                    method="ora",
                    de_path=str(de_results),
                    output_dir=str(tmp_path),
                )
                result = await tool.execute(params)

        assert result["success"] is True
        assert result["method"] == "ora"
        assert result["go_terms"] == 5
        assert result["kegg_pathways"] == 3


class TestEnrichmentRealR:
    """Real Rscript execution tests using the project-local R environment."""

    @pytest.fixture
    def ranked_genes_and_gmt(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a ranked gene list and a matching GMT file for fgsea."""
        import numpy as np

        np.random.seed(42)
        # Two gene sets that overlap with the ranked list.
        up_genes = [
            "TP53", "BRCA1", "EGFR", "MYC", "KRAS",
            "PTEN", "AKT1", "CDKN2A", "BCL2", "CASP3",
        ]
        down_genes = [
            "CDK1", "CDK2", "E2F1", "RB1", "ATM",
            "CHEK2", "MLH1", "MSH2", "PIK3CA", "ERBB2",
        ]
        other_genes = [
            "VEGFA", "IL6", "TNF", "NFKB1", "STAT1", "STAT3", "JAK1", "JAK2",
            "MAPK1", "MAPK3", "SRC", "MTOR", "FOXO3", "PPARG", "ESR1", "AR",
            "CCND1", "CDK4", "CDK6", "BIRC5", "MKI67", "TOP2A", "AURKA", "PLK1",
        ]
        genes = up_genes + down_genes + other_genes

        ranks = np.random.normal(0, 1, size=len(genes))
        # Strong positive signal for up_genes, strong negative for down_genes.
        ranks[: len(up_genes)] += 3.0
        ranks[len(up_genes) : len(up_genes) + len(down_genes)] -= 2.5

        df = pd.DataFrame({"gene": genes, "rank": ranks}).sort_values("rank", ascending=False)
        ranked_path = tmp_path / "ranked.tsv"
        df.to_csv(ranked_path, sep="\t", index=False, header=False)

        gmt_path = tmp_path / "pathways.gmt"
        gmt_lines = [
            "UP_PATHWAY\tup_pathway\t" + "\t".join(up_genes),
            "DOWN_PATHWAY\tdown_pathway\t" + "\t".join(down_genes),
        ]
        gmt_path.write_text("\n".join(gmt_lines), encoding="utf-8")

        return ranked_path, gmt_path

    @pytest.fixture
    def de_results_real(self, tmp_path: Path) -> Path:
        """Create a DE result table with real human gene symbols for ORA."""
        # Known human symbols; some are classic cancer/immune genes.
        sig_genes = [
            "TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PTEN", "AKT1", "CDKN2A",
            "BCL2", "CASP3", "CDK1", "E2F1", "RB1", "ATM", "CHEK2",
        ]
        background = [
            "GAPDH", "ACTB", "TUBB", "RPLP0", "RPS18", "HSP90AA1", "HSPA1A",
            "ALB", "FN1", "COL1A1", "VIM", "SOX2", "NANOG", "OCT4", "KLF4",
            "FOS", "JUN", "MYC", "MAX", "SOS1", "GRB2", "SHC1", "RASGRP1",
            "PRKCA", "PRKCB", "CAMK2A", "CREB1", "ATF1", "SMAD2", "SMAD3",
            "SMAD4", "TGFB1", "BMP2", "WNT1", "CTNNB1", "TCF7", "LEF1",
            "NOTCH1", "DLL1", "JAG1", "HES1", "HEY1", "SHH", "GLI1", "PTCH1",
        ]
        genes = sig_genes + background
        padj = [0.001] * len(sig_genes) + [0.5] * len(background)
        df = pd.DataFrame({"gene": genes, "padj": padj})
        path = tmp_path / "de.tsv"
        df.to_csv(path, sep="\t", index=False)
        return path

    @pytest.mark.asyncio
    async def test_fgsea_real_r(self, ranked_genes_and_gmt, tmp_path: Path) -> None:
        """fgsea runs with real Rscript and a custom GMT file."""
        ranked_path, gmt_path = ranked_genes_and_gmt
        tool = EnrichmentTool()
        params = EnrichmentParams(
            method="fgsea",
            ranked_path=str(ranked_path),
            gmt_path=str(gmt_path),
            output_dir=str(tmp_path / "result"),
            n_perm=1000,
            min_size=5,
            max_size=500,
        )
        result = await tool.execute(params)

        assert result["success"] is True, result.get("error", "")
        assert result["method"] == "fgsea"
        assert result["total_pathways"] > 0
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_ora_real_r(self, de_results_real: Path, tmp_path: Path) -> None:
        """ORA runs with real Rscript using org.Hs.eg.db and GO (KEGG disabled)."""
        tool = EnrichmentTool()
        params = EnrichmentParams(
            method="ora",
            de_path=str(de_results_real),
            gene_col="gene",
            p_col="padj",
            gene_id_type="SYMBOL",
            ontologies="BP",
            do_kegg=False,
            alpha=0.05,
            output_dir=str(tmp_path / "result"),
        )
        result = await tool.execute(params)

        assert result["success"] is True, result.get("error", "")
        assert result["method"] == "ora"
        assert "go_terms" in result
        assert "summary" in result


class TestEnrichmentParseOutput:
    """Direct tests for EnrichmentTool._parse_output."""

    @pytest.fixture
    def tool(self) -> EnrichmentTool:
        return EnrichmentTool()

    def test_fgsea_done_line_parsed(self, tool: EnrichmentTool, tmp_path: Path) -> None:
        result = tool._parse_output(
            "fgsea",
            tmp_path / "missing.json",
            tmp_path,
            stdout="FGSEA_DONE|pathways=50|up=5|down=3\n",
        )
        assert result["success"] is True
        assert result["pathways"] == 50
        assert result["up"] == 5
        assert result["down"] == 3

    def test_summary_promoted(self, tool: EnrichmentTool, tmp_path: Path) -> None:
        json_path = tmp_path / "fgsea_summary.json"
        json_path.write_text(
            json.dumps({"total_pathways": 100, "up_regulated": 10, "down_regulated": 5}),
            encoding="utf-8",
        )
        result = tool._parse_output("fgsea", json_path, tmp_path)
        assert result["total_pathways"] == 100
        assert result["up_regulated"] == 10
        assert "summary" in result

    def test_parse_warning_on_bad_json(self, tool: EnrichmentTool, tmp_path: Path) -> None:
        json_path = tmp_path / "fgsea_summary.json"
        json_path.write_text("not json", encoding="utf-8")
        result = tool._parse_output("fgsea", json_path, tmp_path)
        assert "parse_warning" in result
