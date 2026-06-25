"""TDD-style unit tests for EnrichmentTool (fgsea + ORA)."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.tools.enrichment_tools import EnrichmentParams, EnrichmentTool


class TestEnrichmentTool:
    """Unit tests for EnrichmentTool — fgsea and ORA."""

    @pytest.fixture
    def ranked_genes(self, tmp_path: Path) -> Path:
        """Create a synthetic ranked gene list for fgsea."""
        import numpy as np
        import pandas as pd

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
