"""TDD-style unit tests for genomics tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.tools.genomics_tools import (
    GwasAnalysisParams,
    GwasAnalysisTool,
    ManhattanPlotParams,
    ManhattanPlotTool,
    VcfStatsParams,
    VcfStatsTool,
)


class TestVcfStatsTool:
    """Unit tests for VcfStatsTool."""

    @pytest.fixture
    def vcf_file(self, tmp_path: Path) -> Path:
        """Create a minimal VCF file for testing."""
        content = (
            "##fileformat=VCFv4.2\n"
            "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\n"
            "1\t100\trs001\tA\tG\t50\tPASS\tAF=0.25\tGT\t0/1\t0/0\n"
            "1\t200\trs002\tC\tT\t80\tPASS\tAF=0.50\tGT\t1/1\t0/1\n"
            "2\t300\trs003\tG\tGC\t99\tPASS\tAF=0.10\tGT\t0/0\t0/1\n"
            "3\t400\trs004\tT\tA\t30\tPASS\tAF=0.75\tGT\t0/1\t1/1\n"
            "3\t500\trs005\tA\tG,T\t60\tPASS\t.\tGT\t0/0\t0/0\n"
        )
        path = tmp_path / "test.vcf"
        path.write_text(content)
        return path

    @pytest.mark.asyncio
    async def test_vcf_stats_with_valid_vcf(self, vcf_file):
        tool = VcfStatsTool()
        params = VcfStatsParams(vcf_path=str(vcf_file))

        result = await tool.execute(params)
        assert result["success"] is True
        stats = result["stats"]
        assert stats["total_variants"] >= 4
        assert stats["snp_count"] >= 2
        assert stats["indel_count"] >= 1
        assert stats["samples"] == 2

    @pytest.mark.asyncio
    async def test_vcf_stats_missing_file(self):
        tool = VcfStatsTool()
        params = VcfStatsParams(vcf_path="/nonexistent/test.vcf")

        result = await tool.execute(params)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_vcf_stats_params_validation(self):
        params = VcfStatsParams(vcf_path="/data/variants.vcf.gz", output_dir="/tmp/out")
        assert params.vcf_path == "/data/variants.vcf.gz"
        assert params.output_dir == "/tmp/out"


class TestGwasAnalysisTool:
    """Unit tests for GwasAnalysisTool."""

    @pytest.fixture
    def vcf_gwas_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a VCF + phenotype file for GWAS testing."""
        import numpy as np

        np.random.seed(42)
        n_variants = 50
        n_samples = 100

        # Phenotype: case/control
        pheno_lines = ["sample_id\tphenotype\tage"]
        sample_ids = [f"SMP{i:03d}" for i in range(n_samples)]
        pheno = np.random.binomial(1, 0.5, n_samples)
        for sid, p in zip(sample_ids, pheno):
            pheno_lines.append(f"{sid}\t{p}\t{np.random.normal(50, 15):.0f}")

        pheno_path = tmp_path / "pheno.tsv"
        pheno_path.write_text("\n".join(pheno_lines))

        # VCF with causal variants
        vcf_lines = [
            "##fileformat=VCFv4.2",
            "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_ids),
        ]
        for i in range(n_variants):
            chrom = np.random.choice(["1", "2", "3", "4", "5"])
            pos = np.random.randint(1, 1000000)
            vid = f"rs{i:04d}"
            ref = np.random.choice(["A", "C", "G", "T"])
            alt = np.random.choice([x for x in ["A", "C", "G", "T"] if x != ref])
            qual = np.random.randint(10, 100)

            # Make some variants associated with phenotype
            gts = []
            for p in pheno:
                if i < 5:  # causal variants
                    dosage = np.random.binomial(2, 0.3 + p * 0.3)
                else:
                    dosage = np.random.binomial(2, 0.3)
                if dosage == 0:
                    gt = "0/0"
                elif dosage == 1:
                    gt = "0/1"
                else:
                    gt = "1/1"
                gts.append(gt)

            af = sum(1 for g in gts if "1" in g) / (2 * len(gts))
            vcf_lines.append(
                f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t{qual}\tPASS\tAF={af:.3f}\tGT\t" + "\t".join(gts)
            )

        vcf_path = tmp_path / "gwas_test.vcf"
        vcf_path.write_text("\n".join(vcf_lines))

        return vcf_path, pheno_path

    @pytest.mark.asyncio
    async def test_gwas_with_vcf_returns_results(self, vcf_gwas_data):
        vcf_path, pheno_path = vcf_gwas_data
        tool = GwasAnalysisTool()
        params = GwasAnalysisParams(
            genotype_path=str(vcf_path),
            phenotype_path=str(pheno_path),
            phenotype_col="phenotype",
            maf_threshold=0.01,
            output_dir=str(vcf_path.parent / "gwas_out"),
        )

        result = await tool.execute(params)
        assert result["success"] is True
        assert result["total_variants"] > 0
        assert "min_p_value" in result

    @pytest.mark.asyncio
    async def test_gwas_missing_file_returns_error(self):
        tool = GwasAnalysisTool()
        params = GwasAnalysisParams(
            genotype_path="/nonexistent/data.vcf",
            phenotype_path="/nonexistent/pheno.tsv",
        )

        result = await tool.execute(params)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_gwas_unsupported_format_returns_error(self, tmp_path: Path):
        unknown = tmp_path / "data.xyz"
        unknown.write_text("not a real file")
        pheno = tmp_path / "pheno.tsv"
        pheno.write_text("sample_id\tphenotype\nS1\t0\n")

        tool = GwasAnalysisTool()
        params = GwasAnalysisParams(
            genotype_path=str(unknown),
            phenotype_path=str(pheno),
        )

        result = await tool.execute(params)
        assert result["success"] is False
        assert "format" in result["error"].lower()


class TestManhattanPlotTool:
    """Unit tests for ManhattanPlotTool."""

    @pytest.fixture
    def gwas_results(self, tmp_path: Path) -> Path:
        import numpy as np
        import pandas as pd

        np.random.seed(123)
        df = pd.DataFrame({
            "chrom": np.repeat([str(i) for i in range(1, 23)], 50),
            "pos": np.concatenate([np.random.randint(1, 1000000, 50) for _ in range(22)]),
            "p_value": np.random.uniform(1e-15, 1, 22 * 50),
        })
        path = tmp_path / "gwas_results.tsv"
        df.to_csv(path, sep="\t", index=False)
        return path

    @pytest.mark.asyncio
    async def test_manhattan_plot_generates_files(self, gwas_results):
        tool = ManhattanPlotTool()
        params = ManhattanPlotParams(
            gwas_result_path=str(gwas_results),
            output_dir=str(gwas_results.parent / "plots"),
            title="Test GWAS",
        )

        result = await tool.execute(params)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_manhattan_plot_missing_file_returns_error(self):
        tool = ManhattanPlotTool()
        params = ManhattanPlotParams(gwas_result_path="/nonexistent/gwas.tsv")

        result = await tool.execute(params)
        assert result["success"] is False
