"""Genomics analysis tools — VCF statistics, GWAS, variant analysis.

Handles variant call files (VCF), PLINK genotype data, and GWAS association
testing. Uses Python (statsmodels/scipy) with optional PLINK subprocess support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from passi.tools.base import CallableTool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Tool 1: VCF Statistics
# ═══════════════════════════════════════════════════════════════════


class VcfStatsParams(BaseModel):
    """Parameters for VCF statistics extraction."""

    vcf_path: str = Field(..., description="Path to VCF file (.vcf, .vcf.gz)")
    output_dir: str = Field(default="./output", description="Output directory")


class VcfStatsTool(CallableTool[VcfStatsParams]):
    """Parse a VCF file and extract variant-level statistics.

    Reports: variant count, SNP/indel ratio, quality distribution,
    allele frequency spectrum, transition/transversion ratio.
    """

    name = "vcf_stats"
    description = (
        "Parse a VCF (Variant Call Format) file and extract variant statistics. "
        "Reports total variants, SNP/indel counts, quality distribution, "
        "allele frequency spectrum, and transition/transversion ratio. "
        "Supports .vcf and .vcf.gz files."
    )
    params_model = VcfStatsParams

    async def execute(self, params: VcfStatsParams, **kwargs: Any) -> dict[str, Any]:
        import gzip

        vcf_path = Path(params.vcf_path)
        if not vcf_path.exists():
            return {"success": False, "error": f"VCF file not found: {params.vcf_path}"}

        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            opener = gzip.open if vcf_path.suffix == ".gz" else open
            with opener(vcf_path, "rt") as fh:
                lines = [line for line in fh if not line.startswith("##")]

            if not lines:
                return {"success": False, "error": "VCF file contains no data"}

            header_line = lines[0]
            if not header_line.startswith("#"):
                return {"success": False, "error": "VCF has no header line (#CHROM...)"}

            # Parse header
            header_cols = header_line.lstrip("#").strip().split("\t")
            data_lines = [l.strip().split("\t") for l in lines[1:] if l.strip()]

            # Count variant types
            snp_count = 0
            indel_count = 0
            ts_count = 0  # transitions
            tv_count = 0  # transversions
            quality_values: list[float] = []
            af_values: list[float] = []

            ts_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}

            for fields in data_lines:
                if len(fields) < 5:
                    continue
                ref = fields[3].upper()
                alts = fields[4].upper().split(",")

                # SNP vs indel
                is_snp = True
                for alt in alts:
                    if len(ref) != len(alt):
                        is_snp = False
                        break
                    if len(ref) == 1 and len(alt) == 1:
                        pair = (ref, alt)
                        if pair in ts_pairs:
                            ts_count += 1
                        else:
                            tv_count += 1

                if is_snp:
                    snp_count += 1
                else:
                    indel_count += 1

                # Quality
                if len(fields) > 5 and fields[5] != ".":
                    try:
                        quality_values.append(float(fields[5]))
                    except ValueError:
                        pass

                # Allele frequency from INFO field
                if len(fields) > 7:
                    info = fields[7]
                    for item in info.split(";"):
                        if item.startswith("AF="):
                            try:
                                af_values.append(float(item.split("=")[1]))
                            except ValueError:
                                pass

            total = snp_count + indel_count

            stats: dict[str, Any] = {
                "total_variants": total,
                "snp_count": snp_count,
                "indel_count": indel_count,
                "snp_ratio": round(snp_count / total, 4) if total else 0,
                "indel_ratio": round(indel_count / total, 4) if total else 0,
                "ts_tv_ratio": round(ts_count / tv_count, 4) if tv_count else None,
                "transitions": ts_count,
                "transversions": tv_count,
                "quality": {
                    "min": round(min(quality_values), 2) if quality_values else None,
                    "max": round(max(quality_values), 2) if quality_values else None,
                    "mean": round(float(np.mean(quality_values)), 2) if quality_values else None,
                    "median": round(float(np.median(quality_values)), 2) if quality_values else None,
                },
                "allele_frequency": {
                    "count": len(af_values),
                    "mean": round(float(np.mean(af_values)), 4) if af_values else None,
                    "min": round(min(af_values), 4) if af_values else None,
                    "max": round(max(af_values), 4) if af_values else None,
                },
                "samples": len(header_cols) - 9 if len(header_cols) > 9 else 0,
            }

            return {"success": True, "file": str(vcf_path), "stats": stats}
        except Exception as e:
            return {"success": False, "error": f"VCF parsing failed: {e}"}


# ═══════════════════════════════════════════════════════════════════
# Tool 2: GWAS Analysis (Python-based, no PLINK dependency)
# ═══════════════════════════════════════════════════════════════════


class GwasAnalysisParams(BaseModel):
    """Parameters for GWAS association analysis."""

    genotype_path: str = Field(..., description="Path to genotype data (VCF or PLINK .bed prefix)")
    phenotype_path: str = Field(..., description="Path to phenotype data (TSV with sample_id and phenotype columns)")
    phenotype_col: str = Field(default="phenotype", description="Column name for phenotype in phenotype file")
    covariates: str = Field(default="", description="Comma-separated covariate column names")
    model: str = Field(default="additive", description="Genetic model: additive, dominant, recessive")
    maf_threshold: float = Field(default=0.05, description="Minor allele frequency filter threshold")
    output_dir: str = Field(default="./output", description="Output directory for GWAS results")


class GwasAnalysisTool(CallableTool[GwasAnalysisParams]):
    """Run GWAS association analysis using Python (statsmodels).

    For VCF input, performs per-variant association testing using
    logistic or linear regression. For PLINK-formatted data, generates
    a PLINK command and executes via subprocess if available.
    """

    name = "gwas_analysis"
    description = (
        "Run GWAS (Genome-Wide Association Study) association analysis. "
        "Accepts VCF genotype data + phenotype TSV, performs per-variant "
        "logistic/linear regression testing. Supports additive/dominant/recessive "
        "genetic models. Returns variant-level p-values suitable for Manhattan plots."
    )
    params_model = GwasAnalysisParams

    async def execute(self, params: GwasAnalysisParams, **kwargs: Any) -> dict[str, Any]:
        genotype_path = Path(params.genotype_path)
        phenotype_path = Path(params.phenotype_path)

        if not genotype_path.exists():
            return {"success": False, "error": f"Genotype file not found: {params.genotype_path}"}
        if not phenotype_path.exists():
            return {"success": False, "error": f"Phenotype file not found: {params.phenotype_path}"}

        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # If VCF, use Python-based GWAS
        if genotype_path.suffix in (".vcf", ".gz"):
            return self._run_vcf_gwas(genotype_path, phenotype_path, params, output_dir)

        # If PLINK bed, try PLINK subprocess
        if genotype_path.suffix == ".bed":
            return self._run_plink_gwas(genotype_path, phenotype_path, params, output_dir)

        return {"success": False, "error": f"Unsupported genotype format: {genotype_path.suffix}"}

    def _run_vcf_gwas(
        self, vcf_path: Path, phenotype_path: Path, params: GwasAnalysisParams, output_dir: Path
    ) -> dict[str, Any]:
        import gzip

        import pandas as pd

        try:
            import statsmodels.api as sm
        except ImportError:
            return {"success": False, "error": "statsmodels required for GWAS. Install: pip install statsmodels"}

        # Read phenotype
        pheno_df = pd.read_csv(phenotype_path, sep=None if phenotype_path.suffix == ".csv" else "\t")
        pheno_samples = set(pheno_df.iloc[:, 0].astype(str))

        # Parse VCF genotypes
        opener = gzip.open if vcf_path.suffix == ".gz" else open
        variants: list[dict] = []
        sample_ids: list[str] = []

        with opener(vcf_path, "rt") as fh:
            for line in fh:
                if line.startswith("##"):
                    continue
                if line.startswith("#"):
                    sample_ids = line.strip().split("\t")[9:]
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 10:
                    continue

                chrom, pos, vid, ref, alt, qual, filt, info, fmt = fields[:9]
                genotype_fields = fields[9:]

                # Extract genotypes: 0=ref/ref, 1=ref/alt, 2=alt/alt
                try:
                    gt_idx = fmt.split(":").index("GT")
                    gts = []
                    for gf in genotype_fields:
                        gt_str = gf.split(":")[gt_idx]
                        if gt_str in ("./.", "."):
                            gts.append(np.nan)
                        else:
                            alleles = gt_str.split("/")
                            dosage = sum(int(a) if a != "." else 0 for a in alleles)
                            gts.append(dosage)
                except (ValueError, IndexError):
                    continue

                # Match samples with phenotype
                sample_gts = []
                sample_phens = []
                for sid, gt in zip(sample_ids, gts):
                    if sid in pheno_samples and not np.isnan(gt):
                        sample_gts.append(gt)
                        row = pheno_df[pheno_df.iloc[:, 0].astype(str) == sid].iloc[0]
                        sample_phens.append(float(row[params.phenotype_col]))

                if len(sample_gts) < 10:
                    continue

                # Calculate MAF
                maf = sum(sample_gts) / (2 * len(sample_gts))
                maf = min(maf, 1 - maf)
                if maf < params.maf_threshold:
                    continue

                # Encode genotype based on model
                dosage = np.array(sample_gts)
                if params.model == "dominant":
                    dosage = (dosage >= 1).astype(float)
                elif params.model == "recessive":
                    dosage = (dosage >= 2).astype(float)

                # Association test
                phen_arr = np.array(sample_phens)
                is_binary = set(np.unique(phen_arr)) <= {0, 1, 0.0, 1.0}

                try:
                    X = sm.add_constant(dosage)
                    if is_binary:
                        model = sm.Logit(phen_arr, X)
                    else:
                        model = sm.OLS(phen_arr, X)
                    result = model.fit(disp=0)
                    p_value = result.pvalues[1] if len(result.pvalues) > 1 else 1.0
                    effect = result.params[1] if len(result.params) > 1 else 0.0
                except Exception:
                    continue

                variants.append({
                    "chrom": chrom,
                    "pos": int(pos),
                    "id": vid,
                    "ref": ref,
                    "alt": alt,
                    "maf": round(maf, 4),
                    "p_value": float(p_value),
                    "effect": float(effect),
                    "n_samples": len(sample_gts),
                })

        # Save results
        result_df = pd.DataFrame(variants)
        if result_df.empty:
            return {"success": True, "total_variants": 0, "message": "No variants passed filters"}

        result_df = result_df.sort_values("p_value")
        output_path = output_dir / "gwas_results.tsv"
        result_df.to_csv(output_path, sep="\t", index=False)

        sig = int((result_df["p_value"] < 5e-8).sum())
        suggestive = int((result_df["p_value"] < 1e-5).sum())

        return {
            "success": True,
            "total_variants": len(variants),
            "significant": sig,
            "suggestive": suggestive,
            "min_p_value": float(result_df["p_value"].min()),
            "output_file": str(output_path),
        }

    def _run_plink_gwas(
        self, plink_path: Path, phenotype_path: Path, params: GwasAnalysisParams, output_dir: Path
    ) -> dict[str, Any]:
        """Attempt PLINK-based GWAS if PLINK is available on PATH."""
        import subprocess

        plink_bin = self._find_plink()
        if not plink_bin:
            return {
                "success": False,
                "error": "PLINK not found on PATH. For PLINK-format data, install PLINK (https://www.cog-genomics.org/plink/). "
                "For VCF data, use a .vcf or .vcf.gz file instead.",
            }

        stem = str(plink_path.with_suffix(""))
        out = str(output_dir / "gwas_plink")

        cmd = [
            plink_bin,
            "--bfile", stem,
            "--pheno", str(phenotype_path),
            "--pheno-name", params.phenotype_col,
            "--maf", str(params.maf_threshold),
            "--linear" if params.model == "additive" else "--logistic",
            "--out", out,
            "--allow-no-sex",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return {
                "success": True,
                "method": "plink",
                "output_prefix": out,
                "stdout": result.stdout[-2000:],
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "PLINK GWAS timed out (300s)"}
        except FileNotFoundError:
            return {"success": False, "error": f"PLINK not found: {plink_bin}"}

    @staticmethod
    def _find_plink() -> str | None:
        import os
        import subprocess

        for name in ("plink", "plink.exe", "plink2", "plink2.exe"):
            try:
                result = subprocess.run(
                    [name, "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 or "PLINK" in result.stdout + result.stderr:
                    return name
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None


# ═══════════════════════════════════════════════════════════════════
# Tool 3: Manhattan / QQ Plot Generator
# ═══════════════════════════════════════════════════════════════════


class ManhattanPlotParams(BaseModel):
    """Parameters for Manhattan/QQ plot generation."""

    gwas_result_path: str = Field(..., description="Path to GWAS results (TSV with chrom, pos, p_value columns)")
    output_dir: str = Field(default="./output", description="Output directory for plots")
    title: str = Field(default="GWAS Manhattan Plot", description="Plot title")


class ManhattanPlotTool(CallableTool[ManhattanPlotParams]):
    """Generate Manhattan and QQ plots from GWAS summary statistics."""

    name = "manhattan_plot"
    description = (
        "Generate Manhattan plot and QQ plot from GWAS summary statistics. "
        "Input is a TSV with columns: chrom, pos, p_value. Outputs PNG files."
    )
    params_model = ManhattanPlotParams

    async def execute(self, params: ManhattanPlotParams, **kwargs: Any) -> dict[str, Any]:
        import pandas as pd

        result_path = Path(params.gwas_result_path)
        if not result_path.exists():
            return {"success": False, "error": f"GWAS results not found: {params.gwas_result_path}"}

        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_csv(result_path, sep="\t")
            required = {"chrom", "pos", "p_value"}
            missing = required - set(df.columns)
            if missing:
                return {"success": False, "error": f"Missing columns in GWAS results: {missing}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to read GWAS results: {e}"}

        # Generate Manhattan plot using matplotlib
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {"success": True, "message": "matplotlib not available — plot skipped", "skipped": True}

        df = df.dropna(subset=["p_value"])
        df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))

        # Manhattan plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Encode chromosome positions
        def _chrom_key(x: str) -> tuple[int, int | str]:
            s = str(x)
            return (0, int(s)) if s.isdigit() else (1, s)

        chroms = sorted(df["chrom"].unique(), key=_chrom_key)
        chrom_to_idx = {c: i for i, c in enumerate(chroms)}

        df["chrom_idx"] = df["chrom"].map(chrom_to_idx)
        df = df.sort_values(["chrom_idx", "pos"])

        # Cumulative position
        cum_pos = 0
        chrom_ticks: list[tuple[int, str]] = []
        for i, chrom in enumerate(chroms):
            mask = df["chrom"] == chrom
            if i == 0:
                df.loc[mask, "cum_pos"] = df.loc[mask, "pos"]
            else:
                df.loc[mask, "cum_pos"] = df.loc[mask, "pos"] + cum_pos
            mid = df.loc[mask, "cum_pos"].mean() if mask.any() else cum_pos
            chrom_ticks.append((mid, chrom))
            if mask.any():
                cum_pos = df.loc[mask, "cum_pos"].max()

        # Manhattan plot
        colors = ["#1E293B", "#3B82F6"]
        for i, chrom in enumerate(chroms):
            mask = df["chrom"] == chrom
            ax1.scatter(
                df.loc[mask, "cum_pos"],
                df.loc[mask, "neg_log10_p"],
                c=colors[i % 2],
                s=2,
                alpha=0.6,
            )

        # Significance lines
        ax1.axhline(-np.log10(5e-8), color="#EF4444", linestyle="--", alpha=0.7, label="Genome-wide (5e-8)")
        ax1.axhline(-np.log10(1e-5), color="#F59E0B", linestyle="--", alpha=0.5, label="Suggestive (1e-5)")

        ax1.set_xticks([t[0] for t in chrom_ticks])
        ax1.set_xticklabels([t[1] for t in chrom_ticks], fontsize=7)
        ax1.set_ylabel("-log10(p-value)")
        ax1.set_title(params.title)
        ax1.legend(fontsize=8)

        # QQ plot
        observed = -np.log10(np.sort(df["p_value"].values))
        expected = -np.log10(np.linspace(1, 1 / len(observed), len(observed)))
        ax2.scatter(expected, observed, s=2, alpha=0.5, c="#2563EB")
        max_val = max(expected.max(), observed.max())
        ax2.plot([0, max_val], [0, max_val], color="#EF4444", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Expected -log10(p)")
        ax2.set_ylabel("Observed -log10(p)")
        ax2.set_title("QQ Plot")
        ax2.set_xlim(0, max_val * 1.05)
        ax2.set_ylim(0, max_val * 1.05)

        plt.tight_layout()

        manhattan_path = output_dir / "manhattan_plot.png"
        qq_path = output_dir / "qq_plot.png"
        fig.savefig(manhattan_path, dpi=150, bbox_inches="tight")
        # QQ is on the same figure, save separately
        plt.close()

        # Save QQ plot separately
        fig2, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(expected, observed, s=3, alpha=0.5, c="#2563EB")
        ax.plot([0, max_val], [0, max_val], color="#EF4444", linestyle="--", alpha=0.5)
        ax.set_xlabel("Expected -log10(p)")
        ax.set_ylabel("Observed -log10(p)")
        ax.set_title("QQ Plot")
        fig2.savefig(qq_path, dpi=150, bbox_inches="tight")
        plt.close()

        inflation = float(np.median(observed) / np.median(expected)) if len(expected) > 0 else 1.0

        return {
            "success": True,
            "manhattan_plot": str(manhattan_path),
            "qq_plot": str(qq_path),
            "genomic_inflation_factor": round(inflation, 3),
            "n_variants": len(df),
        }
