"""Epigenetics analysis tools — peak QC, methylation analysis.

Handles ChIP-seq/ATAC-seq peak files (narrowPeak, broadPeak, BED) and
DNA methylation data (beta matrices, Bismark coverage files).
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
# Tool 1: Peak QC
# ═══════════════════════════════════════════════════════════════════


class PeakQcParams(BaseModel):
    """Parameters for peak file QC analysis."""

    peak_path: str = Field(..., description="Path to peak file (narrowPeak, broadPeak, or BED)")
    align_path: str = Field(default="", description="Optional path to BAM file for FRiP calculation")
    output_dir: str = Field(default="./output", description="Output directory")


class PeakQcTool(CallableTool[PeakQcParams]):
    """Calculate QC metrics for ChIP-seq / ATAC-seq peak files.

    Reports: peak count, width distribution, signal value distribution,
    FRiP score (if BAM provided), and ENCODE-style QC recommendations.
    """

    name = "peak_qc"
    description = (
        "Calculate quality control metrics for ChIP-seq / ATAC-seq peak files. "
        "Accepts narrowPeak, broadPeak, or BED format. Reports peak count, "
        "width distribution, signal value statistics, and FRiP (Fraction of "
        "Reads in Peaks) if a BAM alignment file is provided."
    )
    params_model = PeakQcParams

    async def execute(self, params: PeakQcParams, **kwargs: Any) -> dict[str, Any]:
        peak_path = Path(params.peak_path)
        if not peak_path.exists():
            return {"success": False, "error": f"Peak file not found: {params.peak_path}"}

        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Detect if gzipped
            if peak_path.suffix == ".gz":
                import gzip
                fh = gzip.open(peak_path, "rt")
            else:
                fh = open(peak_path)

            with fh:
                peaks = _parse_peak_file(fh)
        except Exception as e:
            return {"success": False, "error": f"Peak file parsing failed: {e}"}

        if not peaks:
            return {"success": False, "error": "No peaks found in file"}

        # Extract statistics
        widths = [p["width"] for p in peaks]
        signals = [p["signal"] for p in peaks if p["signal"] is not None]
        p_values = [p["p_value"] for p in peaks if p["p_value"] is not None]
        q_values = [p["q_value"] for p in peaks if p["q_value"] is not None]

        chrom_dist: dict[str, int] = {}
        for p in peaks:
            chrom_dist[p["chrom"]] = chrom_dist.get(p["chrom"], 0) + 1

        width_arr = np.array(widths)
        stats: dict[str, Any] = {
            "total_peaks": len(peaks),
            "chromosomes": len(chrom_dist),
            "top_chromosomes": dict(sorted(chrom_dist.items(), key=lambda x: -x[1])[:5]),
            "width": {
                "min": int(width_arr.min()),
                "max": int(width_arr.max()),
                "mean": round(float(width_arr.mean()), 1),
                "median": int(np.median(width_arr)),
                "q25": int(np.percentile(width_arr, 25)),
                "q75": int(np.percentile(width_arr, 75)),
                "n_lt_200bp": int((width_arr < 200).sum()),
                "n_gt_2000bp": int((width_arr > 2000).sum()),
            },
        }

        if signals:
            sig_arr = np.array(signals)
            stats["signal"] = {
                "min": round(float(sig_arr.min()), 2),
                "max": round(float(sig_arr.max()), 2),
                "mean": round(float(sig_arr.mean()), 2),
                "median": round(float(np.median(sig_arr)), 2),
            }

        if q_values:
            q_arr = np.array(q_values)
            stats["q_value"] = {
                "significant_peaks": int((q_arr < 0.05).sum()),
                "significant_pct": round(float((q_arr < 0.05).mean()) * 100, 1),
                "min_log10_q": round(float(-np.log10(q_arr.min())), 2),
            }

        # FRiP calculation (if BAM provided)
        if params.align_path:
            align_path = Path(params.align_path)
            if align_path.exists():
                frip = _calculate_frip(peaks, align_path)
                if frip is not None:
                    stats["frip"] = frip

        # Recommendations
        recommendations = []
        median_width = stats["width"]["median"]
        if median_width < 150:
            recommendations.append("Peak widths are narrow (median < 150bp) — typical for ATAC-seq or TF ChIP-seq")
        elif median_width > 500:
            recommendations.append("Peak widths are broad (median > 500bp) — typical for histone marks")

        if stats.get("frip", {}).get("frip_score", 1.0) < 0.01:
            recommendations.append("FRiP < 1%: Low signal-to-noise ratio, check IP efficiency")

        n_peaks = stats["total_peaks"]
        if n_peaks < 100:
            recommendations.append("Very few peaks (<100): may indicate failed experiment or high stringency")
        elif n_peaks > 100000:
            recommendations.append("Very high peak count (>100K): consider increasing stringency")

        if not recommendations:
            recommendations.append("Peak QC metrics are within normal ranges — proceed with downstream analysis")

        return {
            "success": True,
            "stats": stats,
            "recommendations": recommendations,
            "output_dir": str(output_dir),
        }


def _parse_peak_file(fh: Any) -> list[dict[str, Any]]:
    """Parse narrowPeak/broadPeak/BED format."""
    peaks = []
    for line in fh:
        # Skip track lines and headers
        line = line.strip()
        if not line or line.startswith("track") or line.startswith("#") or line.startswith("browser"):
            continue

        fields = line.split("\t")
        if len(fields) < 3:
            continue

        chrom = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        width = end - start

        name = fields[3] if len(fields) > 3 else "."
        score = int(fields[4]) if len(fields) > 4 and fields[4] != "." else None
        strand = fields[5] if len(fields) > 5 else "."

        signal = None
        p_value = None
        q_value = None

        # narrowPeak/broadPeak have extra columns
        if len(fields) >= 7:
            try:
                signal = float(fields[6]) if fields[6] != "." else None
            except ValueError:
                pass
        if len(fields) >= 8:
            try:
                p_value = float(fields[7]) if fields[7] != "." else None
            except ValueError:
                pass
        if len(fields) >= 9:
            try:
                q_value = float(fields[8]) if fields[8] != "." else None
            except ValueError:
                pass

        peaks.append({
            "chrom": chrom, "start": start, "end": end, "width": width,
            "name": name, "score": score, "strand": strand,
            "signal": signal, "p_value": p_value, "q_value": q_value,
        })

    return peaks


def _calculate_frip(peaks: list[dict], bam_path: Path) -> dict[str, Any] | None:
    """Estimate FRiP using peak coverage approximation from BAM index."""
    try:
        import pysam
    except ImportError:
        return None

    try:
        bam = pysam.AlignmentFile(str(bam_path), "rb")
        total_reads = bam.mapped + bam.unmapped

        reads_in_peaks = 0
        for peak in peaks:
            reads_in_peaks += bam.count(peak["chrom"], peak["start"], peak["end"])

        frip = reads_in_peaks / total_reads if total_reads > 0 else 0.0
        bam.close()

        return {
            "total_reads": total_reads,
            "reads_in_peaks": reads_in_peaks,
            "frip_score": round(frip, 4),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Tool 2: Methylation Analysis
# ═══════════════════════════════════════════════════════════════════


class MethylationAnalysisParams(BaseModel):
    """Parameters for methylation data analysis."""

    data_path: str = Field(..., description="Path to methylation data (beta matrix TSV or Bismark .cov)")
    metadata_path: str = Field(default="", description="Optional sample metadata for group comparison")
    group_col: str = Field(default="", description="Column name for group comparison")
    output_dir: str = Field(default="./output", description="Output directory")


class MethylationAnalysisTool(CallableTool[MethylationAnalysisParams]):
    """Analyze DNA methylation data from beta-value matrices or Bismark output.

    Reports: beta value distribution, differentially methylated CpGs,
    chromosome-level methylation patterns, and QC metrics.
    """

    name = "methylation_analysis"
    description = (
        "Analyze DNA methylation data including beta-value distribution, "
        "differentially methylated CpG detection, chromosome-level patterns, "
        "and quality metrics. Accepts beta-value matrices (TSV/CSV) or "
        "Bismark coverage files. Supports group comparison for DMR detection."
    )
    params_model = MethylationAnalysisParams

    async def execute(self, params: MethylationAnalysisParams, **kwargs: Any) -> dict[str, Any]:
        import pandas as pd

        data_path = Path(params.data_path)
        if not data_path.exists():
            return {"success": False, "error": f"Data file not found: {params.data_path}"}

        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = _read_methylation_data(data_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to read methylation data: {e}"}

        # Detect format: beta matrix (CpG × samples) or Bismark cov
        beta_cols = [c for c in df.columns if _is_beta_column(df[c])]
        is_beta_matrix = len(beta_cols) > 1

        if is_beta_matrix:
            return self._analyze_beta_matrix(df, beta_cols, params, output_dir)
        else:
            return self._analyze_bismark_cov(df, output_dir)

    def _analyze_beta_matrix(
        self, df: Any, beta_cols: list[str], params: MethylationAnalysisParams, output_dir: Path
    ) -> dict[str, Any]:
        import pandas as pd

        beta = df[beta_cols]
        n_cpgs, n_samples = beta.shape

        # Global methylation statistics
        global_mean = float(beta.mean().mean())
        global_std = float(beta.std().mean())

        # By sample
        sample_means = beta.mean().to_dict()
        sample_stds = beta.std().to_dict()

        # Beta distribution bins
        bins = {"hypomethylated (<0.2)": 0, "low (0.2-0.4)": 0, "intermediate (0.4-0.6)": 0,
                "high (0.6-0.8)": 0, "hypermethylated (>0.8)": 0}
        flat_beta = beta.values.flatten()
        flat_beta = flat_beta[~np.isnan(flat_beta)]
        bins["hypomethylated (<0.2)"] = int((flat_beta < 0.2).sum())
        bins["low (0.2-0.4)"] = int(((flat_beta >= 0.2) & (flat_beta < 0.4)).sum())
        bins["intermediate (0.4-0.6)"] = int(((flat_beta >= 0.4) & (flat_beta < 0.6)).sum())
        bins["high (0.6-0.8)"] = int(((flat_beta >= 0.6) & (flat_beta < 0.8)).sum())
        bins["hypermethylated (>0.8)"] = int((flat_beta >= 0.8).sum())

        total = sum(bins.values())
        bin_pcts = {k: round(v / total * 100, 1) for k, v in bins.items()} if total else {}

        result: dict[str, Any] = {
            "format": "beta_matrix",
            "n_cpgs": n_cpgs,
            "n_samples": n_samples,
            "global": {
                "mean_beta": round(global_mean, 4),
                "std_beta": round(global_std, 4),
            },
            "distribution": bin_pcts,
            "sample_stats": {
                "mean_range": [round(min(sample_means.values()), 4), round(max(sample_means.values()), 4)],
                "std_range": [round(min(sample_stds.values()), 4), round(max(sample_stds.values()), 4)],
            },
        }

        # Group comparison (simple differential methylation)
        if params.group_col and params.metadata_path:
            meta_path = Path(params.metadata_path)
            if meta_path.exists():
                try:
                    meta_df = pd.read_csv(meta_path, sep=None if meta_path.suffix == ".csv" else "\t", index_col=0)
                    if params.group_col in meta_df.columns:
                        groups = meta_df[params.group_col].dropna().unique()
                        if len(groups) == 2:
                            g1_cols = meta_df.index[meta_df[params.group_col] == groups[0]].intersection(beta.columns).tolist()
                            g2_cols = meta_df.index[meta_df[params.group_col] == groups[1]].intersection(beta.columns).tolist()

                            if g1_cols and g2_cols:
                                g1_mean = beta[g1_cols].mean(axis=1)
                                g2_mean = beta[g2_cols].mean(axis=1)
                                delta = (g2_mean - g1_mean).dropna()

                                dm_counts = {
                                    "hyper": int((delta > 0.2).sum()),
                                    "hypo": int((delta < -0.2).sum()),
                                    "total_dm": int((abs(delta) > 0.2).sum()),
                                }
                                result["differential_methylation"] = dm_counts
                                result["dm_threshold"] = 0.2
                except Exception as e:
                    result["dm_error"] = str(e)

        # Recommendations
        recommendations = []
        if global_mean < 0.3:
            recommendations.append("Global methylation is low — typical for CpG-poor regions or specific tissues")
        elif global_mean > 0.7:
            recommendations.append("Global methylation is high — verify bisulfite conversion efficiency")

        if result.get("differential_methylation", {}).get("total_dm", 0) > 1000:
            recommendations.append("Large number of DMCs detected — consider DMR aggregation")

        return {
            "success": True,
            "stats": result,
            "recommendations": recommendations,
            "output_dir": str(output_dir),
        }

    def _analyze_bismark_cov(self, df: Any, output_dir: Path) -> dict[str, Any]:
        """Analyze Bismark coverage file."""
        # Bismark cov: chrom, start, end, methylation_pct, count_methylated, count_unmethylated
        if len(df.columns) >= 5:
            pct_col = df.columns[3]
            meth_pct = df[pct_col].dropna()

            return {
                "success": True,
                "format": "bismark_cov",
                "n_cpgs": len(df),
                "stats": {
                    "mean_methylation": round(float(meth_pct.mean()), 2),
                    "median_methylation": round(float(meth_pct.median()), 2),
                    "std_methylation": round(float(meth_pct.std()), 2),
                },
                "recommendations": [
                    "Bismark coverage file analysis provides per-CpG methylation levels",
                    "For DMR detection, consider using DSS or bumphunter via R",
                ],
                "output_dir": str(output_dir),
            }

        return {"success": False, "error": "Unrecognized Bismark format — expected 5+ columns"}


def _read_methylation_data(path: Path) -> Any:
    """Read methylation data in various formats."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh, sep=None)
    elif suffix in (".csv",):
        return pd.read_csv(path)
    elif suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    else:
        return pd.read_csv(path, sep=None)


def _is_beta_column(col: Any) -> bool:
    """Check if a column contains beta values (0-1 range)."""
    try:
        vals = col.dropna()
        if len(vals) < 2:
            return False
        mn, mx = vals.min(), vals.max()
        if mn < -0.01 or mx > 1.01:
            return False
        return float(mn) >= 0 and float(mx) <= 1
    except Exception:
        return False
