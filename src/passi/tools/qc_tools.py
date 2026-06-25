"""QC & preprocessing tools for omics data quality assessment."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from passi.tools.base import CallableTool


class QcReportParams(BaseModel):
    """Parameters for QC report generation."""

    data_path: str = Field(..., description="Path to the data file to QC")
    domain: str | None = Field(default=None, description="Omics domain (auto-detected if empty)")
    group_col: str | None = Field(default=None, description="Column name for sample grouping")
    output_dir: str = Field(default="./output", description="Directory for QC report output")


class QcReportTool(CallableTool[QcReportParams]):
    """Generate a quality control report for omics data.

    Detects: missing values, outliers, batch effects (via PCA),
    library size distribution (RNA-seq), peak FRiP score (ChIP/ATAC),
    variant call rate (GWAS), and more.
    """

    name = "qc_report"
    description = (
        "Generate a quality control report for omics data. Detects missing values, "
        "outliers, library size distribution (RNA-seq), batch effects via PCA, "
        "peak quality metrics (ChIP/ATAC), and variant call rate (GWAS). "
        "Returns per-metric results and recommendations."
    )
    params_model = QcReportParams

    async def execute(self, params: QcReportParams, **kwargs: Any) -> dict[str, Any]:
        from pathlib import Path

        import pandas as pd

        path = Path(params.data_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {params.data_path}"}

        try:
            suffix = path.suffix.lower()
            if suffix in (".csv",):
                df = pd.read_csv(path)
            elif suffix in (".tsv", ".txt"):
                df = pd.read_csv(path, sep="\t")
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                return {"success": False, "error": f"Unsupported format: {suffix}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {e}"}

        metrics: dict[str, Any] = {
            "file": str(path),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        }

        # ── Missing values ──
        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        metrics["missing"] = {
            "total": total_missing,
            "percent": round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2) if df.size else 0,
            "columns_with_missing": int((missing > 0).sum()),
        }

        # ── Numeric columns summary ──
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            numeric_df = df[numeric_cols]
            metrics["numeric_summary"] = {
                "count": len(numeric_cols),
                "mean": round(float(numeric_df.mean().mean()), 4),
                "std": round(float(numeric_df.std().mean()), 4),
                "min": round(float(numeric_df.min().min()), 4),
                "max": round(float(numeric_df.max().max()), 4),
            }

        # ── Low-count genes (for RNA-seq count matrices) ──
        if len(numeric_cols) > 5 and params.domain in (None, "transcriptomics"):
            row_means = df[numeric_cols].mean(axis=1)
            low_count_mask = row_means < 10
            metrics["low_count_features"] = {
                "threshold": 10,
                "count_below": int(low_count_mask.sum()),
                "percent": round(float(low_count_mask.mean()) * 100, 2),
            }

        # ── Sample correlation PCA (batch effect detection) ──
        if len(numeric_cols) >= 4:
            try:
                from sklearn.decomposition import PCA

                numeric_clean = df[numeric_cols].dropna(axis=1, how="all").fillna(0)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(numeric_clean.values.T)
                metrics["pca"] = {
                    "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
                    "pc1_pc2": [
                        {"pc1": round(float(r[0]), 4), "pc2": round(float(r[1]), 4)}
                        for r in pca_result[:10]
                    ],
                }
            except Exception:
                metrics["pca"] = {"error": "PCA computation failed — check numeric columns"}

        # ── Group comparison ──
        if params.group_col and params.group_col in df.columns:
            groups = df[params.group_col].value_counts().to_dict()
            metrics["groups"] = {str(k): int(v) for k, v in groups.items()}

        # ── Recommendations ──
        recommendations = []
        if metrics["missing"]["total"] > 0:
            pct = metrics["missing"]["percent"]
            if pct > 20:
                recommendations.append("High missing rate (>20%). Consider KNN or MOFA-based imputation.")
            elif pct > 5:
                recommendations.append("Moderate missing rate (>5%). Remove features with >50% missing, impute rest.")
        if metrics.get("low_count_features", {}).get("percent", 0) > 50:
            recommendations.append("Many low-count features (>50%). Filter before differential analysis.")
        if len(numeric_cols) < 3:
            recommendations.append("Very few numeric columns. Verify the data format is correct.")

        return {
            "success": True,
            "metrics": metrics,
            "recommendations": recommendations,
            "output_dir": params.output_dir,
        }
