"""Transcriptomics analysis tools — differential expression, GSEA, WGCNA.

DESeq2 / edgeR / limma wrappers execute R code via rpy2 bridge.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from passi.tools.base import CallableTool

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# R code templates
# ═══════════════════════════════════════════════════════════════════

DESEQ2_SCRIPT = r"""
suppressMessages(library(DESeq2))

# Read inputs
counts <- read.table("{counts_path}", header=TRUE, row.names=1, sep="\t", check.names=FALSE)
if("{metadata_path}" != "") {{
    metadata <- read.table("{metadata_path}", header=TRUE, row.names=1, sep="\t", check.names=FALSE)
    metadata${group_col} <- factor(metadata${group_col})
    common_samples <- intersect(colnames(counts), rownames(metadata))
    counts <- counts[, common_samples, drop=FALSE]
    metadata <- metadata[common_samples, , drop=FALSE]
}} else {{
    stop("metadata_path is required for DESeq2")
}}

# DESeq2
dds <- DESeqDataSetFromMatrix(
    countData = counts,
    colData = metadata,
    design = as.formula(paste("~", "{group_col}"))
)
dds <- DESeq2::DESeq(dds)
res <- DESeq2::results(dds, alpha={alpha})
res <- as.data.frame(res[order(res$padj), ])

# Summary
up <- sum(res$padj < {alpha} & res$log2FoldChange > 0, na.rm=TRUE)
down <- sum(res$padj < {alpha} & res$log2FoldChange < 0, na.rm=TRUE)
cat(sprintf("DESeq2_DONE|up=%d|down=%d|total_genes=%d\\n", up, down, nrow(res)))
write.table(res, file="{output_path}", sep="\t", quote=FALSE, col.names=NA)
"""

EDGER_SCRIPT = r"""
suppressMessages(library(edgeR))

counts <- read.table("{counts_path}", header=TRUE, row.names=1, sep="\t", check.names=FALSE)
metadata <- read.table("{metadata_path}", header=TRUE, row.names=1, sep="\t", check.names=FALSE)
metadata${group_col} <- as.factor(metadata${group_col})

common_samples <- intersect(colnames(counts), rownames(metadata))
counts <- counts[, common_samples, drop=FALSE]
metadata <- metadata[common_samples, , drop=FALSE]

group <- metadata${group_col}
design <- model.matrix(~ group)

y <- DGEList(counts=counts, group=group)
keep <- filterByExpr(y, design=design)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)
qlf <- glmQLFTest(fit, coef=2)
res <- as.data.frame(topTags(qlf, n=Inf))

up <- sum(res$FDR < {alpha} & res$logFC > 0, na.rm=TRUE)
down <- sum(res$FDR < {alpha} & res$logFC < 0, na.rm=TRUE)
cat(sprintf("edgeR_DONE|up=%d|down=%d|total_genes=%d\\n", up, down, nrow(res)))
write.table(res, file="{output_path}", sep="\t", quote=FALSE, col.names=NA)
"""

LIMMA_SCRIPT = r"""
suppressMessages(library(limma))
suppressMessages(library(edgeR))

counts <- read.table("{counts_path}", header=TRUE, row.names=1, sep="\t", check.names=FALSE)
metadata <- read.table("{metadata_path}", header=TRUE, row.names=1, sep="\t", check.names=FALSE)
metadata${group_col} <- as.factor(metadata${group_col})

common_samples <- intersect(colnames(counts), rownames(metadata))
counts <- counts[, common_samples, drop=FALSE]
metadata <- metadata[common_samples, , drop=FALSE]

group <- metadata${group_col}
design <- model.matrix(~ group)

dge <- DGEList(counts=counts)
keep <- filterByExpr(dge, design=design)
dge <- dge[keep, , keep.lib.sizes=FALSE]
dge <- calcNormFactors(dge)

v <- voom(dge, design)
fit <- lmFit(v, design)
fit <- eBayes(fit)
res <- as.data.frame(topTable(fit, coef=2, number=Inf))

up <- sum(res$adj.P.Val < {alpha} & res$logFC > 0, na.rm=TRUE)
down <- sum(res$adj.P.Val < {alpha} & res$logFC < 0, na.rm=TRUE)
cat(sprintf("limma_DONE|up=%d|down=%d|total_genes=%d\\n", up, down, nrow(res)))
write.table(res, file="{output_path}", sep="\t", quote=FALSE, col.names=NA)
"""

METHOD_SCRIPTS = {
    "deseq2": DESEQ2_SCRIPT,
    "edger": EDGER_SCRIPT,
    "limma": LIMMA_SCRIPT,
}


# ═══════════════════════════════════════════════════════════════════
# Tool
# ═══════════════════════════════════════════════════════════════════


class DifferentialAnalysisParams(BaseModel):
    """Parameters for differential expression analysis."""

    counts_path: str = Field(..., description="Path to count matrix (TSV, genes × samples)")
    metadata_path: str = Field(..., description="Path to sample metadata (TSV, samples × columns)")
    group_col: str = Field(..., description="Column name in metadata for group comparison")
    method: str = Field(
        default="deseq2",
        description="Method: 'deseq2' (count data), 'edger' (count data), or 'limma' (normalized data)",
    )
    alpha: float = Field(default=0.05, description="Significance threshold (FDR)")
    output_dir: str = Field(default="./output", description="Output directory for results")


class DifferentialAnalysisTool(CallableTool[DifferentialAnalysisParams]):
    """Differential expression analysis via DESeq2 / edgeR / limma.

    Accepts a count matrix + metadata table, executes the chosen R method
    via rpy2 (or Rscript fallback), and returns the results table with
    log2FoldChange, p-value, and adjusted p-value.
    """

    name = "differential_analysis"
    description = (
        "Run differential expression analysis on a count matrix. "
        "Supports DESeq2 (recommended for raw counts), edgeR (raw counts), "
        "and limma-voom (normalized data). Requires a metadata file with group labels. "
        "Returns significantly differentially expressed genes, fold changes, and p-values."
    )
    params_model = DifferentialAnalysisParams

    def __init__(self, r_home: str = "", r_lib_path: str = "", r_path: str = "Rscript") -> None:
        self.r_home = r_home
        self.r_lib_path = r_lib_path
        self.r_path = r_path

    async def execute(self, params: DifferentialAnalysisParams, **kwargs: Any) -> dict[str, Any]:
        from pathlib import Path

        from passi.executors.r_executor import init_rpy2

        # Validate inputs
        counts_path = Path(params.counts_path)
        if not counts_path.exists():
            return {"success": False, "error": f"Counts file not found: {params.counts_path}"}
        if not Path(params.metadata_path).exists():
            return {"success": False, "error": f"Metadata file not found: {params.metadata_path}"}

        method = params.method.lower()
        if method not in METHOD_SCRIPTS:
            return {"success": False, "error": f"Unknown method: {method}. Choose: {list(METHOD_SCRIPTS)}"}

        # Ensure output directory
        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"de_results_{method}_{counts_path.stem}.tsv"

        # Build R script
        script = METHOD_SCRIPTS[method].format(
            counts_path=str(counts_path.resolve()).replace("\\", "/"),
            metadata_path=str(Path(params.metadata_path).resolve()).replace("\\", "/"),
            group_col=params.group_col,
            alpha=params.alpha,
            output_path=str(output_path.resolve()).replace("\\", "/"),
        )

        # Execute via rpy2 or Rscript fallback
        status = init_rpy2(self.r_home, self.r_lib_path)
        if status["ready"]:
            try:
                import rpy2.robjects as ro
                from rpy2.robjects.conversion import localconverter
                from rpy2.robjects import numpy2ri, pandas2ri

                with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                    ro.r(script)

                # Try to read R's printed output for the summary line
                return self._parse_output(method, output_path)
            except Exception as e:
                logger.warning("rpy2 execution failed, falling back to Rscript: %s", e)

        # Rscript fallback
        return self._execute_via_rscript(script, method, output_path)

    def _execute_via_rscript(self, script: str, method: str, output_path: Path) -> dict[str, Any]:
        import os
        import subprocess
        import tempfile

        rscript = self.r_path or "Rscript"
        if self.r_home and not os.path.isabs(rscript):
            home = Path(self.r_home)
            for subpath in ("bin/Rscript.exe", "bin/Rscript"):
                exe = home / subpath
                if exe.exists():
                    rscript = str(exe)
                    break

        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, encoding="utf-8") as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [rscript, "--no-save", script_path],
                capture_output=True, text=True, timeout=600,
                cwd=str(Path.cwd()),
            )
            return self._parse_output(method, output_path, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Differential analysis timed out (600s)"}
        except FileNotFoundError:
            return {"success": False, "error": f"Rscript not found: {rscript}"}
        finally:
            Path(script_path).unlink(missing_ok=True)

    def _parse_output(
        self, method: str, output_path: Path, stdout: str = "", stderr: str = ""
    ) -> dict[str, Any]:
        if not output_path.exists():
            return {
                "success": False,
                "error": f"Output file not created: {output_path}",
                "stderr": stderr[-2000:],
            }

        try:
            import pandas as pd

            df = pd.read_csv(output_path, sep="\t", index_col=0, nrows=5001)
            if len(df) > 5000:
                df = df.head(5000)

            result: dict[str, Any] = {
                "success": True,
                "method": method,
                "output_file": str(output_path),
                "total_genes": len(df),
            }

            # Determine column names based on method
            if method == "deseq2":
                padj_col = "padj"
                fc_col = "log2FoldChange"
            elif method == "edger":
                padj_col = "FDR"
                fc_col = "logFC"
            else:  # limma
                padj_col = "adj.P.Val"
                fc_col = "logFC"

            if padj_col in df.columns:
                alpha = 0.05
                sig = df[df[padj_col].notna() & (df[padj_col] < alpha)]
                up = int((sig[fc_col] > 0).sum()) if fc_col in sig.columns else 0
                down = int((sig[fc_col] < 0).sum()) if fc_col in sig.columns else 0
                result["up_regulated"] = up
                result["down_regulated"] = down
                result["significant"] = up + down

            # Top genes preview
            top_cols = [c for c in [fc_col, padj_col, "pvalue", "stat"] if c in df.columns]
            if top_cols:
                result["top_genes"] = df[top_cols].head(10).reset_index().to_dict(orient="records")

            return result
        except Exception as e:
            return {"success": True, "method": method, "output_file": str(output_path), "parse_warning": str(e)}
