"""Gene set enrichment analysis tools 鈥?GSEA (fgsea) and ORA (clusterProfiler).

Supports preranked GSEA and over-representation analysis for GO, KEGG,
and MSigDB gene sets. Executes via rpy2 or Rscript fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from passi.tools.base import CallableTool

logger = logging.getLogger(__name__)

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# R code templates
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
FGSEA_SCRIPT = r"""
suppressMessages(library(fgsea))
suppressMessages(library(jsonlite))

# Read ranked gene list (two-column: gene_id, ranking_metric)
ranked <- read.table("{ranked_path}", header=FALSE, sep="\t", stringsAsFactors=FALSE)
colnames(ranked) <- c("gene", "rank_val")
ranks <- setNames(ranked$rank_val, ranked$gene)
ranks <- sort(ranks, decreasing=TRUE)

# Read or generate gene sets
if ("{gmt_path}" != "") {{
    pathways <- gmtPathways("{gmt_path}")
    cat(sprintf("Read %d gene sets from GMT\n", length(pathways)))
}} else {{
    # Use built-in example pathways (Hallmark)
    data("examplePathways", package="fgsea")
    pathways <- examplePathways
    cat(sprintf("Using %d example pathways\n", length(pathways)))
}}

# Run fgsea
fgsea_res <- fgsea(
    pathways = pathways,
    stats = ranks,
    minSize = {min_size},
    maxSize = {max_size},
    nperm = {n_perm},
    nproc = 1
)

fgsea_res <- as.data.frame(fgsea_res[order(fgsea_res$padj), ])
fgsea_res$leadingEdge <- sapply(fgsea_res$leadingEdge, paste, collapse=",")

up <- sum(fgsea_res$padj < {alpha} & fgsea_res$NES > 0, na.rm=TRUE)
down <- sum(fgsea_res$padj < {alpha} & fgsea_res$NES < 0, na.rm=TRUE)
cat(sprintf("FGSEA_DONE|pathways=%d|up=%d|down=%d\n", nrow(fgsea_res), up, down))

# Write results
write.table(fgsea_res, file="{output_path}", sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)

# Write summary JSON
result <- list(
    method = "fgsea",
    total_pathways = nrow(fgsea_res),
    up_regulated = up,
    down_regulated = down,
    top_pathways = head(fgsea_res[fgsea_res$padj < {alpha}, c("pathway","pval","padj","NES","size")], 20)
)
writeLines(toJSON(result, auto_unbox=TRUE, pretty=TRUE), "{output_json}")
"""

ORA_SCRIPT = r"""
suppressMessages(library(clusterProfiler))
suppressMessages(library(jsonlite))

# Read DE results (must have gene_id and pvalue/padj columns)
de_data <- read.table("{de_path}", header=TRUE, sep="\t", stringsAsFactors=FALSE)

# Determine gene column and significance column
gene_col <- "{gene_col}"
p_col <- "{p_col}"

if (!(gene_col %in% colnames(de_data))) stop(paste("Gene column not found:", gene_col))
if (!(p_col %in% colnames(de_data))) stop(paste("P-value column not found:", p_col))

# Select significant genes
sig_genes <- de_data[de_data[[p_col]] < {alpha} & !is.na(de_data[[p_col]]), gene_col]
# Universe: all tested genes
universe <- de_data[!is.na(de_data[[p_col]]), gene_col]

cat(sprintf("ORA: %d significant genes out of %d tested\n", length(sig_genes), length(universe)))

# Determine gene ID type
gene_id_type <- "{gene_id_type}"

# GO enrichment (if requested)
go_results <- NULL
kegg_results <- NULL

if ("{ontologies}" != "") {{
    onts <- strsplit("{ontologies}", ",")[[1]]
    go_list <- list()
    for (ont in trimws(onts)) {{
        tryCatch({{
            ego <- enrichGO(
                gene = sig_genes,
                universe = universe,
                OrgDb = org.Hs.eg.db,
                keyType = gene_id_type,
                ont = ont,
                pAdjustMethod = "BH",
                pvalueCutoff = {alpha},
                qvalueCutoff = {alpha}
            )
            if (nrow(as.data.frame(ego)) > 0) {{
                go_list[[ont]] <- as.data.frame(ego)
                cat(sprintf("GO %s: %d enriched terms\n", ont, nrow(go_list[[ont]])))
            }}
        }}, error = function(e) {{
            cat(sprintf("GO %s failed: %s\n", ont, e$message))
        }})
    }}
    if (length(go_list) > 0) {{
        go_results <- do.call(rbind, lapply(names(go_list), function(n) {{
            df <- go_list[[n]]
            df$ontology <- n
            df
        }}))
    }}
}}

# KEGG enrichment (if requested)
if ("{do_kegg}" == "true") {{
    tryCatch({{
        # Convert gene IDs to Entrez if needed
        if (gene_id_type != "ENTREZID") {{
            entrez_genes <- bitr(sig_genes, fromType=gene_id_type, toType="ENTREZID", OrgDb=org.Hs.eg.db)
            entrez_univ <- bitr(universe, fromType=gene_id_type, toType="ENTREZID", OrgDb=org.Hs.eg.db)
            kk <- enrichKEGG(
                gene = entrez_genes$ENTREZID,
                universe = entrez_univ$ENTREZID,
                pAdjustMethod = "BH",
                pvalueCutoff = {alpha},
                qvalueCutoff = {alpha}
            )
        }} else {{
            kk <- enrichKEGG(
                gene = sig_genes,
                universe = universe,
                pAdjustMethod = "BH",
                pvalueCutoff = {alpha},
                qvalueCutoff = {alpha}
            )
        }}
        kegg_results <- as.data.frame(kk)
        cat(sprintf("KEGG: %d enriched pathways\n", nrow(kegg_results)))
    }}, error = function(e) {{
        cat(sprintf("KEGG failed: %s\n", e$message))
    }})
}}

cat(sprintf("ORA_DONE|go_terms=%d|kegg_pathways=%d\n",
    if (is.null(go_results)) 0 else nrow(go_results),
    if (is.null(kegg_results)) 0 else nrow(kegg_results)))

# Write results
combined <- list()
if (!is.null(go_results)) {{
    write.table(go_results, file="{output_go}", sep="\t", quote=FALSE, row.names=FALSE)
    combined$go <- head(go_results[, c("ontology","ID","Description","GeneRatio","BgRatio","pvalue","p.adjust","qvalue","Count")], 20)
}}
if (!is.null(kegg_results)) {{
    write.table(kegg_results, file="{output_kegg}", sep="\t", quote=FALSE, row.names=FALSE)
    combined$kegg <- head(kegg_results[, c("ID","Description","GeneRatio","BgRatio","pvalue","p.adjust","qvalue","Count")], 20)
}}

result <- list(
    method = "ora",
    n_significant_genes = length(sig_genes),
    n_universe = length(universe),
    go_terms = if (is.null(go_results)) 0 else nrow(go_results),
    kegg_pathways = if (is.null(kegg_results)) 0 else nrow(kegg_results),
    top_terms = combined
)
writeLines(toJSON(result, auto_unbox=TRUE, pretty=TRUE), "{output_json}")
"""

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Tool definitions
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class EnrichmentParams(BaseModel):
    """Parameters for gene set enrichment analysis."""

    method: str = Field(
        default="fgsea",
        description="Method: 'fgsea' (preranked GSEA) or 'ora' (over-representation analysis)",
    )
    ranked_path: str = Field(
        default="",
        description="Path to ranked gene list for fgsea (TSV: gene_id, ranking_value). Only for fgsea.",
    )
    gmt_path: str = Field(
        default="",
        description="Path to GMT gene set file for fgsea. Uses built-in example pathways if empty.",
    )
    de_path: str = Field(
        default="",
        description="Path to differential expression results for ORA (must have gene_id and p-value columns). Only for ORA.",
    )
    gene_col: str = Field(
        default="gene",
        description="Column name for gene identifiers in DE results (ORA only)",
    )
    p_col: str = Field(
        default="padj",
        description="Column name for adjusted p-value in DE results (ORA only)",
    )
    gene_id_type: str = Field(
        default="ENSEMBL",
        description="Gene ID type for ORA: ENSEMBL, SYMBOL, ENTREZID, UNIPROT",
    )
    ontologies: str = Field(
        default="BP,MF,CC",
        description="Comma-separated GO ontologies for ORA: BP, MF, CC",
    )
    do_kegg: bool = Field(default=True, description="Run KEGG enrichment (ORA only)")
    alpha: float = Field(default=0.05, description="Significance threshold (FDR)")
    min_size: int = Field(default=15, description="Minimum gene set size (fgsea)")
    max_size: int = Field(default=500, description="Maximum gene set size (fgsea)")
    n_perm: int = Field(default=10000, description="Number of permutations (fgsea)")
    output_dir: str = Field(default="./result", description="Output directory for results")


class EnrichmentTool(CallableTool[EnrichmentParams]):
    """Gene set enrichment analysis via fgsea (preranked) or clusterProfiler (ORA).

    Run GSEA on a ranked gene list or over-representation analysis on
    a set of differentially expressed genes. GO, KEGG, and custom gene
    set databases supported.
    """

    name = "enrichment_analysis"
    description = (
        "Run gene set enrichment analysis on omics results. "
        "Supports preranked GSEA (fgsea) 鈥?provide a ranked gene list and optionally a GMT file. "
        "Also supports over-representation analysis (ORA) via clusterProfiler 鈥?provide DE results "
        "with gene IDs and p-values. Returns enriched pathways with statistics."
    )
    params_model = EnrichmentParams

    def __init__(self, r_home: str = "", r_lib_path: str = "", r_path: str = "Rscript") -> None:
        self.r_home = r_home
        self.r_lib_path = r_lib_path
        self.r_path = r_path

    async def execute(self, params: EnrichmentParams, **kwargs: Any) -> dict[str, Any]:
        import json

        from passi.executors.r_executor import init_rpy2

        method = params.method.lower()
        if method not in ("fgsea", "ora"):
            return {"success": False, "error": f"Unknown method: {method}. Choose: fgsea, ora"}

        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if method == "fgsea":
            ranked_path = Path(params.ranked_path) if params.ranked_path else None
            if ranked_path and not ranked_path.exists():
                return {"success": False, "error": f"Ranked gene list not found: {params.ranked_path}"}
            if not ranked_path:
                return {"success": False, "error": "ranked_path is required for fgsea"}

            gmt_path = ""
            if params.gmt_path:
                gmt_p = Path(params.gmt_path)
                if not gmt_p.exists():
                    return {"success": False, "error": f"GMT file not found: {params.gmt_path}"}
                gmt_path = str(gmt_p.resolve()).replace("\\", "/")

            output_path = output_dir / "fgsea_results.tsv"
            output_json = output_dir / "fgsea_summary.json"

            script = FGSEA_SCRIPT.format(
                ranked_path=str(ranked_path.resolve()).replace("\\", "/"),
                gmt_path=gmt_path,
                min_size=params.min_size,
                max_size=params.max_size,
                n_perm=params.n_perm,
                alpha=params.alpha,
                output_path=str(output_path.resolve()).replace("\\", "/"),
                output_json=str(output_json.resolve()).replace("\\", "/"),
            )

        else:  # ora
            if not params.de_path:
                return {"success": False, "error": "de_path is required for ORA"}
            de_path = Path(params.de_path)
            if not de_path.exists():
                return {"success": False, "error": f"DE results file not found: {params.de_path}"}

            output_go = output_dir / "go_enrichment.tsv"
            output_kegg = output_dir / "kegg_enrichment.tsv"
            output_json = output_dir / "ora_summary.json"

            script = ORA_SCRIPT.format(
                de_path=str(de_path.resolve()).replace("\\", "/"),
                gene_col=params.gene_col,
                p_col=params.p_col,
                gene_id_type=params.gene_id_type,
                ontologies=params.ontologies if params.ontologies else "",
                do_kegg="true" if params.do_kegg else "false",
                alpha=params.alpha,
                output_go=str(output_go.resolve()).replace("\\", "/"),
                output_kegg=str(output_kegg.resolve()).replace("\\", "/"),
                output_json=str(output_json.resolve()).replace("\\", "/"),
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

                return self._parse_output(method, output_json, output_dir)
            except Exception as e:
                logger.warning("rpy2 execution failed, falling back to Rscript: %s", e)

        return self._execute_via_rscript(script, method, output_json, output_dir)

    def _execute_via_rscript(self, script: str, method: str, output_json: Path, output_dir: Path) -> dict[str, Any]:
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
                cwd=str(os.getcwd()),
            )
            return self._parse_output(method, output_json, output_dir, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Enrichment analysis timed out (600s)"}
        except FileNotFoundError:
            return {"success": False, "error": f"Rscript not found: {rscript}"}
        finally:
            Path(script_path).unlink(missing_ok=True)

    def _parse_output(
        self, method: str, output_json: Path, output_dir: Path,
        stdout: str = "", stderr: str = "",
    ) -> dict[str, Any]:
        import json

        result: dict[str, Any] = {
            "success": True,
            "method": method,
            "output_dir": str(output_dir),
        }

        # Parse summary from stdout
        for line in stdout.splitlines():
            if "FGSEA_DONE" in line or "ORA_DONE" in line:
                for part in line.split("|")[1:]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        try:
                            result[k] = int(v)
                        except ValueError:
                            result[k] = v

        # Read JSON summary
        if output_json.exists():
            try:
                with open(output_json, encoding="utf-8") as f:
                    json_data = json.load(f)
                result["summary"] = json_data
                for key in ("up_regulated", "down_regulated", "total_pathways",
                           "go_terms", "kegg_pathways", "n_significant_genes"):
                    if key in json_data and key not in result:
                        result[key] = json_data[key]
            except (json.JSONDecodeError, Exception) as e:
                result["parse_warning"] = str(e)

        return result
