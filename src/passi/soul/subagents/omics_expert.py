"""Omics Expert sub-agent for single-omics domain-specific analysis.

Handles transcriptomics, genomics, epigenetics, proteomics, and metabolomics
analysis with domain-specific knowledge and tool selection.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

OMICS_EXPERT_PROMPT = """You are the Omics Expert, a specialized bioinformatics sub-agent.

Your expertise covers single-omics analysis across:
- **Transcriptomics**: differential expression (DESeq2, edgeR, limma), gene set enrichment (GSEA, ORA), WGCNA
- **Genomics**: GWAS (PLINK), variant annotation, CNV calling
- **Epigenetics**: peak calling QC, differential binding (DiffBind), DMR calling, motif enrichment
- **Proteomics**: differential protein abundance, PTM analysis, pathway enrichment
- **Metabolomics**: differential metabolite abundance, pathway mapping

## Workflow
1. Identify the omics domain from the data format and user intent
2. Recommend appropriate methods and parameters
3. Execute analysis with proper QC checks
4. Interpret results in biological context

## Guidelines
- Always verify data quality before running analysis
- Use R/Bioconductor for established methods, Python for custom analysis
- Report effect sizes and statistical significance, not just p-values
- Flag potential batch effects, outliers, and confounders
"""

# Domain -> default analysis task mapping
DOMAIN_DEFAULTS: dict[str, dict[str, str]] = {
    "transcriptomics": {
        "qc": "qc_report",
        "de": "differential_analysis",
        "explore": "qc_report",
    },
    "genomics": {
        "qc": "qc_report",
        "explore": "qc_report",
    },
    "epigenetics": {
        "qc": "qc_report",
        "explore": "qc_report",
    },
    "proteomics": {
        "qc": "qc_report",
        "explore": "qc_report",
    },
    "metabolomics": {
        "qc": "qc_report",
        "explore": "qc_report",
    },
}


class OmicsExpert:
    """Sub-agent specialized in single-omics analysis workflows.

    Runs in an isolated context; only analysis results flow back to the main agent.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._system_prompt = OMICS_EXPERT_PROMPT
        self._tool_registry = None

    async def _ensure_registry(self) -> Any:
        """Lazy-load the tool registry from the runtime."""
        if self._tool_registry is None:
            from passi.tools.registry import ToolRegistry
            from passi.tools.io_tools import ParseOmicsDataTool, ReadFileTool
            from passi.tools.qc_tools import QcReportTool
            from passi.tools.transcriptomics_tools import DifferentialAnalysisTool

            exec_cfg = self.runtime.config.execution
            registry = ToolRegistry()
            registry.register(ReadFileTool(), "io")
            registry.register(ParseOmicsDataTool(), "io")
            registry.register(QcReportTool(), "qc")

            de_tool = DifferentialAnalysisTool(
                r_home=exec_cfg.r_home or "",
                r_lib_path=exec_cfg.r_lib_path or "",
                r_path=exec_cfg.rscript_binary,
            )
            registry.register(de_tool, "transcriptomics")
            self._tool_registry = registry
        return self._tool_registry

    async def analyze(
        self,
        data_path: str,
        domain: str,
        task: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single-omics analysis task.

        Args:
            data_path: Path to the input data file
            domain: Omics domain (transcriptomics, genomics, epigenetics, proteomics, metabolomics)
            task: Specific analysis task (qc, de, explore, etc.)
            params: Additional tool parameters

        Returns:
            Analysis results dict with 'success', 'results', and 'recommendations'
        """
        params = params or {}
        registry = await self._ensure_registry()
        results: list[dict[str, Any]] = []

        # Determine the tool to use based on domain + task
        domain_defaults = DOMAIN_DEFAULTS.get(domain, {})
        tool_name = params.pop("tool_name", None) or domain_defaults.get(task, "qc_report")

        # ── Step 1: QC / data exploration ──
        if tool_name == "qc_report" or task == "qc":
            qc_result = await registry.execute("qc_report", {
                "data_path": data_path,
                "domain": domain,
                "group_col": params.get("group_col", ""),
                "output_dir": params.get("output_dir", "./output"),
            })
            results.append({"step": "qc", "tool": "qc_report", "result": qc_result})

            if not qc_result.get("success"):
                return {
                    "success": False,
                    "error": f"QC failed: {qc_result.get('error')}",
                    "results": results,
                }

        # ── Step 2: Domain-specific analysis ──
        if task == "de" or tool_name == "differential_analysis":
            if domain != "transcriptomics":
                return {
                    "success": False,
                    "error": f"Differential analysis not yet supported for domain: {domain}",
                    "results": results,
                }

            de_params = {
                "counts_path": data_path,
                "metadata_path": params.get("metadata_path", ""),
                "group_col": params.get("group_col", "condition"),
                "method": params.get("method", "deseq2"),
                "alpha": params.get("alpha", 0.05),
                "output_dir": params.get("output_dir", "./output"),
            }
            de_result = await registry.execute("differential_analysis", de_params)
            results.append({"step": "de", "tool": "differential_analysis", "result": de_result})

        # ── Generate recommendations ──
        recommendations = []
        for r in results:
            if r["step"] == "qc" and r["result"].get("success"):
                recs = r["result"].get("recommendations", [])
                recommendations.extend(recs)

        return {
            "success": True,
            "domain": domain,
            "task": task,
            "results": results,
            "recommendations": recommendations,
        }
