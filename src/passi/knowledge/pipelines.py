"""Predefined analysis pipelines as YAML workflow definitions.

Each pipeline is a reusable analysis workflow that the agent can execute.
"""

from __future__ import annotations

from typing import Any

PIPELINES: dict[str, dict[str, Any]] = {
    "rnaseq_diff_expr": {
        "name": "RNA-seq Differential Expression",
        "description": "Standard RNA-seq differential expression analysis pipeline",
        "domain": "transcriptomics",
        "steps": [
            {"name": "load_counts", "tool": "parse_omics_data", "params": {"path": "{{counts_file}}"}},
            {"name": "load_metadata", "tool": "parse_omics_data", "params": {"path": "{{metadata_file}}"}},
            {"name": "qc", "tool": "qc_report", "params": {"data": "{{load_counts.result}}"}},
            {"name": "normalize", "tool": "normalize_data", "params": {"method": "tmm", "data": "{{load_counts.result}}"}},
            {"name": "diff_expr", "tool": "differential_analysis", "params": {
                "method": "deseq2",
                "data": "{{normalize.result}}",
                "group_col": "{{group_column}}",
                "contrast": "{{contrast}}",
            }},
            {"name": "volcano", "tool": "plot_volcano", "params": {"data": "{{diff_expr.result}}", "fdr_cutoff": 0.05, "log2fc_cutoff": 1.0}},
            {"name": "export", "tool": "export_results", "params": {"data": "{{diff_expr.result}}", "path": "{{output_dir}}/diff_results.csv", "format": "csv"}},
        ],
    },
    "rnaseq_gsea": {
        "name": "RNA-seq GSEA Pathway Analysis",
        "description": "Gene set enrichment analysis for RNA-seq results",
        "domain": "transcriptomics",
        "steps": [
            {"name": "load_results", "tool": "parse_omics_data", "params": {"path": "{{diff_results_file}}"}},
            {"name": "rank_genes", "tool": "run_python", "params": {"code": "..."}},
            {"name": "gsea", "tool": "gsea_analysis", "params": {"data": "{{rank_genes.result}}", "gene_sets": "MSigDB_Hallmark"}},
            {"name": "plot_enrichment", "tool": "plot_heatmap", "params": {"data": "{{gsea.result}}"}},
        ],
    },
    "wgcna_pipeline": {
        "name": "WGCNA Co-expression Network",
        "description": "Weighted gene co-expression network analysis",
        "domain": "transcriptomics",
        "steps": [
            {"name": "load_expr", "tool": "parse_omics_data", "params": {"path": "{{expression_file}}"}},
            {"name": "wgcna", "tool": "wgcna_analysis", "params": {
                "data": "{{load_expr.result}}",
                "power": "auto",
                "min_module_size": 30,
            }},
            {"name": "module_trait", "tool": "run_r", "params": {"code": "..."}},
            {"name": "export_modules", "tool": "export_results", "params": {"data": "{{wgcna.result}}", "path": "{{output_dir}}/wgcna_modules.csv"}},
        ],
    },
    "survival_analysis": {
        "name": "Survival Analysis Pipeline",
        "description": "Kaplan-Meier + Cox regression survival analysis",
        "domain": "clinical",
        "steps": [
            {"name": "load_clinical", "tool": "parse_omics_data", "params": {"path": "{{clinical_file}}"}},
            {"name": "km_analysis", "tool": "survival_analysis", "params": {
                "data": "{{load_clinical.result}}",
                "time_col": "{{time_column}}",
                "event_col": "{{event_column}}",
                "group_col": "{{group_column}}",
                "method": "km",
            }},
            {"name": "cox_model", "tool": "survival_analysis", "params": {
                "data": "{{load_clinical.result}}",
                "time_col": "{{time_column}}",
                "event_col": "{{event_column}}",
                "covariates": "{{covariates}}",
                "method": "cox",
            }},
            {"name": "plot_km", "tool": "plot_survival", "params": {"data": "{{km_analysis.result}}"}},
            {"name": "export", "tool": "export_results", "params": {"data": "{{cox_model.result}}", "path": "{{output_dir}}/cox_results.csv"}},
        ],
    },
    "multi_omics_mofa": {
        "name": "Multi-Omics MOFA Integration",
        "description": "MOFA+ integration of multiple omics datasets",
        "domain": "multi-omics",
        "steps": [
            {"name": "load_omics1", "tool": "parse_omics_data", "params": {"path": "{{omics1_file}}"}},
            {"name": "load_omics2", "tool": "parse_omics_data", "params": {"path": "{{omics2_file}}"}},
            {"name": "load_omics3", "tool": "parse_omics_data", "params": {"path": "{{omics3_file}}"}},
            {"name": "mofa", "tool": "mofa_integration", "params": {
                "data": ["{{load_omics1.result}}", "{{load_omics2.result}}", "{{load_omics3.result}}"],
                "n_factors": "auto",
            }},
            {"name": "plot_factors", "tool": "plot_heatmap", "params": {"data": "{{mofa.result}}"}},
            {"name": "export", "tool": "export_results", "params": {"data": "{{mofa.result}}", "path": "{{output_dir}}/mofa_factors.csv"}},
        ],
    },
    "multi_omics_diablo": {
        "name": "Multi-Omics DIABLO Integration",
        "description": "Supervised DIABLO multi-block discriminant analysis",
        "domain": "multi-omics",
        "steps": [
            {"name": "load_data", "tool": "parse_omics_data", "params": {"path": "{{data_dir}}"}},
            {"name": "diablo", "tool": "diablo_integration", "params": {
                "data": "{{load_data.result}}",
                "outcome": "{{outcome_variable}}",
                "design": "full",
            }},
            {"name": "plot_diablo", "tool": "run_r", "params": {"code": "..."}},
        ],
    },
    "ml_predictor": {
        "name": "Multi-Omics ML Predictor",
        "description": "Machine learning predictor from integrated omics features",
        "domain": "multi-omics",
        "steps": [
            {"name": "load_features", "tool": "parse_omics_data", "params": {"path": "{{feature_matrix}}"}},
            {"name": "train_model", "tool": "multi_omics_ml", "params": {
                "data": "{{load_features.result}}",
                "target": "{{target_variable}}",
                "method": "{{ml_method}}",
                "cv_folds": 5,
            }},
            {"name": "shap_analysis", "tool": "run_python", "params": {"code": "..."}},
            {"name": "plot_importance", "tool": "run_python", "params": {"code": "..."}},
        ],
    },
}


def get_pipeline(name: str) -> dict[str, Any] | None:
    """Get a pipeline definition by name."""
    return PIPELINES.get(name)


def list_pipelines(domain: str | None = None) -> list[dict[str, str]]:
    """List available pipelines, optionally filtered by domain."""
    result = []
    for name, pipe in PIPELINES.items():
        if domain is None or pipe["domain"] == domain:
            result.append({
                "name": name,
                "title": pipe["name"],
                "description": pipe["description"],
                "domain": pipe["domain"],
                "n_steps": len(pipe["steps"]),
            })
    return result
