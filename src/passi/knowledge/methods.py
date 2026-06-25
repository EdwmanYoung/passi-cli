"""Statistical methods catalog for omics data analysis.

Maps analysis requirements to methods, packages, and execution backends (Python/R).
"""

from __future__ import annotations

from typing import Any

# ═════════════════════════════════════════════════════════════════════
# Method catalog: method -> {description, backend, packages, ...}
# ═════════════════════════════════════════════════════════════════════

MethodInfo = dict[str, Any]

SINGLE_OMICS_METHODS: dict[str, MethodInfo] = {
    # ── Differential Expression / Abundance ──
    "deseq2": {
        "name": "DESeq2",
        "description": "Differential gene expression analysis for RNA-seq count data using negative binomial GLM",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["DESeq2", "BiocGenerics"],
        "input": "count_matrix",
        "output": "differential_results",
        "reference": "Love, Huber, & Anders (2014). Genome Biology, 15:550",
    },
    "edger": {
        "name": "edgeR",
        "description": "Empirical Bayes differential expression for RNA-seq count data",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["edgeR", "limma"],
        "input": "count_matrix",
        "output": "differential_results",
    },
    "limma_voom": {
        "name": "limma-voom",
        "description": "Linear models for microarray/RNA-seq data with voom transformation",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["limma"],
        "input": "expression_matrix",
        "output": "differential_results",
    },
    "pydeseq2": {
        "name": "PyDESeq2",
        "description": "Python implementation of DESeq2 for differential expression",
        "domain": "transcriptomics",
        "backend": "python",
        "packages": ["pydeseq2"],
        "input": "count_matrix",
        "output": "differential_results",
    },

    # ── Gene Set Enrichment ──
    "gsea": {
        "name": "GSEA",
        "description": "Gene Set Enrichment Analysis using ranked gene lists",
        "domain": "transcriptomics",
        "backend": "python",
        "packages": ["gseapy"],
        "input": "ranked_gene_list",
        "output": "enrichment_results",
    },
    "fgsea": {
        "name": "fgsea",
        "description": "Fast Gene Set Enrichment Analysis (R)",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["fgsea"],
        "input": "ranked_gene_list",
        "output": "enrichment_results",
    },
    "ora": {
        "name": "ORA",
        "description": "Over-Representation Analysis for gene sets",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["clusterProfiler"],
        "input": "gene_list",
        "output": "enrichment_results",
    },
    "clusterprofiler": {
        "name": "clusterProfiler",
        "description": "GO/KEGG enrichment and visualization (R/Bioconductor)",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["clusterProfiler", "org.Hs.eg.db"],
        "input": "gene_list",
        "output": "enrichment_results",
    },

    # ── Network Analysis ──
    "wgcna": {
        "name": "WGCNA",
        "description": "Weighted Gene Co-expression Network Analysis",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["WGCNA", "dynamicTreeCut"],
        "input": "expression_matrix",
        "output": "network_modules",
    },

    # ── Survival Analysis ──
    "km_survival": {
        "name": "Kaplan-Meier",
        "description": "Kaplan-Meier survival curve estimation with log-rank test",
        "domain": "clinical",
        "backend": "r",
        "packages": ["survival", "survminer"],
        "input": "clinical_data",
        "output": "survival_curves",
    },
    "cox_ph": {
        "name": "Cox PH Regression",
        "description": "Cox proportional hazards regression for survival analysis",
        "domain": "clinical",
        "backend": "r",
        "packages": ["survival"],
        "input": "clinical_data",
        "output": "hazard_ratios",
    },
    "regularized_cox": {
        "name": "Regularized Cox",
        "description": "Lasso/Ridge/Elastic-net regularized Cox regression for high-dimensional data",
        "domain": "clinical",
        "backend": "r",
        "packages": ["glmnet", "survival"],
        "input": "omics_matrix",
        "output": "selected_features",
    },
    "competing_risks": {
        "name": "Competing Risks",
        "description": "Fine-Gray subdistribution hazard model / cause-specific hazards",
        "domain": "clinical",
        "backend": "r",
        "packages": ["cmprsk", "survival"],
        "input": "clinical_data",
        "output": "competing_risk_model",
    },
    "rmst": {
        "name": "RMST",
        "description": "Restricted Mean Survival Time analysis",
        "domain": "clinical",
        "backend": "r",
        "packages": ["survRM2", "survival"],
        "input": "clinical_data",
        "output": "rmst_results",
    },
    "roc_analysis": {
        "name": "ROC Analysis",
        "description": "ROC curve, AUC, sensitivity/specificity analysis",
        "domain": "clinical",
        "backend": "python",
        "packages": ["sklearn", "matplotlib"],
        "input": "predictions",
        "output": "roc_curves",
    },

    # ── GWAS ──
    "gwas_plink": {
        "name": "GWAS (PLINK)",
        "description": "Genome-wide association study using PLINK",
        "domain": "genomics",
        "backend": "cli",
        "packages": ["plink"],
        "input": "plink_files",
        "output": "association_results",
    },
    "gwas_visualization": {
        "name": "GWAS Visualization",
        "description": "Manhattan plot and QQ plot for GWAS results",
        "domain": "genomics",
        "backend": "python",
        "packages": ["matplotlib"],
        "input": "gwas_results",
        "output": "figures",
    },

    # ── Epigenetics ──
    "diffbind": {
        "name": "DiffBind",
        "description": "Differential binding analysis for ChIP-seq/ATAC-seq peaks",
        "domain": "epigenetics",
        "backend": "r",
        "packages": ["DiffBind"],
        "input": "peak_files",
        "output": "differential_peaks",
    },
    "dmr_calling": {
        "name": "DMR Calling",
        "description": "Differentially methylated region detection (DSS/bumphunter)",
        "domain": "epigenetics",
        "backend": "r",
        "packages": ["DSS", "bumphunter", "minfi"],
        "input": "methylation_data",
        "output": "dmr_results",
    },
    "motif_enrichment": {
        "name": "Motif Enrichment",
        "description": "Transcription factor motif enrichment in genomic regions",
        "domain": "epigenetics",
        "backend": "python",
        "packages": [],
        "input": "peak_regions",
        "output": "motif_results",
    },

    # ── Proteomics ──
    "protein_diff": {
        "name": "Differential Protein Abundance",
        "description": "Differential protein abundance analysis (limma/MSstats)",
        "domain": "proteomics",
        "backend": "r",
        "packages": ["limma", "MSstats"],
        "input": "protein_matrix",
        "output": "differential_proteins",
    },

    # ── Metabolomics ──
    "metabolite_diff": {
        "name": "Differential Metabolite Abundance",
        "description": "Differential analysis for metabolomics data",
        "domain": "metabolomics",
        "backend": "python",
        "packages": ["scipy", "statsmodels"],
        "input": "metabolite_matrix",
        "output": "differential_metabolites",
    },
    "pathway_mapping": {
        "name": "Pathway Mapping",
        "description": "Map metabolites/genes to KEGG/MetaCyc pathways",
        "domain": "metabolomics",
        "backend": "python",
        "packages": [],
        "input": "metabolite_list",
        "output": "pathway_maps",
    },

    # ── Single-cell ──
    "single_cell_scanpy": {
        "name": "Single-cell Analysis (Scanpy)",
        "description": "scRNA-seq clustering, differential expression, trajectory (Python)",
        "domain": "transcriptomics",
        "backend": "python",
        "packages": ["scanpy", "anndata", "scipy"],
        "input": "h5ad",
        "output": "sc_results",
    },
    "single_cell_seurat": {
        "name": "Single-cell Analysis (Seurat)",
        "description": "scRNA-seq comprehensive pipeline (R/Seurat)",
        "domain": "transcriptomics",
        "backend": "r",
        "packages": ["Seurat", "SeuratObject"],
        "input": "h5ad",
        "output": "sc_results",
    },

    # ── General Statistics ──
    "ttest": {
        "name": "t-test (Student/Welch)",
        "description": "Two-sample t-test (parametric), Welch variant for unequal variance",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["scipy"],
        "input": "numeric_data",
        "output": "test_results",
    },
    "wilcoxon": {
        "name": "Wilcoxon Rank-Sum / Signed-Rank",
        "description": "Non-parametric two-group comparison",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["scipy"],
        "input": "numeric_data",
        "output": "test_results",
    },
    "anova": {
        "name": "ANOVA / ANCOVA",
        "description": "Analysis of variance / covariance for multi-group comparison",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["statsmodels"],
        "input": "numeric_data",
        "output": "test_results",
    },
    "kruskal": {
        "name": "Kruskal-Wallis",
        "description": "Non-parametric multi-group comparison",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["scipy"],
        "input": "numeric_data",
        "output": "test_results",
    },
    "correlation": {
        "name": "Correlation Analysis",
        "description": "Pearson/Spearman correlation with multiple testing correction",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["scipy", "pandas"],
        "input": "numeric_matrix",
        "output": "correlation_matrix",
    },
    "fisher_exact": {
        "name": "Fisher's Exact Test",
        "description": "Exact test for contingency table independence (small counts)",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["scipy"],
        "input": "contingency_table",
        "output": "test_results",
    },
    "chi_squared": {
        "name": "Chi-Squared Test",
        "description": "Chi-squared test of independence for contingency tables",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["scipy"],
        "input": "contingency_table",
        "output": "test_results",
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "description": "Binary/multinomial logistic regression for classification and odds ratio",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["statsmodels", "sklearn"],
        "input": "tabular_data",
        "output": "model_results",
    },
    "mixed_effects": {
        "name": "Mixed Effects Models",
        "description": "Linear/nonlinear mixed-effects models for repeated measures and longitudinal data",
        "domain": "biostatistics",
        "backend": "r",
        "packages": ["lme4", "lmerTest"],
        "input": "longitudinal_data",
        "output": "model_results",
    },
    "meta_analysis": {
        "name": "Meta-Analysis",
        "description": "Fixed/random effects meta-analysis with forest plot and heterogeneity testing",
        "domain": "biostatistics",
        "backend": "r",
        "packages": ["meta", "metafor"],
        "input": "study_results",
        "output": "meta_results",
    },
    "propensity_score": {
        "name": "Propensity Score Matching",
        "description": "Causal inference via propensity score matching/weighting",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["sklearn", "statsmodels"],
        "input": "clinical_data",
        "output": "matched_cohort",
    },
    "power_analysis": {
        "name": "Power Analysis",
        "description": "Sample size estimation and statistical power calculation",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["statsmodels"],
        "input": "parameters",
        "output": "power_results",
    },
    "multiple_testing": {
        "name": "Multiple Testing Correction",
        "description": "Bonferroni, FDR (Benjamini-Hochberg), permutation-based correction",
        "domain": "biostatistics",
        "backend": "python",
        "packages": ["statsmodels", "scipy"],
        "input": "p_values",
        "output": "adjusted_pvalues",
    },
}

MULTI_OMICS_METHODS: dict[str, MethodInfo] = {
    # ── Unsupervised Integration ──
    "mofa": {
        "name": "MOFA / MOFA+",
        "description": "Multi-Omics Factor Analysis — infers latent factors capturing variation across omics layers",
        "backend": "r",
        "packages": ["MOFA2"],
        "input": "multi_omics_list",
        "output": "latent_factors",
    },
    "mofa_py": {
        "name": "MOFA+ (Python)",
        "description": "Multi-Omics Factor Analysis in Python (mofapy2)",
        "backend": "python",
        "packages": ["mofapy2"],
        "input": "multi_omics_list",
        "output": "latent_factors",
    },
    "diablo": {
        "name": "DIABLO",
        "description": "Supervised multi-block discriminant analysis for biomarker discovery and classification",
        "backend": "r",
        "packages": ["mixOmics"],
        "input": "multi_omics_list",
        "output": "discriminant_features",
    },
    "splsda": {
        "name": "sPLS-DA",
        "description": "Sparse Partial Least Squares Discriminant Analysis for supervised multi-omics",
        "backend": "r",
        "packages": ["mixOmics"],
        "input": "multi_omics_list",
        "output": "selected_features",
    },
    "rcca": {
        "name": "Regularized CCA",
        "description": "Regularized Canonical Correlation Analysis for cross-omics correlation",
        "backend": "r",
        "packages": ["mixOmics"],
        "input": "two_omics_matrices",
        "output": "correlated_features",
    },
    "snf": {
        "name": "SNF",
        "description": "Similarity Network Fusion — builds and fuses patient similarity networks",
        "backend": "r",
        "packages": ["SNFtool"],
        "input": "multi_omics_list",
        "output": "fused_network",
    },
    "mcia": {
        "name": "MCIA",
        "description": "Multiple Co-Inertia Analysis for multi-table ordination",
        "backend": "r",
        "packages": ["omicade4", "made4"],
        "input": "multi_omics_list",
        "output": "co_inertia_results",
    },
    "mfa": {
        "name": "Multiple Factor Analysis",
        "description": "PCA with group-wise variable weighting to balance omics contributions",
        "backend": "r",
        "packages": ["FactoMineR", "factoextra"],
        "input": "multi_omics_list",
        "output": "factor_scores",
    },
    "consensus_cluster": {
        "name": "Consensus Clustering",
        "description": "Consensus clustering with resampling for robust patient subtyping",
        "backend": "r",
        "packages": ["ConsensusClusterPlus"],
        "input": "omics_matrix",
        "output": "cluster_assignments",
    },
    "icluster": {
        "name": "iCluster / iClusterBayes",
        "description": "Bayesian latent variable integrative clustering for multi-omics",
        "backend": "r",
        "packages": ["iClusterPlus"],
        "input": "multi_omics_list",
        "output": "cluster_assignments",
    },

    # ── ML-based Integration ──
    "random_forest_omics": {
        "name": "Random Forest (Multi-Omics)",
        "description": "Random Forest for multi-omics classification with feature importance",
        "backend": "python",
        "packages": ["sklearn", "pandas"],
        "input": "integrated_matrix",
        "output": "model_results",
    },
    "xgboost_omics": {
        "name": "XGBoost (Multi-Omics)",
        "description": "Gradient boosting for multi-omics prediction with SHAP interpretation",
        "backend": "python",
        "packages": ["xgboost", "shap"],
        "input": "integrated_matrix",
        "output": "model_results",
    },
    "autoencoder_omics": {
        "name": "Autoencoder Integration",
        "description": "Deep autoencoder for multi-omics latent representation learning",
        "backend": "python",
        "packages": ["torch", "sklearn"],
        "input": "integrated_matrix",
        "output": "latent_representation",
    },

    # ── Network Integration ──
    "xmwas": {
        "name": "xMWAS",
        "description": "Pairwise association + PLS components + community detection for integrative networks",
        "backend": "r",
        "packages": ["xMWAS"],
        "input": "multi_omics_list",
        "output": "integrated_network",
    },
}

QC_METHODS: dict[str, MethodInfo] = {
    "normalize": {
        "name": "Normalization",
        "variants": ["log10", "clr", "quantile", "zscore", "tmm", "rle", "total_sum"],
        "description": "Data normalization methods for omics data",
    },
    "impute_missing": {
        "name": "Missing Value Imputation",
        "variants": ["knn", "mofa", "probabilistic_min", "mean", "median", "zero"],
        "description": "Missing value imputation for omics data matrices",
    },
    "batch_correction": {
        "name": "Batch Effect Correction",
        "variants": ["combat", "harmony", "limma_removebatcheffect"],
        "description": "Batch effect removal from omics data",
    },
    "outlier_detection": {
        "name": "Outlier Detection",
        "variants": ["pca", "mahalanobis", "iqr", "zscore"],
        "description": "Outlier sample/feature detection methods",
    },
    "dim_reduction": {
        "name": "Dimensionality Reduction",
        "variants": ["pca", "tsne", "umap", "spca", "mds", "pcoa"],
        "description": "Dimensionality reduction for visualization and preprocessing",
    },
}


def get_method(method_id: str) -> MethodInfo | None:
    """Look up a method by ID across all catalogs."""
    for catalog in [SINGLE_OMICS_METHODS, MULTI_OMICS_METHODS, QC_METHODS]:
        if method_id in catalog:
            return catalog[method_id]
    return None


def get_methods_by_domain(domain: str) -> dict[str, MethodInfo]:
    """Get all methods for a specific omics domain."""
    result: dict[str, MethodInfo] = {}
    for mid, info in SINGLE_OMICS_METHODS.items():
        if info.get("domain") == domain:
            result[mid] = info
    return result


def get_multi_omics_methods() -> dict[str, MethodInfo]:
    """Get all multi-omics integration methods."""
    return dict(MULTI_OMICS_METHODS)


def search_methods(query: str) -> dict[str, MethodInfo]:
    """Search methods by keyword in name, description, or domain."""
    query_lower = query.lower()
    results: dict[str, MethodInfo] = {}
    for catalog in [SINGLE_OMICS_METHODS, MULTI_OMICS_METHODS, QC_METHODS]:
        for mid, info in catalog.items():
            searchable = f"{info['name']} {info.get('description','')} {info.get('domain','')}".lower()
            if query_lower in searchable:
                results[mid] = info
    return results


def list_all_methods() -> dict[str, list[str]]:
    """List all available methods grouped by category."""
    return {
        "single_omics": sorted(SINGLE_OMICS_METHODS.keys()),
        "multi_omics": sorted(MULTI_OMICS_METHODS.keys()),
        "qc_preprocessing": sorted(QC_METHODS.keys()),
    }
