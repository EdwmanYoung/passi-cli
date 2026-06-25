"""Stats Expert sub-agent for clinical statistics and biostatistics analysis."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

STATS_EXPERT_PROMPT = """You are the Stats Expert, a specialized biostatistics sub-agent.

Your expertise covers:
- **Survival Analysis**: Kaplan-Meier curves, log-rank test, Cox PH regression, competing risks (Fine-Gray), RMST
- **Clinical Statistics**: ROC/AUC, sensitivity/specificity, diagnostic test evaluation
- **Hypothesis Testing**: t-test, Wilcoxon, ANOVA, Kruskal-Wallis, chi-squared, Fisher's exact test
- **Regression Models**: linear, logistic, Cox regularized (lasso/ridge/elastic net)
- **Advanced Methods**: mixed-effects models (longitudinal data), propensity score matching, meta-analysis
- **Study Design**: power analysis, sample size calculation
- **Multiple Testing**: Bonferroni, FDR (Benjamini-Hochberg), permutation correction

## Workflow
1. Identify the study design (case-control, cohort, RCT, etc.)
2. Check data assumptions (normality, proportional hazards, independence)
3. Select appropriate statistical method
4. Report effect sizes, confidence intervals, and p-values
5. Apply multiple testing correction when appropriate

## Guidelines
- Always check proportional hazards assumption before Cox regression
- Use non-parametric methods when normality is violated
- Report both unadjusted and adjusted p-values
- Consider confounders and effect modification
- Use R (survival, glmnet, metafor) for specialized clinical methods
- Use Python (scipy, statsmodels) for general hypothesis testing
"""

ANALYSIS_TOOL_MAP: dict[str, str] = {
    "survival": "survival_analysis",
    "km": "survival_analysis",
    "cox": "survival_analysis",
    "competing_risks": "survival_analysis",
    "survival_analysis": "survival_analysis",
}


class StatsExpert:
    """Sub-agent specialized in clinical statistics and biostatistics.

    Runs in an isolated context; only statistical results flow back to the main agent.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._system_prompt = STATS_EXPERT_PROMPT
        self._tool_registry = None

    async def _ensure_registry(self) -> Any:
        """Lazy-load the tool registry with clinical tools."""
        if self._tool_registry is None:
            from passi.tools.registry import ToolRegistry
            from passi.tools.clinical_tools import SurvivalAnalysisTool

            exec_cfg = self.runtime.config.execution
            registry = ToolRegistry()

            surv_tool = SurvivalAnalysisTool(
                r_home=exec_cfg.r_home or "",
                r_lib_path=exec_cfg.r_lib_path or "",
                r_path=exec_cfg.rscript_binary,
            )
            registry.register(surv_tool, "clinical")
            self._tool_registry = registry
        return self._tool_registry

    async def analyze(
        self,
        data_path: str,
        analysis_type: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a statistical analysis task.

        Args:
            data_path: Path to the clinical/phenotype data
            analysis_type: Type of analysis (survival, km, cox, competing_risks, regression, hypothesis_test)
            params: Method-specific parameters

        Returns:
            Statistical results dict with 'success', 'results', and 'interpretation'
        """
        params = params or {}
        registry = await self._ensure_registry()

        tool_name = ANALYSIS_TOOL_MAP.get(analysis_type)
        if tool_name is None:
            # Try generic statistical analysis via Python
            result = self._run_python_stats(data_path, analysis_type, params)
            interpretation = self._interpret(result, analysis_type)
            return {
                "success": result.get("success", False),
                "analysis_type": analysis_type,
                "tool": f"python_{analysis_type}",
                "result": result,
                "interpretation": interpretation,
            }

        # Map analysis_type to survival_analysis method parameter
        method = "km"
        if analysis_type in ("cox",):
            method = "cox"
        elif analysis_type in ("competing_risks",):
            method = "competing_risks"

        tool_params = {
            "data_path": data_path,
            "time_col": params.get("time_col", "time"),
            "event_col": params.get("event_col", "status"),
            "group_col": params.get("group_col", ""),
            "covariates": params.get("covariates", ""),
            "method": method,
            "output_dir": params.get("output_dir", "./output"),
        }

        result = await registry.execute(tool_name, tool_params)

        # Generate interpretation
        interpretation = self._interpret(result, analysis_type)

        return {
            "success": result.get("success", False),
            "analysis_type": analysis_type,
            "tool": tool_name,
            "result": result,
            "interpretation": interpretation,
        }

    def _run_python_stats(
        self, data_path: str, analysis_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Run statistical analysis using Python (scipy/statsmodels)."""
        import numpy as np
        import pandas as pd
        from pathlib import Path

        path = Path(data_path)
        if not path.exists():
            return {"success": False, "error": f"Data file not found: {data_path}"}

        try:
            if path.suffix in (".csv",):
                df = pd.read_csv(path)
            elif path.suffix in (".tsv", ".txt"):
                df = pd.read_csv(path, sep="\t")
            else:
                return {"success": False, "error": f"Unsupported format: {path.suffix}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {e}"}

        try:
            if analysis_type in ("ttest", "t_test", "t-test"):
                return self._run_ttest(df, params)
            elif analysis_type in ("anova",):
                return self._run_anova(df, params)
            elif analysis_type in ("correlation", "corr"):
                return self._run_correlation(df, params)
            elif analysis_type in ("chi2", "chisquare", "chi_squared"):
                return self._run_chi2(df, params)
            else:
                return {
                    "success": False,
                    "error": f"Unknown analysis type: {analysis_type}. "
                    f"Supported Python stats: ttest, anova, correlation, chi2",
                }
        except Exception as e:
            logger.exception("Python stats analysis failed")
            return {"success": False, "error": str(e)}

    def _run_ttest(self, df: Any, params: dict[str, Any]) -> dict[str, Any]:
        from scipy import stats as sp_stats

        group_col = params.get("group_col", "")
        value_col = params.get("value_col", "")
        if not group_col or not value_col:
            return {"success": False, "error": "ttest requires group_col and value_col"}

        groups = df[group_col].unique()
        if len(groups) != 2:
            return {"success": False, "error": f"ttest requires exactly 2 groups, got {len(groups)}"}

        g1 = df[df[group_col] == groups[0]][value_col].dropna()
        g2 = df[df[group_col] == groups[1]][value_col].dropna()

        t_stat, p_val = sp_stats.ttest_ind(g1, g2)
        return {
            "success": True,
            "method": "independent_t_test",
            "groups": [str(groups[0]), str(groups[1])],
            "n1": len(g1), "n2": len(g2),
            "mean1": float(g1.mean()), "mean2": float(g2.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
        }

    def _run_anova(self, df: Any, params: dict[str, Any]) -> dict[str, Any]:
        from scipy import stats as sp_stats

        group_col = params.get("group_col", "")
        value_col = params.get("value_col", "")
        if not group_col or not value_col:
            return {"success": False, "error": "ANOVA requires group_col and value_col"}

        groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
        if len(groups) < 2:
            return {"success": False, "error": "ANOVA requires at least 2 groups"}

        f_stat, p_val = sp_stats.f_oneway(*groups)
        return {
            "success": True,
            "method": "one_way_anova",
            "n_groups": len(groups),
            "total_n": sum(len(g) for g in groups),
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
        }

    def _run_correlation(self, df: Any, params: dict[str, Any]) -> dict[str, Any]:
        from scipy import stats as sp_stats

        col1 = params.get("col1", df.columns[0])
        col2 = params.get("col2", df.columns[1] if len(df.columns) > 1 else df.columns[0])

        x = df[col1].dropna()
        y = df[col2].dropna()
        # Align by index
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]

        r, p = sp_stats.pearsonr(x, y)
        rho, p_s = sp_stats.spearmanr(x, y)
        return {
            "success": True,
            "method": "correlation",
            "n": len(x),
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(p_s),
        }

    def _run_chi2(self, df: Any, params: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd
        from scipy import stats as sp_stats

        col1 = params.get("col1", df.columns[0])
        col2 = params.get("col2", df.columns[1] if len(df.columns) > 1 else df.columns[0])

        table = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = sp_stats.chi2_contingency(table)
        return {
            "success": True,
            "method": "chi_squared",
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "contingency_table": table.to_dict(),
        }

    def _interpret(self, result: dict[str, Any], analysis_type: str) -> str:
        """Generate a plain-language interpretation of statistical results."""
        if not result.get("success"):
            return f"Analysis failed: {result.get('error', 'Unknown error')}"

        if analysis_type in ("survival", "km", "cox", "competing_risks"):
            parts = []
            logrank = result.get("logrank_p")
            if logrank is not None:
                sig = "statistically significant" if logrank < 0.05 else "not statistically significant"
                parts.append(f"Log-rank test p={logrank:.4f} ({sig}).")

            concordance = result.get("concordance")
            if concordance is not None:
                parts.append(f"Model concordance = {concordance:.3f} (higher is better discrimination).")

            ph_p = result.get("ph_global_p")
            if ph_p is not None and ph_p < 0.05:
                parts.append(f"Warning: proportional hazards assumption may be violated (p={ph_p:.4f}).")

            details = result.get("details", {})
            coefs = details.get("coefficients", []) if isinstance(details, dict) else []
            for c in coefs:
                hr = c.get("exp_coef", 1)
                direction = "increased" if hr > 1 else "decreased"
                parts.append(
                    f"{c['variable']}: HR={hr:.2f} ({direction} risk), p={c.get('p', 0):.4f}"
                )

            return " ".join(parts) if parts else "Analysis completed successfully."

        if analysis_type in ("ttest", "t_test", "t-test"):
            p = result.get("p_value", 1)
            sig = "significant" if p < 0.05 else "not significant"
            return (
                f"t-test: t={result.get('t_statistic', 0):.3f}, p={p:.4f} ({sig}). "
                f"Mean difference: {result.get('mean1', 0) - result.get('mean2', 0):.3f}"
            )

        if analysis_type in ("anova",):
            p = result.get("p_value", 1)
            sig = "significant" if p < 0.05 else "not significant"
            return f"One-way ANOVA: F({result.get('n_groups', 0) - 1},{result.get('total_n', 0) - result.get('n_groups', 0)})={result.get('f_statistic', 0):.3f}, p={p:.4f} ({sig})."

        return "Analysis completed successfully."
