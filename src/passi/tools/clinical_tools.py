"""Clinical statistics tools — survival analysis, Cox regression, competing risks.

Kaplan-Meier, Cox PH, and Fine-Gray competing risks via R's survival package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# R code templates
# ═══════════════════════════════════════════════════════════════════

KM_SURVFIT_SCRIPT = r"""
suppressMessages(library(survival))
suppressMessages(library(jsonlite))

data <- read.table("{data_path}", header=TRUE, sep="\t", stringsAsFactors=FALSE)

# Build Surv object
surv_obj <- Surv(time = data${time_col}, event = data${event_col})
logrank_p <- NA
n_groups <- 1

if ("{group_col}" != "") {{
    if (!("{group_col}" %in% colnames(data))) stop("Group column '{group_col}' not found")
    group_var <- data${group_col}
    fit <- survfit(surv_obj ~ group_var)
    lr_test <- survdiff(surv_obj ~ group_var)
    n_groups <- length(unique(group_var))
    logrank_p <- 1 - pchisq(lr_test$chisq, df = n_groups - 1)
    cat(sprintf("KM_DONE|groups=%d|logrank_p=%.6f\n", n_groups, logrank_p))
}} else {{
    fit <- survfit(surv_obj ~ 1)
    cat("KM_DONE|groups=1|overall\n")
}}

# Write all results to JSON
smry <- summary(fit)
result <- list(
    method = "km",
    groups = n_groups,
    logrank_p = logrank_p,
    strata = if (is.null(fit$strata)) NULL else as.character(names(fit$strata)),
    n = as.integer(fit$n),
    events = as.integer(smry$table[,"events"]),
    median_survival = as.numeric(smry$table[,"median"]),
    ci_lower = as.numeric(smry$table[,"0.95LCL"]),
    ci_upper = as.numeric(smry$table[,"0.95UCL"])
)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), "{output_json}")
"""

COXPH_SCRIPT = r"""
suppressMessages(library(survival))
suppressMessages(library(jsonlite))

data <- read.table("{data_path}", header=TRUE, sep="\t", stringsAsFactors=FALSE)

# Build formula
if ("{covariates}" != "") {{
    covs <- strsplit("{covariates}", ",")[[1]]
    covs <- trimws(covs)
    missing_covs <- setdiff(covs, colnames(data))
    if (length(missing_covs) > 0) stop(paste("Covariates not found:", paste(missing_covs, collapse=", ")))
    rhs <- paste(covs, collapse = " + ")
}} else {{
    rhs <- "{group_col}"
}}
form <- as.formula(paste("Surv(", "{time_col}", ",", "{event_col}", ") ~", rhs))

fit <- coxph(form, data = data)
fit_summary <- summary(fit)

# Check proportional hazards
ph_test <- cox.zph(fit)
ph_global_p <- ph_test$table[1, "p"]
cat(sprintf("COXPH_DONE|n=%d|events=%d|concordance=%.4f|ph_p=%.4f\n",
    fit$n, fit$nevent, fit_summary$concordance[1], ph_global_p))

# Build coefficients list
coefs <- list()
coef_table <- fit_summary$coefficients
if (nrow(coef_table) > 0) {{
    for (i in 1:nrow(coef_table)) {{
        coefs[[i]] <- list(
            variable = rownames(coef_table)[i],
            coef = as.numeric(coef_table[i, "coef"]),
            exp_coef = as.numeric(coef_table[i, "exp(coef)"]),
            se = as.numeric(coef_table[i, "se(coef)"]),
            z = as.numeric(coef_table[i, "z"]),
            p = as.numeric(coef_table[i, "Pr(>|z|)"])
        )
    }}
}}

result <- list(
    method = "cox",
    n = fit$n,
    events = fit$nevent,
    concordance = as.numeric(fit_summary$concordance[1]),
    ph_global_p = ph_global_p,
    log_likelihood = as.numeric(fit$loglik[2]),
    wald_test_p = as.numeric(fit_summary$waldtest["pvalue"]),
    coefficients = coefs
)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), "{output_json}")
"""

COMPETING_RISK_SCRIPT = r"""
suppressMessages(library(survival))
suppressMessages(library(jsonlite))

data <- read.table("{data_path}", header=TRUE, sep="\t", stringsAsFactors=FALSE)

# Build Surv with event type (competing risks)
# event_col should be coded: 0=censored, 1=event of interest, 2=competing event
surv_obj <- Surv(time = data${time_col}, event = data${event_col}, type = "mstate")

# Cumulative incidence
fit <- survfit(surv_obj ~ 1)
n_ev1 <- sum(data${event_col} == 1)
n_ev2 <- sum(data${event_col} == 2)
n_cen <- sum(data${event_col} == 0)
cat(sprintf("CR_DONE|n=%d|events_type1=%d|events_type2=%d\n", nrow(data), n_ev1, n_ev2))

result <- list(
    method = "competing_risks",
    n_total = nrow(data),
    events_primary = n_ev1,
    events_competing = n_ev2,
    censored = n_cen
)
writeLines(toJSON(result, auto_unbox = TRUE, pretty = TRUE), "{output_json}")
"""

# ═══════════════════════════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════════════════════════


class SurvivalAnalysisParams(BaseModel):
    """Parameters for survival analysis."""

    data_path: str = Field(..., description="Path to clinical data (TSV/CSV)")
    time_col: str = Field(..., description="Column name for survival time (e.g., 'OS.time')")
    event_col: str = Field(..., description="Column name for event indicator (1=event, 0=censored)")
    group_col: str = Field(
        default="",
        description="Column name for group comparison (KM curves). Omit for overall survival.",
    )
    covariates: str = Field(
        default="",
        description="Comma-separated covariate column names for Cox regression",
    )
    method: str = Field(
        default="km",
        description="Method: 'km' (Kaplan-Meier), 'cox' (Cox PH), 'competing_risks'",
    )
    output_dir: str = Field(default="./output", description="Output directory for results")


class SurvivalAnalysisTool:
    """Survival analysis: Kaplan-Meier, Cox PH regression, competing risks.

    Executes via R's survival package through rpy2 bridge or Rscript fallback.
    """

    name = "survival_analysis"
    description = (
        "Perform survival analysis on clinical data with time-to-event endpoints. "
        "Supports Kaplan-Meier curves with log-rank test, Cox proportional hazards "
        "regression, and competing risks analysis. Requires columns for survival time "
        "and event status. Optionally specify group_col for KM comparison or "
        "covariates for Cox regression."
    )
    params_model = SurvivalAnalysisParams

    def __init__(self, r_home: str = "", r_lib_path: str = "", r_path: str = "Rscript") -> None:
        self.r_home = r_home
        self.r_lib_path = r_lib_path
        self.r_path = r_path

    async def execute(self, params: SurvivalAnalysisParams, **kwargs: Any) -> dict[str, Any]:
        import json

        from passi.executors.r_executor import init_rpy2

        # Validate inputs
        data_path = Path(params.data_path)
        if not data_path.exists():
            return {"success": False, "error": f"Data file not found: {params.data_path}"}

        method = params.method.lower()
        if method not in ("km", "cox", "competing_risks"):
            return {
                "success": False,
                "error": f"Unknown method: {method}. Choose: km, cox, competing_risks",
            }

        output_dir = Path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / f"survival_{method}_result.json"

        # Select script template
        script_map = {
            "km": KM_SURVFIT_SCRIPT,
            "cox": COXPH_SCRIPT,
            "competing_risks": COMPETING_RISK_SCRIPT,
        }

        script = script_map[method].format(
            data_path=str(data_path.resolve()).replace("\\", "/"),
            time_col=params.time_col,
            event_col=params.event_col,
            group_col=params.group_col,
            covariates=params.covariates,
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

                return self._parse_output(method, output_json, data_path)
            except Exception as e:
                logger.warning("rpy2 execution failed, falling back to Rscript: %s", e)

        return self._execute_via_rscript(script, method, output_json, data_path)

    def _execute_via_rscript(self, script: str, method: str, output_json: Path, data_path: Path) -> dict[str, Any]:
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
            return self._parse_output(method, output_json, data_path, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Survival analysis timed out (600s)"}
        except FileNotFoundError:
            return {"success": False, "error": f"Rscript not found: {rscript}"}
        finally:
            Path(script_path).unlink(missing_ok=True)

    def _parse_output(
        self, method: str, output_json: Path, data_path: Path,
        stdout: str = "", stderr: str = "",
    ) -> dict[str, Any]:
        import json

        result: dict[str, Any] = {
            "success": True,
            "method": method,
            "data_file": str(data_path),
        }

        # Parse summary line from stdout (primary for backward compat)
        for line in stdout.splitlines():
            if "KM_DONE|" in line:
                for part in line.split("|")[1:]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        try:
                            result[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            result[k] = v
            elif "COXPH_DONE|" in line:
                for part in line.split("|")[1:]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        try:
                            result[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            result[k] = v
            elif "CR_DONE|" in line:
                for part in line.split("|")[1:]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        try:
                            result[k] = int(v)
                        except ValueError:
                            result[k] = v

        if result.get("ph_p") is not None and result["ph_p"] < 0.05:
            result["ph_warning"] = "Proportional hazards assumption may be violated (p < 0.05)"

        # Read JSON output (contains the full structured results)
        if output_json.exists():
            try:
                with open(output_json, encoding="utf-8") as f:
                    json_data = json.load(f)
                result["details"] = json_data
                # Promote key fields to top level
                for key in ("n", "events", "logrank_p", "concordance",
                           "ph_global_p", "wald_test_p", "groups",
                           "events_primary", "events_competing", "n_total"):
                    if key in json_data and key not in result:
                        result[key] = json_data[key]
            except (json.JSONDecodeError, Exception) as e:
                result["parse_warning"] = str(e)

        if not result.get("details") and not any(
            k in result for k in ("n", "logrank_p", "concordance", "events_primary")
        ):
            result["warning"] = "No output data produced — check stderr"
            result["stderr"] = stderr[-2000:]

        return result
