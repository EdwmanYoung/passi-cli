"""File I/O tools for PassiAgent.

Core tools for reading, writing, and parsing omics data files.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


def _resolve_suffix(path: Path) -> tuple[str, bool]:
    """Resolve the effective file suffix, handling gzip compression.

    Returns (suffix, is_compressed).
    """
    name = path.name.lower()
    if name.endswith(".gz"):
        inner = Path(name[:-3])  # strip .gz
        return inner.suffix.lower(), True
    return path.suffix.lower(), False

from passi.tools.base import CallableTool


# ═════════════════════════════════════════════════════════════════════
# read_file
# ═════════════════════════════════════════════════════════════════════

class ReadFileParams(BaseModel):
    path: str = Field(..., description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")
    max_lines: int = Field(default=1000, description="Maximum lines to return")
    sheet_name: str | None = Field(default=None, description="Sheet name for Excel files")


class ReadFileTool(CallableTool[ReadFileParams]):
    name = "read_file"
    description = "Read contents of a file. Supports text, CSV, TSV, Excel, JSON, Parquet, and other common formats."
    params_model = ReadFileParams

    async def execute(self, params: ReadFileParams, **kwargs: Any) -> dict[str, Any]:
        path = Path(params.path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {params.path}"}

        suffix, is_gz = _resolve_suffix(path)

        try:
            if suffix in (".csv",):
                df = pd.read_csv(path, nrows=params.max_lines)
                return {
                    "success": True,
                    "format": "csv",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "preview": df.head(100).to_dict(orient="records"),
                    "summary": df.describe(include="all").to_dict() if len(df.columns) < 50 else None,
                }
            elif suffix in (".tsv", ".txt"):
                df = pd.read_csv(path, sep="\t", nrows=params.max_lines)
                return {
                    "success": True,
                    "format": "tsv",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "preview": df.head(100).to_dict(orient="records"),
                }
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path, sheet_name=params.sheet_name or 0, nrows=params.max_lines)
                return {
                    "success": True,
                    "format": "excel",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "preview": df.head(100).to_dict(orient="records"),
                }
            elif suffix in (".json",):
                with open(path, encoding=params.encoding) as f:
                    data = json.load(f)
                return {"success": True, "format": "json", "data": data}
            elif suffix in (".parquet",):
                df = pd.read_parquet(path)
                return {
                    "success": True,
                    "format": "parquet",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "preview": df.head(100).to_dict(orient="records"),
                }
            else:
                if is_gz:
                    with gzip.open(path, "rt", encoding=params.encoding, errors="replace") as f:
                        lines = [f.readline() for _ in range(params.max_lines)]
                else:
                    with open(path, encoding=params.encoding, errors="replace") as f:
                        lines = [f.readline() for _ in range(params.max_lines)]
                return {
                    "success": True,
                    "format": "text",
                    "content": "".join(lines),
                    "line_count": len(lines),
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {e}"}


# ═════════════════════════════════════════════════════════════════════
# write_file
# ═════════════════════════════════════════════════════════════════════

class WriteFileParams(BaseModel):
    path: str = Field(..., description="Output file path")
    content: str = Field(default="", description="Text content to write")
    data: list[dict[str, Any]] | None = Field(default=None, description="Tabular data to write")
    format: str = Field(default="csv", description="Output format: csv, tsv, json, txt")


class WriteFileTool(CallableTool[WriteFileParams]):
    name = "write_file"
    description = "Write data to a file. Supports CSV, TSV, JSON, and text formats."
    params_model = WriteFileParams

    async def execute(self, params: WriteFileParams, **kwargs: Any) -> dict[str, Any]:
        path = Path(params.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if params.data is not None and params.format in ("csv", "tsv"):
                df = pd.DataFrame(params.data)
                sep = "\t" if params.format == "tsv" else ","
                df.to_csv(path, sep=sep, index=False)
            elif params.data is not None and params.format == "json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(params.data, f, ensure_ascii=False, indent=2)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(params.content)

            return {
                "success": True,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to write file: {e}"}


# ═════════════════════════════════════════════════════════════════════
# parse_omics_data
# ═════════════════════════════════════════════════════════════════════

class ParseOmicsDataParams(BaseModel):
    path: str = Field(..., description="Path to omics data file")
    file_type: str = Field(
        default="auto",
        description="File type hint: auto, count_matrix, vcf, fasta, fastq, gff, bed, mzml, h5ad, pheno_table",
    )
    sample_sheet: str | None = Field(default=None, description="Optional sample metadata file")


class ParseOmicsDataTool(CallableTool[ParseOmicsDataParams]):
    name = "parse_omics_data"
    description = (
        "Parse and auto-detect omics data files. Supports: count matrices (CSV/TSV), "
        "VCF, FASTA, FASTQ, GFF/GTF, BED, mzML, AnnData (h5ad), and phenotype tables. "
        "Returns detected format, dimensions, and preview."
    )
    params_model = ParseOmicsDataParams

    async def execute(self, params: ParseOmicsDataParams, **kwargs: Any) -> dict[str, Any]:
        path = Path(params.path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {params.path}"}

        suffix, is_gz = _resolve_suffix(path)
        detected: dict[str, Any] = {
            "file_path": str(path),
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "suffix": path.suffix.lower(),
            "is_compressed": is_gz,
        }

        try:
            if suffix in (".csv", ".tsv", ".txt"):
                return await self._parse_matrix(path, suffix, detected, params)
            elif suffix in (".vcf", ".bcf"):
                return self._parse_vcf(path, detected)
            elif suffix in (".fasta", ".fa", ".fna"):
                return self._parse_fasta(path, detected)
            elif suffix in (".fastq", ".fq"):
                return self._parse_fastq(path, detected)
            elif suffix in (".gff", ".gtf", ".gff3"):
                return self._parse_gff(path, detected)
            elif suffix in (".bed", ".narrowPeak", ".broadPeak"):
                return self._parse_bed(path, suffix, detected)
            elif suffix in (".h5ad",):
                return await self._parse_anndata(path, detected)
            elif suffix in (".xlsx", ".xls"):
                return await self._parse_matrix(path, suffix, detected)
            else:
                detected["detected_type"] = "unknown"
                detected["message"] = f"Cannot auto-detect format for suffix: {suffix}"
                return {"success": True, **detected}
        except Exception as e:
            return {"success": False, "error": f"Parse failed: {e}", "detected": detected}

    async def _parse_matrix(
        self, path: Path, suffix: str, detected: dict[str, Any], params: ParseOmicsDataParams
    ) -> dict[str, Any]:
        sep = "\t" if suffix in (".tsv", ".txt") else ","
        engine = "openpyxl" if suffix in (".xlsx", ".xls") else None
        if engine:
            df = pd.read_excel(path, nrows=10)
        else:
            df = pd.read_csv(path, sep=sep, nrows=10)
        # Try to detect if this is a count matrix (gene names in first column)
        first_col = df.columns[0]
        is_matrix = df.shape[1] >= 3 and df.iloc[:, 1:].dtypes.apply(
            lambda d: pd.api.types.is_numeric_dtype(d)
        ).all()
        detected.update({
            "detected_type": "count_matrix" if is_matrix else "tabular",
            "shape": list(df.shape),
            "columns": list(df.columns)[:30],
            "preview_rows": len(df),
            "omic_hint": _guess_omic_type(df),
        })
        return {"success": True, **detected}

    def _parse_vcf(self, path: Path, detected: dict[str, Any]) -> dict[str, Any]:
        variant_count = 0
        samples: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#CHROM"):
                    samples = line.strip().split("\t")[9:]
                elif not line.startswith("#"):
                    variant_count += 1
                    if variant_count > 1000:
                        break
        detected.update({
            "detected_type": "vcf",
            "variant_count_preview": variant_count,
            "samples": samples,
        })
        return {"success": True, **detected}

    def _parse_fasta(self, path: Path, detected: dict[str, Any]) -> dict[str, Any]:
        seq_count = 0
        total_bases = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith(">"):
                    seq_count += 1
                elif not line.startswith(";"):
                    total_bases += len(line.strip())
        detected.update({
            "detected_type": "fasta",
            "sequence_count": seq_count,
            "total_bases_approx": total_bases,
        })
        return {"success": True, **detected}

    def _parse_fastq(self, path: Path, detected: dict[str, Any]) -> dict[str, Any]:
        record_count = 0
        with open(path, encoding="utf-8") as f:
            for i, _ in enumerate(f):
                if i % 4 == 0:
                    record_count += 1
                    if record_count > 10000:
                        break
        detected.update({
            "detected_type": "fastq",
            "read_count_approx": record_count,
            "estimated_records": record_count if record_count <= 10000 else f">{record_count}",
        })
        return {"success": True, **detected}

    def _parse_gff(self, path: Path, detected: dict[str, Any]) -> dict[str, Any]:
        feature_count = 0
        feature_types: dict[str, int] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    ft = parts[2]
                    feature_types[ft] = feature_types.get(ft, 0) + 1
                    feature_count += 1
        detected.update({
            "detected_type": "gff",
            "feature_count": feature_count,
            "feature_types": feature_types,
        })
        return {"success": True, **detected}

    def _parse_bed(self, path: Path, suffix: str, detected: dict[str, Any]) -> dict[str, Any]:
        interval_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("track"):
                    continue
                interval_count += 1
                if interval_count > 5000:
                    break
        bed_type = {
            ".narrowPeak": "narrowPeak",
            ".broadPeak": "broadPeak",
            ".bed": "bed",
        }.get(suffix, "bed")
        detected.update({
            "detected_type": bed_type,
            "interval_count_approx": interval_count,
        })
        return {"success": True, **detected}

    async def _parse_anndata(self, path: Path, detected: dict[str, Any]) -> dict[str, Any]:
        try:
            import anndata

            adata = anndata.read_h5ad(path)
            detected.update({
                "detected_type": "anndata",
                "shape": list(adata.shape),
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "obs_keys": list(adata.obs.columns)[:30],
                "var_keys": list(adata.var.columns)[:30] if adata.var.columns.tolist() else [],
                "layers": list(adata.layers.keys()) if adata.layers else [],
            })
            return {"success": True, **detected}
        except ImportError:
            return {"success": False, "error": "anndata package not installed", "detected": detected}


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _guess_omic_type(df: pd.DataFrame) -> str:
    """Guess the omics type from column names and data patterns."""
    col_str = " ".join(df.columns.astype(str).str.lower())
    # Check first column name patterns
    first_col = str(df.columns[0]).lower()
    if any(kw in first_col for kw in ("gene", "symbol", "ensembl", "entrez")):
        if df.shape[1] >= 3:
            return "transcriptomics"
        return "genomics"
    if any(kw in first_col for kw in ("protein", "uniprot")):
        return "proteomics"
    if any(kw in first_col for kw in ("metabolite", "compound", "kegg", "hmdb", "pubchem")):
        return "metabolomics"
    if any(kw in first_col for kw in ("cpg", "probe", "methylation")):
        return "epigenetics"
    if any(kw in first_col for kw in ("patient", "sample", "subject", "id")):
        return "clinical"
    # Check column content
    if any(kw in col_str for kw in ("survival", "os_status", "pfs", "time_to_event")):
        return "clinical"
    if any(kw in col_str for kw in ("fpkm", "rpkm", "tpm", "count")):
        return "transcriptomics"
    return "unknown"


# ═════════════════════════════════════════════════════════════════════
# export_results
# ═════════════════════════════════════════════════════════════════════

class ExportResultsParams(BaseModel):
    data: list[dict[str, Any]] | None = Field(default=None, description="Tabular data to export")
    path: str = Field(..., description="Output file path")
    format: str = Field(default="csv", description="Export format: csv, tsv, json, xlsx, html")


class ExportResultsTool(CallableTool[ExportResultsParams]):
    name = "export_results"
    description = "Export analysis results to various formats (CSV, TSV, JSON, Excel, HTML)."
    params_model = ExportResultsParams

    async def execute(self, params: ExportResultsParams, **kwargs: Any) -> dict[str, Any]:
        path = Path(params.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if params.data is None:
            return {"success": False, "error": "No data provided for export"}

        df = pd.DataFrame(params.data)

        try:
            if params.format == "csv":
                df.to_csv(path, index=False)
            elif params.format == "tsv":
                df.to_csv(path, sep="\t", index=False)
            elif params.format == "json":
                df.to_json(path, orient="records", indent=2, force_ascii=False)
            elif params.format == "xlsx":
                df.to_excel(path, index=False, engine="openpyxl")
            elif params.format == "html":
                html = df.to_html(index=False, classes="table table-striped")
                path.write_text(html, encoding="utf-8")
            else:
                return {"success": False, "error": f"Unsupported export format: {params.format}"}

            return {
                "success": True,
                "path": str(path),
                "rows": len(df),
                "columns": len(df.columns),
                "size_bytes": path.stat().st_size,
            }
        except Exception as e:
            return {"success": False, "error": f"Export failed: {e}"}
