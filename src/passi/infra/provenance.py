"""Provenance tracking for reproducible bioinformatics analysis.

Records every tool invocation, data transformation, and analysis step
so that sessions can be replayed and results reproduced.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProvenanceRecord(BaseModel):
    """A single provenance entry recording a step in the analysis pipeline."""

    step_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_name: str
    tool_params: dict[str, Any] = Field(default_factory=dict)
    input_files: list[str] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)
    input_checksums: dict[str, str] = Field(default_factory=dict)
    output_checksums: dict[str, str] = Field(default_factory=dict)
    exit_code: int = 0
    error_message: str = ""
    duration_ms: float = 0.0
    package_versions: dict[str, str] = Field(default_factory=dict)
    session_id: str = ""


class ProvenanceTracker:
    """Tracks analysis provenance for reproducibility."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[ProvenanceRecord] = []
        self._provenance_file = self.output_dir / "provenance.jsonl"

    def record_step(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        input_files: list[str] | None = None,
        output_files: list[str] | None = None,
        exit_code: int = 0,
        error_message: str = "",
        duration_ms: float = 0.0,
        session_id: str = "",
    ) -> ProvenanceRecord:
        """Record a provenance step with checksums."""
        step_id = _generate_step_id(tool_name, tool_params)
        input_files = input_files or []
        output_files = output_files or []

        record = ProvenanceRecord(
            step_id=step_id,
            tool_name=tool_name,
            tool_params=tool_params,
            input_files=input_files,
            output_files=output_files,
            input_checksums=self._compute_checksums(input_files),
            output_checksums=self._compute_checksums(output_files),
            exit_code=exit_code,
            error_message=error_message,
            duration_ms=duration_ms,
            session_id=session_id,
        )
        self._records.append(record)
        self._append_to_file(record)
        return record

    def get_records(self) -> list[ProvenanceRecord]:
        return list(self._records)

    def get_record(self, step_id: str) -> ProvenanceRecord | None:
        for r in self._records:
            if r.step_id == step_id:
                return r
        return None

    def export_report(self) -> str:
        """Generate a markdown provenance report."""
        lines = [
            "# Analysis Provenance Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Total steps: {len(self._records)}",
            "",
            "| # | Tool | Inputs | Outputs | Status | Duration |",
            "|---|------|--------|---------|--------|----------|",
        ]
        for i, r in enumerate(self._records, 1):
            status = "OK" if r.exit_code == 0 else f"ERR({r.exit_code})"
            duration = f"{r.duration_ms:.0f}ms"
            inputs = ", ".join(Path(f).name for f in r.input_files) or "-"
            outputs = ", ".join(Path(f).name for f in r.output_files) or "-"
            lines.append(f"| {i} | {r.tool_name} | {inputs} | {outputs} | {status} | {duration} |")
        report = "\n".join(lines)
        report_path = self.output_dir / "provenance_report.md"
        report_path.write_text(report, encoding="utf-8")
        return report

    def _compute_checksums(self, paths: list[str]) -> dict[str, str]:
        """Compute SHA-256 checksums for files."""
        checksums: dict[str, str] = {}
        for path_str in paths:
            p = Path(path_str)
            if p.exists() and p.is_file():
                try:
                    sha = hashlib.sha256()
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha.update(chunk)
                    checksums[str(p)] = sha.hexdigest()
                except OSError:
                    checksums[str(p)] = "ERROR"
            else:
                checksums[str(p)] = "MISSING"
        return checksums

    def _append_to_file(self, record: ProvenanceRecord) -> None:
        with open(self._provenance_file, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")


def _generate_step_id(tool_name: str, params: dict[str, Any]) -> str:
    """Generate a unique step ID from tool name and params hash."""
    params_str = json.dumps(params, sort_keys=True, default=str)
    hash_val = hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()[:12]
    return f"{tool_name}_{hash_val}"
