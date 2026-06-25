"""Root test fixtures for PassiAgent.

Provides shared fixtures at session, module, and function scope.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from passi.config import PassiConfig


# ═══════════════════════════════════════════════════════════════
# Session-scoped fixtures (expensive, shared across all tests)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def test_config() -> PassiConfig:
    """A PassiConfig with in-memory session and test defaults."""
    return PassiConfig(
        anthropic={"api_key": "test-key", "model": "claude-sonnet-4-6"},
        openai={"api_key": "test-key", "model": "gpt-4o"},
        default_provider="anthropic",
        execution={"timeout_seconds": 10},
        session={"sessions_dir": Path(tempfile.mkdtemp())},
        debug=True,
    )


# ═══════════════════════════════════════════════════════════════
# Function-scoped fixtures (isolated per-test)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def temp_session_dir(tmp_path: Path) -> Path:
    """Create an isolated session directory per test."""
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create an isolated data directory per test."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# ═══════════════════════════════════════════════════════════════
# Test data factories
# ═══════════════════════════════════════════════════════════════

def make_csv(
    dir_path: Path,
    filename: str = "test_data.csv",
    rows: int = 10,
    cols: int = 5,
) -> Path:
    """Create a CSV file with synthetic data."""
    df = pd.DataFrame(
        {f"col_{i}": range(rows) for i in range(cols)}
        for _ in range(rows)
    )
    filepath = dir_path / filename
    df.to_csv(filepath, index=False)
    return filepath


def make_tsv(dir_path: Path, filename: str = "test_data.tsv") -> Path:
    """Create a TSV file."""
    filepath = dir_path / filename
    pd.DataFrame({"gene": ["BRCA1", "TP53"], "sample_1": [10, 20]}).to_csv(
        filepath, sep="\t", index=False
    )
    return filepath


def make_vcf(dir_path: Path, filename: str = "test_variants.vcf") -> Path:
    """Create a minimal VCF file."""
    filepath = dir_path / filename
    content = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t100\t.\tA\tG\t100\tPASS\t."
    )
    filepath.write_text(content)
    return filepath


def make_count_matrix(
    dir_path: Path,
    filename: str = "counts.csv",
    n_genes: int = 100,
    n_samples: int = 6,
) -> Path:
    """Create a synthetic RNA-seq count matrix."""
    genes = [f"GENE{i}" for i in range(1, n_genes + 1)]
    samples = [f"SAMPLE_{i}" for i in range(1, n_samples + 1)]
    data: dict[str, list[float]] = {}
    for gene in genes:
        data[gene] = [float(i * 10 + j) for j, _ in enumerate(samples)]
    df = pd.DataFrame(data)
    filepath = dir_path / filename
    df.to_csv(filepath, index=False)
    return filepath


def make_clinical_data(dir_path: Path) -> Path:
    """Create synthetic clinical/phenotype data with survival columns."""
    filepath = dir_path / "clinical.csv"
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "age": [55, 62, 48],
        "sex": ["M", "F", "M"],
        "treatment": ["DrugA", "DrugB", "Placebo"],
        "os_days": [365, 180, 90],
        "os_event": [1, 0, 1],
    })
    df.to_csv(filepath, index=False)
    return filepath
