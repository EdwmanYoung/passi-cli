"""Test data factories for creating controlled datasets.

Each factory returns a Path to a temporary file with known structure,
enabling deterministic assertions in tests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_count_matrix(dir_path: Path, n_genes: int = 10, n_samples: int = 4) -> Path:
    """Build a synthetic count matrix (genes × samples) with known structure.

    - genes: GENE1..GENE10
    - samples: S1..S4
    - values: row * 10 + col (deterministic)
    """
    genes = [f"GENE{i}" for i in range(1, n_genes + 1)]
    columns = ["gene_id"] + [f"S{i}" for i in range(1, n_samples + 1)]
    data: list[list] = []
    for i, gene in enumerate(genes):
        row = [gene] + [i * 10 + j for j in range(1, n_samples + 1)]
        data.append(row)
    df = pd.DataFrame(data, columns=columns)
    path = dir_path / "test_count_matrix.csv"
    df.to_csv(path, index=False)
    return path


def build_sample_metadata(dir_path: Path) -> Path:
    """Build sample metadata with two groups (Control/Treatment)."""
    df = pd.DataFrame({
        "sample_id": ["S1", "S2", "S3", "S4"],
        "condition": ["Control", "Control", "Treatment", "Treatment"],
        "batch": ["A", "A", "B", "B"],
    })
    path = dir_path / "test_metadata.csv"
    df.to_csv(path, index=False)
    return path


def build_clinical_survival(dir_path: Path, n_patients: int = 20) -> Path:
    """Build synthetic clinical data with survival columns.

    - time: uniformly distributed 30-1095 days
    - event: random 0/1
    - group: High/Low for KM comparison
    """
    import numpy as np

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(1, n_patients + 1)],
        "time": rng.integers(30, 1095, n_patients),
        "event": rng.integers(0, 2, n_patients),
        "group": rng.choice(["High", "Low"], n_patients),
        "age": rng.normal(60, 10, n_patients).astype(int),
        "sex": rng.choice(["M", "F"], n_patients),
    })
    path = dir_path / "test_clinical.csv"
    df.to_csv(path, index=False)
    return path


def build_gene_list(dir_path: Path, n_genes: int = 50) -> Path:
    """Build a list of differentially expressed genes."""
    genes = [f"GENE{i}" for i in range(1, n_genes + 1)]
    df = pd.DataFrame({
        "gene": genes,
        "log2FC": [2.5 if i % 2 == 0 else -2.5 for i in range(1, n_genes + 1)],
        "pvalue": [0.001 / (i + 1) for i in range(n_genes)],
        "padj": [0.05 / (i + 1) for i in range(n_genes)],
    })
    path = dir_path / "test_de_genes.csv"
    df.to_csv(path, index=False)
    return path


def build_protein_matrix(dir_path: Path) -> Path:
    """Build a synthetic protein quantification matrix."""
    proteins = [f"PROT{i}" for i in range(1, 21)]
    columns = ["protein_id"] + [f"S{i}" for i in range(1, 7)]
    data: list[list] = []
    for i, prot in enumerate(proteins):
        row = [prot] + [15.0 + i * 0.3 + j * 0.5 for j in range(1, 7)]
        data.append(row)
    df = pd.DataFrame(data, columns=columns)
    path = dir_path / "test_proteomics.csv"
    df.to_csv(path, index=False)
    return path


def build_metabolite_matrix(dir_path: Path) -> Path:
    """Build a synthetic metabolite abundance table."""
    metabolites = [f"METAB{i}" for i in range(1, 31)]
    columns = ["metabolite_id"] + [f"S{i}" for i in range(1, 9)]
    data: list[list] = []
    for i, met in enumerate(metabolites):
        row = [met] + [100.0 + i * 5 + j * 2.5 for j in range(1, 9)]
        data.append(row)
    df = pd.DataFrame(data, columns=columns)
    path = dir_path / "test_metabolomics.csv"
    df.to_csv(path, index=False)
    return path
