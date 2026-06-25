"""Data format registry for omics file formats.

Maps file extensions and content patterns to omics domains and format descriptions.
"""

from __future__ import annotations

from typing import Any

# ═════════════════════════════════════════════════════════════════════
# Format definitions: (suffixes, domain, description, parser hint)
# ═════════════════════════════════════════════════════════════════════

FORMAT_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Genomics ──
    "fasta": {
        "suffixes": [".fasta", ".fa", ".fna", ".ffn", ".faa"],
        "domain": "genomics",
        "description": "FASTA nucleotide/peptide sequences",
        "parser": "fasta",
    },
    "fastq": {
        "suffixes": [".fastq", ".fq"],
        "domain": "genomics",
        "description": "FASTQ raw sequencing reads with quality scores",
        "parser": "fastq",
    },
    "sam": {
        "suffixes": [".sam"],
        "domain": "genomics",
        "description": "Sequence Alignment/Map (text alignment format)",
        "parser": "alignment",
    },
    "bam": {
        "suffixes": [".bam"],
        "domain": "genomics",
        "description": "Binary Alignment Map (compressed alignment)",
        "parser": "bam",
    },
    "cram": {
        "suffixes": [".cram"],
        "domain": "genomics",
        "description": "Ultra-compressed alternative to BAM",
        "parser": "bam",
    },
    "vcf": {
        "suffixes": [".vcf", ".vcf.gz"],
        "domain": "genomics",
        "description": "Variant Call Format (SNPs, indels, CNVs)",
        "parser": "vcf",
    },
    "bcf": {
        "suffixes": [".bcf"],
        "domain": "genomics",
        "description": "Binary VCF",
        "parser": "vcf",
    },
    "gff": {
        "suffixes": [".gff", ".gff3", ".gtf"],
        "domain": "genomics",
        "description": "Gene annotation format",
        "parser": "gff",
    },
    "bed": {
        "suffixes": [".bed"],
        "domain": "genomics",
        "description": "Browser Extensible Data (genomic intervals)",
        "parser": "bed",
    },
    "plink": {
        "suffixes": [".bed", ".bim", ".fam"],
        "domain": "genomics",
        "description": "PLINK GWAS genotype format",
        "parser": "plink",
    },
    "maf": {
        "suffixes": [".maf"],
        "domain": "genomics",
        "description": "Mutation Annotation Format",
        "parser": "maf",
    },

    # ── Epigenetics ──
    "narrowpeak": {
        "suffixes": [".narrowPeak"],
        "domain": "epigenetics",
        "description": "ENCODE narrowPeak (ChIP-seq/ATAC-seq peak calls)",
        "parser": "bed",
    },
    "broadpeak": {
        "suffixes": [".broadPeak"],
        "domain": "epigenetics",
        "description": "ENCODE broadPeak (broad ChIP-seq peaks)",
        "parser": "bed",
    },
    "bigwig": {
        "suffixes": [".bigWig", ".bw"],
        "domain": "epigenetics",
        "description": "BigWig coverage track (binary indexed)",
        "parser": "bigwig",
    },
    "bedgraph": {
        "suffixes": [".bedGraph", ".bg"],
        "domain": "epigenetics",
        "description": "BedGraph coverage/wiggle track",
        "parser": "bed",
    },
    "bismark_cov": {
        "suffixes": [".cov"],
        "domain": "epigenetics",
        "description": "Bismark methylation coverage output",
        "parser": "tsv",
    },
    "hic": {
        "suffixes": [".hic"],
        "domain": "epigenetics",
        "description": "Hi-C chromatin contact matrix",
        "parser": "hic",
    },
    "cool": {
        "suffixes": [".cool", ".mcool"],
        "domain": "epigenetics",
        "description": "Cooler Hi-C contact matrix",
        "parser": "cool",
    },

    # ── Transcriptomics ──
    "count_matrix": {
        "suffixes": [".csv", ".tsv", ".txt"],
        "domain": "transcriptomics",
        "description": "Gene expression count matrix (genes × samples)",
        "parser": "matrix",
        "column_pattern": ["gene", "symbol", "ensembl", "entrez"],
    },
    "h5ad": {
        "suffixes": [".h5ad"],
        "domain": "transcriptomics",
        "description": "AnnData (single-cell genomics data)",
        "parser": "h5ad",
    },
    "gct": {
        "suffixes": [".gct"],
        "domain": "transcriptomics",
        "description": "GCT format for GSEA (GenePattern)",
        "parser": "gct",
    },
    "cls": {
        "suffixes": [".cls"],
        "domain": "transcriptomics",
        "description": "CLS class label file for GSEA",
        "parser": "cls",
    },

    # ── Proteomics ──
    "mzml": {
        "suffixes": [".mzML"],
        "domain": "proteomics",
        "description": "Standard mass spectrometry data format (HUPO-PSI)",
        "parser": "mzml",
    },
    "mzxml": {
        "suffixes": [".mzXML"],
        "domain": "proteomics",
        "description": "Legacy mass spectrometry data format",
        "parser": "mzxml",
    },
    "mgf": {
        "suffixes": [".mgf"],
        "domain": "proteomics",
        "description": "Mascot Generic Format (peak lists)",
        "parser": "mgf",
    },
    "mzid": {
        "suffixes": [".mzID", ".mzid"],
        "domain": "proteomics",
        "description": "Peptide/protein identification results",
        "parser": "mzid",
    },
    "mztab": {
        "suffixes": [".mzTab"],
        "domain": "proteomics",
        "description": "Summary peptide/protein quantification table",
        "parser": "tsv",
    },
    "pdb": {
        "suffixes": [".pdb", ".pdb.gz"],
        "domain": "proteomics",
        "description": "Protein Data Bank 3D structure",
        "parser": "pdb",
    },

    # ── Metabolomics ──
    "netcdf": {
        "suffixes": [".cdf", ".nc"],
        "domain": "metabolomics",
        "description": "NetCDF (GC-MS/LC-MS data)",
        "parser": "netcdf",
    },
    "isa_tab": {
        "suffixes": [".isa.txt"],
        "domain": "metabolomics",
        "description": "ISA-Tab metabolomics study metadata",
        "parser": "isa",
    },

    # ── Clinical / Phenotype ──
    "clinical_table": {
        "suffixes": [".csv", ".tsv", ".xlsx"],
        "domain": "clinical",
        "description": "Clinical/phenotype data table",
        "parser": "matrix",
        "column_pattern": ["patient", "sample", "subject", "id", "survival", "status"],
    },

    # ── General Tabular ──
    "tabular": {
        "suffixes": [".csv", ".tsv", ".xlsx", ".xls"],
        "domain": "unknown",
        "description": "Generic tabular data",
        "parser": "matrix",
    },

    # ── Metadata ──
    "yaml_config": {
        "suffixes": [".yaml", ".yml"],
        "domain": "meta",
        "description": "YAML configuration or metadata",
        "parser": "yaml",
    },
    "json_config": {
        "suffixes": [".json"],
        "domain": "meta",
        "description": "JSON data or configuration",
        "parser": "json",
    },
}


def detect_format(file_path: str) -> dict[str, Any]:
    """Detect the omics data format from a file path.

    Returns format info with domain, description, and parser hint.
    Handles compound extensions (e.g. .vcf.gz, .pdb.gz).
    """
    from pathlib import Path

    path = Path(file_path)
    name = path.name.lower()

    # Collect all suffixes from registry (lowercased for comparison)
    # Sort by length descending so compound extensions (.vcf.gz) match before simple ones (.gz)
    all_entries: list[tuple[str, str]] = []
    for fmt_name, fmt_info in FORMAT_REGISTRY.items():
        for s in fmt_info["suffixes"]:
            all_entries.append((s.lower(), fmt_name))
    all_entries.sort(key=lambda x: len(x[0]), reverse=True)

    for suffix_lower, fmt_name in all_entries:
        if name.endswith(suffix_lower):
            fmt_info = FORMAT_REGISTRY[fmt_name]
            return {
                "format": fmt_name,
                "domain": fmt_info["domain"],
                "description": fmt_info["description"],
                "parser": fmt_info["parser"],
            }

    return {
        "format": "unknown",
        "domain": "unknown",
        "description": f"Unrecognized format: {path.suffix.lower()}",
        "parser": "raw",
    }


def get_formats_by_domain(domain: str) -> list[dict[str, Any]]:
    """Get all formats for a specific omics domain."""
    result: list[dict[str, Any]] = []
    for fmt_name, fmt_info in FORMAT_REGISTRY.items():
        if fmt_info["domain"] == domain:
            result.append({
                "format": fmt_name,
                "suffixes": fmt_info["suffixes"],
                "description": fmt_info["description"],
            })
    return result


def list_all_domains() -> list[str]:
    """List all known omics domains."""
    domains = sorted({v["domain"] for v in FORMAT_REGISTRY.values()})
    return [d for d in domains if d != "meta" and d != "unknown"]


def list_all_formats() -> dict[str, list[dict[str, Any]]]:
    """List all formats organized by domain."""
    catalog: dict[str, list[dict[str, Any]]] = {}
    for fmt_name, fmt_info in FORMAT_REGISTRY.items():
        domain = fmt_info["domain"]
        if domain not in catalog:
            catalog[domain] = []
        catalog[domain].append({
            "format": fmt_name,
            "suffixes": fmt_info["suffixes"],
            "description": fmt_info["description"],
        })
    return catalog
