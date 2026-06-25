"""TDD-style unit tests for format detection and domain lookup."""

from __future__ import annotations

import pytest

from passi.knowledge.formats import (
    detect_format,
    get_formats_by_domain,
    list_all_domains,
    list_all_formats,
)


class TestDetectFormat:
    """Tests for format auto-detection from file extensions."""

    @pytest.mark.parametrize(
        "filepath, expected_format, expected_domain",
        [
            ("data.fastq", "fastq", "genomics"),
            ("data.vcf", "vcf", "genomics"),
            ("data.vcf.gz", "vcf", "genomics"),
            ("data.bam", "bam", "genomics"),
            ("data.fasta", "fasta", "genomics"),
            ("data.narrowPeak", "narrowpeak", "epigenetics"),
            ("data.broadPeak", "broadpeak", "epigenetics"),
            ("data.bigWig", "bigwig", "epigenetics"),
            ("data.h5ad", "h5ad", "transcriptomics"),
            ("data.mzML", "mzml", "proteomics"),
            ("data.mgf", "mgf", "proteomics"),
            ("data.pdb", "pdb", "proteomics"),
        ],
    )
    def test_detect_known_format(self, filepath, expected_format, expected_domain):
        # Act
        result = detect_format(filepath)

        # Assert
        assert result["format"] == expected_format
        assert result["domain"] == expected_domain

    def test_detect_unknown_format_returns_unknown(self):
        # Act
        result = detect_format("data.xyz")

        # Assert
        assert result["format"] == "unknown"
        assert result["domain"] == "unknown"

    def test_detect_csv_returns_tabular(self):
        # Act
        result = detect_format("data.csv")

        # Assert
        assert result["format"] in ("count_matrix", "tabular")


class TestFormatCatalog:
    """Tests for format catalog querying."""

    def test_list_all_domains_includes_major_omics_types(self):
        # Act
        domains = list_all_domains()

        # Assert
        assert "genomics" in domains
        assert "transcriptomics" in domains
        assert "epigenetics" in domains
        assert "proteomics" in domains
        assert "metabolomics" in domains
        assert "clinical" in domains

    def test_get_formats_by_domain_returns_correct_formats(self):
        # Act
        genomics_formats = get_formats_by_domain("genomics")

        # Assert
        format_names = [f["format"] for f in genomics_formats]
        assert "fastq" in format_names
        assert "vcf" in format_names
        assert "bam" in format_names

    def test_list_all_formats_is_organized_by_domain(self):
        # Act
        catalog = list_all_formats()

        # Assert
        assert isinstance(catalog, dict)
        assert "genomics" in catalog
        assert len(catalog["genomics"]) > 0
