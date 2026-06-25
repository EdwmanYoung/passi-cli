"""TDD-style unit tests for I/O tools.

Each test follows AAA pattern: Arrange → Act → Assert.
Test naming: test_<method>_<condition>_<expected_result>
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from passi.tools.io_tools import (
    ExportResultsTool,
    ExportResultsParams,
    ParseOmicsDataTool,
    ParseOmicsDataParams,
    ReadFileTool,
    ReadFileParams,
    WriteFileTool,
    WriteFileParams,
)

# ═══════════════════════════════════════════════════════════════
# ReadFileTool tests
# ═══════════════════════════════════════════════════════════════


class TestReadFileTool:
    """Unit tests for ReadFileTool."""

    @pytest.mark.asyncio
    async def test_read_csv_returns_shape_and_preview(self, tmp_path: Path):
        # Arrange
        tool = ReadFileTool()
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv_path, index=False)
        params = ReadFileParams(path=str(csv_path))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["format"] == "csv"
        assert result["shape"] == [3, 2]
        assert result["columns"] == ["a", "b"]
        assert len(result["preview"]) == 3

    @pytest.mark.asyncio
    async def test_read_nonexistent_file_returns_error(self):
        # Arrange
        tool = ReadFileTool()
        params = ReadFileParams(path="/nonexistent/file.csv")

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_read_tsv_returns_correct_format(self, tmp_path: Path):
        # Arrange
        tool = ReadFileTool()
        tsv_path = tmp_path / "test.tsv"
        pd.DataFrame({"x": [10], "y": [20]}).to_csv(tsv_path, sep="\t", index=False)
        params = ReadFileParams(path=str(tsv_path))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["format"] == "tsv"

    @pytest.mark.asyncio
    async def test_read_with_max_lines_limits_output(self, tmp_path: Path):
        # Arrange
        tool = ReadFileTool()
        csv_path = tmp_path / "big.csv"
        pd.DataFrame({"v": range(100)}).to_csv(csv_path, index=False)
        params = ReadFileParams(path=str(csv_path), max_lines=5)

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert len(result["preview"]) == 5


# ═══════════════════════════════════════════════════════════════
# WriteFileTool tests
# ═══════════════════════════════════════════════════════════════


class TestWriteFileTool:
    """Unit tests for WriteFileTool."""

    @pytest.mark.asyncio
    async def test_write_csv_creates_file(self, tmp_path: Path):
        # Arrange
        tool = WriteFileTool()
        out_path = tmp_path / "output.csv"
        params = WriteFileParams(
            path=str(out_path),
            format="csv",
            data=[{"gene": "BRCA1", "log2FC": 2.5}],
        )

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_write_json_creates_valid_json(self, tmp_path: Path):
        # Arrange
        tool = WriteFileTool()
        out_path = tmp_path / "output.json"
        params = WriteFileParams(
            path=str(out_path),
            format="json",
            data=[{"key": "value"}],
        )

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        content = out_path.read_text()
        assert '"key"' in content

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(self, tmp_path: Path):
        # Arrange
        tool = WriteFileTool()
        out_path = tmp_path / "sub" / "nested" / "file.txt"
        params = WriteFileParams(path=str(out_path), content="data")

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert out_path.exists()


# ═══════════════════════════════════════════════════════════════
# ParseOmicsDataTool tests
# ═══════════════════════════════════════════════════════════════


class TestParseOmicsDataTool:
    """Unit tests for ParseOmicsDataTool."""

    @pytest.mark.asyncio
    async def test_parse_count_matrix_detects_transcriptomics(self, tmp_path: Path):
        # Arrange
        tool = ParseOmicsDataTool()
        csv_path = tmp_path / "counts.csv"
        pd.DataFrame({
            "gene_symbol": ["BRCA1", "TP53", "EGFR"],
            "S1": [100, 200, 150],
            "S2": [110, 190, 160],
            "S3": [90, 210, 140],
        }).to_csv(csv_path, index=False)
        params = ParseOmicsDataParams(path=str(csv_path))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["detected_type"] == "count_matrix"
        assert result["omic_hint"] == "transcriptomics"

    @pytest.mark.asyncio
    async def test_parse_vcf_returns_variant_count(self, tmp_path: Path):
        # Arrange
        tool = ParseOmicsDataTool()
        vcf_path = tmp_path / "test.vcf"
        vcf_path.write_text(
            "##fileformat=VCFv4.2\n"
            "##INFO=<ID=DP,Number=1,Type=Integer>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t100\t.\tA\tG\t100\tPASS\t.\n"
            "chr1\t200\t.\tC\tT\t80\tPASS\t.\n"
        )
        params = ParseOmicsDataParams(path=str(vcf_path))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["detected_type"] == "vcf"
        assert result["variant_count_preview"] == 2

    @pytest.mark.asyncio
    async def test_parse_fasta_returns_sequence_count(self, tmp_path: Path):
        # Arrange
        tool = ParseOmicsDataTool()
        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(
            ">seq1\nATCGATCG\n>seq2\nGGCTAGCT\n>seq3\nTTAGCCAA\n"
        )
        params = ParseOmicsDataParams(path=str(fasta_path))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["detected_type"] == "fasta"
        assert result["sequence_count"] == 3

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file_returns_error(self):
        # Arrange
        tool = ParseOmicsDataTool()
        params = ParseOmicsDataParams(path="/nonexistent.csv")

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_parse_bed_detects_interval_format(self, tmp_path: Path):
        # Arrange
        tool = ParseOmicsDataTool()
        bed_path = tmp_path / "peaks.bed"
        bed_path.write_text(
            "chr1\t100\t500\tpeak1\t100\t+\n"
            "chr1\t600\t900\tpeak2\t80\t-\n"
        )
        params = ParseOmicsDataParams(path=str(bed_path))

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["detected_type"] == "bed"
        assert result["interval_count_approx"] == 2


# ═══════════════════════════════════════════════════════════════
# ExportResultsTool tests
# ═══════════════════════════════════════════════════════════════


class TestExportResultsTool:
    """Unit tests for ExportResultsTool."""

    @pytest.mark.asyncio
    async def test_export_csv_with_data_creates_file(self, tmp_path: Path):
        # Arrange
        tool = ExportResultsTool()
        out_path = tmp_path / "results.csv"
        data = [{"gene": "A", "fdr": 0.01}, {"gene": "B", "fdr": 0.05}]
        params = ExportResultsParams(path=str(out_path), format="csv", data=data)

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert result["rows"] == 2
        assert result["columns"] == 2

    @pytest.mark.asyncio
    async def test_export_without_data_returns_error(self, tmp_path: Path):
        # Arrange
        tool = ExportResultsTool()
        params = ExportResultsParams(
            path=str(tmp_path / "empty.csv"),
            data=None,
        )

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is False
        assert "No data" in result["error"]

    @pytest.mark.asyncio
    async def test_export_html_creates_html_file(self, tmp_path: Path):
        # Arrange
        tool = ExportResultsTool()
        out_path = tmp_path / "report.html"
        params = ExportResultsParams(
            path=str(out_path),
            format="html",
            data=[{"col": "val"}],
        )

        # Act
        result = await tool.execute(params)

        # Assert
        assert result["success"] is True
        assert out_path.read_text().startswith("<table")
