"""TDD-style unit tests for statistical methods catalog."""

from __future__ import annotations

import pytest

from passi.knowledge.methods import (
    SINGLE_OMICS_METHODS,
    MULTI_OMICS_METHODS,
    QC_METHODS,
    get_method,
    get_methods_by_domain,
    search_methods,
    list_all_methods,
)


class TestMethodLookup:
    """Tests for method catalog queries."""

    def test_get_method_returns_known_method(self):
        # Act
        method = get_method("deseq2")

        # Assert
        assert method is not None
        assert method["name"] == "DESeq2"
        assert method["backend"] == "r"
        assert method["domain"] == "transcriptomics"

    def test_get_method_returns_none_for_unknown(self):
        # Act
        method = get_method("nonexistent_method")

        # Assert
        assert method is None

    @pytest.mark.parametrize(
        "method_id, backend",
        [
            ("deseq2", "r"),
            ("edger", "r"),
            ("gsea", "python"),
            ("wgcna", "r"),
            ("km_survival", "r"),
            ("cox_ph", "r"),
            ("single_cell_scanpy", "python"),
            ("ttest", "python"),
            ("wilcoxon", "python"),
        ],
    )
    def test_known_methods_have_correct_backend(self, method_id, backend):
        # Act
        method = get_method(method_id)

        # Assert
        assert method is not None, f"Method {method_id} not found"
        assert method["backend"] == backend

    def test_all_methods_have_required_fields(self):
        # Act
        for catalog in [SINGLE_OMICS_METHODS, MULTI_OMICS_METHODS]:
            for mid, method in catalog.items():
                assert "name" in method, f"{mid} missing 'name'"
                assert "backend" in method, f"{mid} missing 'backend'"
                assert "description" in method, f"{mid} missing 'description'"
                assert method["backend"] in ("python", "r", "cli"), f"{mid} unknown backend"


class TestMethodSearch:
    """Tests for method search functionality."""

    def test_search_by_description_finds_relevant_methods(self):
        # Act
        results = search_methods("survival")

        # Assert
        result_ids = list(results.keys())
        assert "km_survival" in result_ids
        assert "cox_ph" in result_ids

    def test_search_by_domain_keyword(self):
        # Act
        results = search_methods("metabolomics")

        # Assert
        assert "metabolite_diff" in results

    def test_search_case_insensitive(self):
        # Act
        lower = search_methods("kaplan-meier")
        upper = search_methods("KAPLAN-MEIER")

        # Assert
        assert lower == upper

    def test_search_returns_empty_for_no_match(self):
        # Act
        results = search_methods("zzz_nonexistent_zzz")

        # Assert
        assert results == {}


class TestMethodCatalog:
    """Tests for method catalog organization."""

    def test_list_all_methods_includes_all_categories(self):
        # Act
        catalog = list_all_methods()

        # Assert
        assert "single_omics" in catalog
        assert "multi_omics" in catalog
        assert "qc_preprocessing" in catalog

    def test_get_methods_by_domain_transcriptomics(self):
        # Act
        methods = get_methods_by_domain("transcriptomics")

        # Assert
        assert "deseq2" in methods
        assert "edger" in methods
        assert "wgcna" in methods

    def test_get_methods_by_domain_clinical(self):
        # Act
        methods = get_methods_by_domain("clinical")

        # Assert
        assert "km_survival" in methods
        assert "cox_ph" in methods

    def test_qc_methods_have_variants(self):
        for mid, method in QC_METHODS.items():
            if "variants" in method:
                assert isinstance(method["variants"], list)
                assert len(method["variants"]) > 0
