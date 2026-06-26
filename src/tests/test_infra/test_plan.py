"""TDD-style unit tests for AnalysisPlan, PlanStep, PlanManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.infra.plan import (
    AnalysisPlan,
    PlanManager,
    PlanStatus,
    PlanStep,
    StepStatus,
    _slugify,
)


class TestPlanStep:
    """PlanStep model defaults and construction."""

    def test_default_step(self):
        step = PlanStep(step_id="s1", order=1, description="Load data")
        assert step.step_id == "s1"
        assert step.order == 1
        assert step.description == "Load data"
        assert step.status == StepStatus.PENDING
        assert step.tool_name == ""
        assert step.expected_params == {}

    def test_step_with_tool(self):
        step = PlanStep(
            step_id="s2",
            order=2,
            description="Run QC",
            tool_name="qc_report",
            expected_params={"data_path": "/data.csv"},
        )
        assert step.tool_name == "qc_report"
        assert step.expected_params["data_path"] == "/data.csv"

    def test_step_status_transitions(self):
        step = PlanStep(step_id="s1", order=1, description="Test")
        assert step.status == StepStatus.PENDING
        step.status = StepStatus.RUNNING
        assert step.status == StepStatus.RUNNING
        step.status = StepStatus.DONE
        assert step.status == StepStatus.DONE


class TestAnalysisPlan:
    """AnalysisPlan model construction and defaults."""

    def test_minimal_plan(self):
        plan = AnalysisPlan(plan_id="p1", title="Test Plan")
        assert plan.plan_id == "p1"
        assert plan.title == "Test Plan"
        assert plan.status == PlanStatus.DRAFT
        assert plan.steps == []
        assert plan.description == ""

    def test_plan_with_steps(self):
        steps = [
            PlanStep(step_id="p1_s1", order=1, description="Step 1"),
            PlanStep(step_id="p1_s2", order=2, description="Step 2", tool_name="read_file"),
        ]
        plan = AnalysisPlan(
            plan_id="p1",
            title="Multi-step plan",
            description="A test plan",
            domain="transcriptomics",
            steps=steps,
            session_id="sess-001",
        )
        assert len(plan.steps) == 2
        assert plan.domain == "transcriptomics"
        assert plan.session_id == "sess-001"


class TestPlanStatusEnum:
    """PlanStatus and StepStatus enum values."""

    def test_plan_status_values(self):
        assert PlanStatus.DRAFT.value == "draft"
        assert PlanStatus.APPROVED.value == "approved"
        assert PlanStatus.RUNNING.value == "running"
        assert PlanStatus.DONE.value == "done"
        assert PlanStatus.FAILED.value == "failed"

    def test_step_status_values(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.DONE.value == "done"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


class TestPlanManager:
    """PlanManager — plan lifecycle and persistence."""

    def test_create_plan(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        plan = pm.create_plan(
            title="RNA-seq DE Analysis",
            description="Differential expression analysis workflow",
            domain="transcriptomics",
            steps=[
                {"description": "Load count matrix", "tool_name": "read_file"},
                {"description": "Run DESeq2", "tool_name": "run_r"},
                {"description": "Generate volcano plot", "tool_name": "run_python"},
            ],
        )
        assert plan.title == "RNA-seq DE Analysis"
        assert len(plan.steps) == 3
        assert plan.steps[0].tool_name == "read_file"
        assert plan.status == PlanStatus.DRAFT
        assert plan.plan_id.startswith("plan_")

    def test_create_plan_persists_to_yaml(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Test", description="Desc")
        assert (tmp_path / "plan.yaml").exists()

    def test_get_plan_returns_none_when_no_plan(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        assert pm.get_plan() is None

    def test_get_plan_after_create(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="My Plan")
        plan = pm.get_plan()
        assert plan is not None
        assert plan.title == "My Plan"

    def test_update_step_status(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(
            title="Plan",
            steps=[{"description": "Step 1"}, {"description": "Step 2"}],
        )
        plan = pm.get_plan()
        assert plan is not None

        updated = pm.update_step_status(plan.steps[0].step_id, StepStatus.RUNNING)
        assert updated is not None
        assert updated.status == StepStatus.RUNNING
        assert updated.started_at != ""

    def test_update_step_status_done(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan", steps=[{"description": "Step 1"}])
        plan = pm.get_plan()
        assert plan is not None

        pm.update_step_status(plan.steps[0].step_id, StepStatus.DONE, output_summary="Completed successfully.")
        step = pm.get_plan().steps[0]  # type: ignore[union-attr]
        assert step.status == StepStatus.DONE
        assert step.output_summary == "Completed successfully."
        assert step.completed_at != ""

    def test_update_step_status_failed(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan", steps=[{"description": "Step 1"}])
        plan = pm.get_plan()
        assert plan is not None

        pm.update_step_status(
            plan.steps[0].step_id,
            StepStatus.FAILED,
            error_message="Missing data file",
        )
        step = pm.get_plan().steps[0]  # type: ignore[union-attr]
        assert step.status == StepStatus.FAILED
        assert step.error_message == "Missing data file"

    def test_update_step_status_unknown_step(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan")
        result = pm.update_step_status("nonexistent", StepStatus.DONE)
        assert result is None

    def test_update_plan_status(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan")
        pm.update_plan_status(PlanStatus.RUNNING)
        assert pm.get_plan().status == PlanStatus.RUNNING  # type: ignore[union-attr]

    def test_get_current_step(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(
            title="Plan",
            steps=[
                {"description": "Step 1"},
                {"description": "Step 2"},
            ],
        )
        step = pm.get_current_step()
        assert step is not None
        assert step.order == 1

    def test_get_current_step_after_first_done(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(
            title="Plan",
            steps=[
                {"description": "Step 1"},
                {"description": "Step 2"},
            ],
        )
        plan = pm.get_plan()
        assert plan is not None
        pm.update_step_status(plan.steps[0].step_id, StepStatus.DONE)

        current = pm.get_current_step()
        assert current is not None
        assert current.order == 2

    def test_load_plan_from_file(self, tmp_path: Path):
        pm1 = PlanManager(tmp_path)
        pm1.create_plan(title="Saved Plan", steps=[{"description": "S1"}])

        # New manager in same directory should load existing plan
        pm2 = PlanManager(tmp_path)
        loaded = pm2.load_plan()
        assert loaded is not None
        assert loaded.title == "Saved Plan"
        assert len(loaded.steps) == 1

    def test_load_plan_no_file(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        assert pm.load_plan() is None

    def test_step_id_uses_descriptive_name(self, tmp_path: Path):
        """step_id follows pattern step_NN_slugified_name (not plan_UUID_step_NN)."""
        pm = PlanManager(tmp_path)
        pm.create_plan(
            title="Test",
            steps=[
                {"description": "Data Inspection"},
                {"description": "Preprocessing & Normalization"},
                {"description": "Differential Analysis"},
            ],
        )
        plan = pm.get_plan()
        assert plan is not None
        assert plan.steps[0].step_id == "step_01_data_inspection"
        assert plan.steps[1].step_id == "step_02_preprocessing_normalization"
        assert plan.steps[2].step_id == "step_03_differential_analysis"

    def test_step_id_without_description_fallback(self, tmp_path: Path):
        """Empty description falls back to step_NN_step_NN."""
        pm = PlanManager(tmp_path)
        pm.create_plan(
            title="Test",
            steps=[{"description": ""}],
        )
        plan = pm.get_plan()
        assert plan is not None
        assert plan.steps[0].step_id == "step_01_step_01"


class TestSlugify:
    """Unit tests for _slugify helper."""

    def test_basic_lowercase(self):
        assert _slugify("Data Inspection") == "data_inspection"

    def test_special_chars_replaced(self):
        assert _slugify("ROC & AUC Analysis") == "roc_auc_analysis"

    def test_multi_spaces_collapsed(self):
        assert _slugify("Run   DESeq2") == "run_deseq2"

    def test_leading_trailing_underscores_stripped(self):
        assert _slugify("  Hello World  ") == "hello_world"

    def test_truncate_to_max_len(self):
        long_text = "comprehensive differential expression analysis of rna seq data"
        result = _slugify(long_text, max_len=30)
        assert len(result) <= 30
        assert not result.endswith("_")

    def test_non_alpha_numeric_removed(self):
        assert _slugify("Step #1: QC") == "step_1_qc"
