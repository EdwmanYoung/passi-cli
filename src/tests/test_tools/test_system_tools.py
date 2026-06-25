"""TDD-style unit tests for system tools (plan management)."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.infra.plan import PlanManager, StepStatus
from passi.tools.system_tools import (
    CreatePlanParams,
    CreatePlanTool,
    GetPlanParams,
    GetPlanTool,
    UpdatePlanStatusParams,
    UpdatePlanStatusTool,
)


class TestCreatePlanTool:
    """CreatePlanTool — creates analysis plans."""

    @pytest.mark.asyncio
    async def test_create_plan_success(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        tool = CreatePlanTool(pm)

        result = await tool.execute(CreatePlanParams(
            title="RNA-seq Analysis",
            description="Differential expression workflow",
            domain="transcriptomics",
            steps=[
                {"description": "Load counts", "tool_name": "read_file"},
                {"description": "Run DESeq2", "tool_name": "run_r"},
            ],
        ))
        assert result["success"] is True
        assert result["plan_id"].startswith("plan_")
        assert result["steps_count"] == 2
        assert len(result["steps"]) == 2
        assert result["steps"][0]["step_id"].startswith(result["plan_id"] + "_step_")
        assert result["steps"][0]["description"] == "Load counts"
        assert pm.get_plan() is not None

    @pytest.mark.asyncio
    async def test_create_plan_no_manager(self):
        tool = CreatePlanTool(plan_manager=None)
        result = await tool.execute(CreatePlanParams(title="Test"))
        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_has_required_fields(self):
        tool = CreatePlanTool(PlanManager())
        schema = tool.to_openai_schema()
        func = schema["function"]
        assert func["name"] == "create_plan"
        assert "parameters" in func
        assert "title" in func["parameters"].get("properties", {})


class TestUpdatePlanStatusTool:
    """UpdatePlanStatusTool — updates step statuses."""

    @pytest.mark.asyncio
    async def test_update_step_to_running(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan", steps=[{"description": "S1"}])
        plan = pm.get_plan()
        assert plan is not None

        tool = UpdatePlanStatusTool(pm)
        result = await tool.execute(UpdatePlanStatusParams(
            step_id=plan.steps[0].step_id,
            status="running",
        ))
        assert result["success"] is True
        assert pm.get_plan().steps[0].status == StepStatus.RUNNING  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_update_step_to_done_with_summary(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan", steps=[{"description": "S1"}])
        plan = pm.get_plan()
        assert plan is not None

        tool = UpdatePlanStatusTool(pm)
        result = await tool.execute(UpdatePlanStatusParams(
            step_id=plan.steps[0].step_id,
            status="done",
            output_summary="1234 DEGs found",
        ))
        assert result["success"] is True
        step = pm.get_plan().steps[0]  # type: ignore[union-attr]
        assert step.status == StepStatus.DONE
        assert step.output_summary == "1234 DEGs found"

    @pytest.mark.asyncio
    async def test_update_step_to_failed_with_error(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan", steps=[{"description": "S1"}])
        plan = pm.get_plan()
        assert plan is not None

        tool = UpdatePlanStatusTool(pm)
        result = await tool.execute(UpdatePlanStatusParams(
            step_id=plan.steps[0].step_id,
            status="failed",
            error_message="DESeq2 error: design matrix singular",
        ))
        assert result["success"] is True
        step = pm.get_plan().steps[0]  # type: ignore[union-attr]
        assert step.status == StepStatus.FAILED
        assert "singular" in step.error_message

    @pytest.mark.asyncio
    async def test_invalid_status(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan", steps=[{"description": "S1"}])
        plan = pm.get_plan()
        assert plan is not None

        tool = UpdatePlanStatusTool(pm)
        result = await tool.execute(UpdatePlanStatusParams(
            step_id=plan.steps[0].step_id,
            status="invalid_status",
        ))
        assert result["success"] is False
        assert "Invalid status" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_step_id(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        tool = UpdatePlanStatusTool(pm)
        result = await tool.execute(UpdatePlanStatusParams(
            step_id="nonexistent",
            status="running",
        ))
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_no_manager(self):
        tool = UpdatePlanStatusTool(plan_manager=None)
        result = await tool.execute(UpdatePlanStatusParams(step_id="x", status="running"))
        assert result["success"] is False
        assert "not initialized" in result["error"]


class TestGetPlanTool:
    """GetPlanTool — queries current plan state."""

    @pytest.mark.asyncio
    async def test_get_plan_when_no_plan_exists(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        tool = GetPlanTool(pm)

        result = await tool.execute(GetPlanParams())
        assert result["success"] is True
        assert result["has_plan"] is False

    @pytest.mark.asyncio
    async def test_get_plan_with_active_plan(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(
            title="DE Analysis",
            steps=[
                {"description": "Load data", "tool_name": "read_file"},
                {"description": "DESeq2", "tool_name": "run_r"},
            ],
        )
        tool = GetPlanTool(pm)

        result = await tool.execute(GetPlanParams())
        assert result["success"] is True
        assert result["has_plan"] is True
        assert result["title"] == "DE Analysis"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_plan_reflects_progress(self, tmp_path: Path):
        pm = PlanManager(tmp_path)
        pm.create_plan(title="Plan", steps=[{"description": "S1"}, {"description": "S2"}])
        plan = pm.get_plan()
        assert plan is not None
        pm.update_step_status(plan.steps[0].step_id, StepStatus.DONE)

        tool = GetPlanTool(pm)
        result = await tool.execute(GetPlanParams())
        assert result["steps"][0]["status"] == "done"
        assert result["steps"][1]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_no_manager(self):
        tool = GetPlanTool(plan_manager=None)
        result = await tool.execute(GetPlanParams())
        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_format(self):
        tool = GetPlanTool(PlanManager())
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "get_plan"
        assert "input_schema" in schema
