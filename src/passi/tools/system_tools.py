"""System tools for plan management and agent introspection.

These tools allow the LLM to create structured analysis plans, update
step progress, and query plan state during execution.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from passi.infra.plan import PlanManager, PlanStatus, StepStatus
from passi.tools.base import CallableTool

logger = logging.getLogger(__name__)


class CreatePlanParams(BaseModel):
    title: str = Field(..., description="Short descriptive title for the analysis plan")
    description: str = Field(default="", description="Overview of the analysis goals and methods")
    domain: str = Field(default="", description="Analysis domain: transcriptomics, genomics, epigenetics, clinical, multi-omics, proteomics, metabolomics")
    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of plan steps, each with: description, tool_name (optional), expected_params (optional)",
    )


class CreatePlanTool(CallableTool[CreatePlanParams]):
    """Create a structured bioinformatics analysis plan.

    The plan is persisted and can be reviewed by the user before execution.
    """

    name: str = "create_plan"
    description: str = (
        "Create a structured analysis plan with ordered steps. "
        "Use this before running complex multi-step analyses so the user can review and approve. "
        "Each step should describe what analysis will be performed and which tool (if any) is needed."
    )
    params_model: type[CreatePlanParams] = CreatePlanParams

    def __init__(self, plan_manager: PlanManager | None = None) -> None:
        super().__init__()
        self._plan_manager = plan_manager

    async def execute(self, params: CreatePlanParams, **kwargs: Any) -> dict[str, Any]:
        if self._plan_manager is None:
            return {"success": False, "error": "Plan manager not initialized."}

        plan = self._plan_manager.create_plan(
            title=params.title,
            description=params.description,
            domain=params.domain,
            steps=params.steps,
        )
        steps_info = [
            {"step_id": s.step_id, "order": s.order, "description": s.description}
            for s in plan.steps
        ]
        return {
            "success": True,
            "plan_id": plan.plan_id,
            "title": plan.title,
            "steps_count": len(plan.steps),
            "status": plan.status.value,
            "steps": steps_info,
            "message": (
                f"Plan '{plan.title}' created with {len(plan.steps)} steps. "
                "Use update_plan_status with the exact step_id values above to mark progress."
            ),
        }


class UpdatePlanStatusParams(BaseModel):
    step_id: str = Field(..., description="The step ID to update")
    status: str = Field(..., description="New status: pending, running, done, failed, skipped")
    error_message: str = Field(default="", description="Error details if step failed")
    output_summary: str = Field(default="", description="Brief summary of step output")


class UpdatePlanStatusTool(CallableTool[UpdatePlanStatusParams]):
    """Update the status of a plan step during execution."""

    name: str = "update_plan_status"
    description: str = (
        "Update the status of a step in the analysis plan. "
        "Call this when starting a step (running), completing it (done), or when it fails (failed). "
        "Include error_message for failures and output_summary for completed steps."
    )
    params_model: type[UpdatePlanStatusParams] = UpdatePlanStatusParams

    def __init__(self, plan_manager: PlanManager | None = None) -> None:
        super().__init__()
        self._plan_manager = plan_manager

    async def execute(self, params: UpdatePlanStatusParams, **kwargs: Any) -> dict[str, Any]:
        if self._plan_manager is None:
            return {"success": False, "error": "Plan manager not initialized."}

        try:
            step_status = StepStatus(params.status)
        except ValueError:
            valid = [s.value for s in StepStatus]
            return {"success": False, "error": f"Invalid status '{params.status}'. Valid: {valid}"}

        updated = self._plan_manager.update_step_status(
            step_id=params.step_id,
            status=step_status,
            error_message=params.error_message,
            output_summary=params.output_summary,
        )
        if updated is None:
            return {"success": False, "error": f"Step '{params.step_id}' not found in plan."}

        return {
            "success": True,
            "step_id": updated.step_id,
            "status": updated.status.value,
            "message": f"Step '{updated.step_id}' → {updated.status.value}",
        }


class GetPlanParams(BaseModel):
    """No parameters needed — returns the current plan state."""


class GetPlanTool(CallableTool[GetPlanParams]):
    """Query the current analysis plan and its progress."""

    name: str = "get_plan"
    description: str = (
        "Get the current analysis plan with all steps and their statuses. "
        "Use this to check progress or remind yourself of the plan before continuing."
    )
    params_model: type[GetPlanParams] = GetPlanParams

    def __init__(self, plan_manager: PlanManager | None = None) -> None:
        super().__init__()
        self._plan_manager = plan_manager

    async def execute(self, params: GetPlanParams, **kwargs: Any) -> dict[str, Any]:
        if self._plan_manager is None:
            return {"success": False, "error": "Plan manager not initialized."}

        plan = self._plan_manager.get_plan()
        if plan is None:
            return {"success": True, "has_plan": False, "message": "No active plan."}

        steps_summary = []
        for s in plan.steps:
            steps_summary.append({
                "step_id": s.step_id,
                "order": s.order,
                "description": s.description,
                "tool_name": s.tool_name,
                "status": s.status.value,
            })

        return {
            "success": True,
            "has_plan": True,
            "plan_id": plan.plan_id,
            "title": plan.title,
            "description": plan.description,
            "domain": plan.domain,
            "status": plan.status.value,
            "steps": steps_summary,
        }
