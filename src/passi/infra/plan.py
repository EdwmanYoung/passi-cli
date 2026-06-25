"""Analysis plan management for structured bioinformatics workflows.

LLM creates a plan before executing complex analyses. Users review and
approve each step. Plans are persisted to session directories for audit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """A single step within an analysis plan."""

    step_id: str
    order: int
    description: str
    tool_name: str = ""
    expected_params: dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    started_at: str = ""
    completed_at: str = ""
    error_message: str = ""
    output_summary: str = ""


class AnalysisPlan(BaseModel):
    """A structured bioinformatics analysis plan."""

    plan_id: str
    title: str
    description: str = ""
    domain: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id: str = ""


class PlanManager:
    """Manage analysis plan lifecycle and persistence.

    Plans are persisted to ``./sessions/{session_id}/plan.yaml``.
    """

    def __init__(self, session_dir: str | Path = "") -> None:
        self._session_dir = Path(session_dir) if session_dir else Path.cwd()
        self._plan: AnalysisPlan | None = None

    @property
    def plan(self) -> AnalysisPlan | None:
        return self._plan

    def create_plan(
        self,
        title: str,
        description: str = "",
        domain: str = "",
        steps: list[dict[str, Any]] | None = None,
        session_id: str = "",
    ) -> AnalysisPlan:
        """Create a new analysis plan."""
        plan_id = _generate_plan_id()
        plan_steps: list[PlanStep] = []
        if steps:
            for i, step_data in enumerate(steps, 1):
                plan_steps.append(PlanStep(
                    step_id=f"{plan_id}_step_{i:02d}",
                    order=i,
                    description=step_data.get("description", ""),
                    tool_name=step_data.get("tool_name", ""),
                    expected_params=step_data.get("expected_params", {}),
                ))

        self._plan = AnalysisPlan(
            plan_id=plan_id,
            title=title,
            description=description,
            domain=domain,
            steps=plan_steps,
            status=PlanStatus.DRAFT,
            session_id=session_id,
        )
        self._persist()
        logger.info("Plan created: %s (%d steps)", plan_id, len(plan_steps))
        return self._plan

    def update_step_status(
        self,
        step_id: str,
        status: StepStatus,
        error_message: str = "",
        output_summary: str = "",
    ) -> PlanStep | None:
        """Update a step's status and optionally record results."""
        if self._plan is None:
            logger.warning("No active plan to update step.")
            return None

        for step in self._plan.steps:
            if step.step_id == step_id:
                step.status = status
                if status == StepStatus.RUNNING and not step.started_at:
                    step.started_at = datetime.now(timezone.utc).isoformat()
                if status in (StepStatus.DONE, StepStatus.FAILED):
                    step.completed_at = datetime.now(timezone.utc).isoformat()
                if error_message:
                    step.error_message = error_message
                if output_summary:
                    step.output_summary = output_summary
                self._persist()
                return step

        logger.warning("Step not found: %s", step_id)
        return None

    def update_plan_status(self, status: PlanStatus) -> None:
        """Update the overall plan status."""
        if self._plan is None:
            return
        self._plan.status = status
        self._persist()

    def get_plan(self) -> AnalysisPlan | None:
        return self._plan

    def get_current_step(self) -> PlanStep | None:
        """Return the first pending or running step."""
        if self._plan is None:
            return None
        for step in self._plan.steps:
            if step.status in (StepStatus.PENDING, StepStatus.RUNNING):
                return step
        return None

    def load_plan(self) -> AnalysisPlan | None:
        """Load an existing plan from the session directory."""
        plan_path = self._session_dir / "plan.yaml"
        if not plan_path.exists():
            return None

        with open(plan_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            self._plan = AnalysisPlan(**data)
            return self._plan
        return None

    def _persist(self) -> None:
        """Persist the plan to YAML."""
        if self._plan is None:
            return
        self._session_dir.mkdir(parents=True, exist_ok=True)
        plan_path = self._session_dir / "plan.yaml"
        with open(plan_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self._plan.model_dump(mode="json"),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )


def _generate_plan_id() -> str:
    import uuid

    return f"plan_{uuid.uuid4().hex[:8]}"
