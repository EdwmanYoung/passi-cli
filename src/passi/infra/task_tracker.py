"""Task tracker for todo-list style execution tracking.

Every tool execution is recorded as a Task with timing, status, and
cross-references to provenance and plan steps. Persisted as JSONL for
audit and replay.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Task(BaseModel):
    """A single task record tracking one tool execution."""

    task_id: str
    step_id: str = ""  # Optional: links to PlanStep
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    status: str = TaskStatus.PENDING
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0.0
    result_summary: str = ""
    error: str = ""
    provenance_step_id: str = ""  # Cross-reference to ProvenanceRecord


class TaskTracker:
    """Tracks tool execution tasks with timing and cross-references.

    Tasks are persisted to ``.passi/sessions/{session_id}/tasks.jsonl``.
    """

    def __init__(self, session_dir: str | Path = "") -> None:
        self._session_dir = Path(session_dir) if session_dir else Path.cwd()
        self._tasks: dict[str, Task] = {}
        self._counter: int = 0

    def create_task(
        self,
        tool_name: str,
        params: dict[str, Any],
        step_id: str = "",
    ) -> Task:
        """Create a new task and mark it as running."""
        self._counter += 1
        task_id = f"task_{self._counter:04d}"
        task = Task(
            task_id=task_id,
            step_id=step_id,
            tool_name=tool_name,
            params=params,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._tasks[task_id] = task
        self._persist(task)
        return task

    def complete_task(
        self,
        task_id: str,
        success: bool,
        result_summary: str = "",
        error: str = "",
        provenance_step_id: str = "",
    ) -> Task | None:
        """Mark a task as done or failed."""
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("Task not found: %s", task_id)
            return None

        task.success = success
        task.status = TaskStatus.DONE if success else TaskStatus.FAILED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.result_summary = result_summary
        task.error = error
        if provenance_step_id:
            task.provenance_step_id = provenance_step_id

        if task.started_at:
            try:
                started = datetime.fromisoformat(task.started_at)
                completed = datetime.fromisoformat(task.completed_at)
                task.duration_ms = (completed - started).total_seconds() * 1000
            except (ValueError, TypeError):
                pass

        self._persist(task)
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_tasks(self) -> list[Task]:
        return sorted(self._tasks.values(), key=lambda t: t.task_id)

    def load_tasks(self) -> None:
        """Load existing tasks from the session directory."""
        tasks_path = self._session_dir / "tasks.jsonl"
        if not tasks_path.exists():
            return

        with open(tasks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        task = Task(**json.loads(line))
                        self._tasks[task.task_id] = task
                        # Restore counter to highest existing task number
                        num = int(task.task_id.split("_")[1])
                        self._counter = max(self._counter, num)
                    except Exception:
                        continue

    def _persist(self, task: Task) -> None:
        """Append a task to tasks.jsonl."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = self._session_dir / "tasks.jsonl"
        with open(tasks_path, "a", encoding="utf-8") as f:
            f.write(task.model_dump_json() + "\n")
