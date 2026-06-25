"""TDD-style unit tests for Task, TaskTracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.infra.task_tracker import Task, TaskStatus, TaskTracker


class TestTask:
    """Task model defaults and construction."""

    def test_default_task(self):
        task = Task(task_id="task_0001", tool_name="read_file", params={"path": "/f.csv"})
        assert task.task_id == "task_0001"
        assert task.tool_name == "read_file"
        assert task.params == {"path": "/f.csv"}
        assert task.status == TaskStatus.PENDING
        assert task.step_id == ""
        assert task.provenance_step_id == ""

    def test_task_with_step_id(self):
        task = Task(task_id="task_0002", tool_name="run_r", step_id="plan_x_step_01")
        assert task.step_id == "plan_x_step_01"


class TestTaskStatus:
    """TaskStatus constants."""

    def test_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.DONE == "done"
        assert TaskStatus.FAILED == "failed"


class TestTaskTracker:
    """TaskTracker — create, complete, query, persistence."""

    def test_create_task(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        task = tt.create_task("read_file", {"path": "/data.csv"})
        assert task.task_id == "task_0001"
        assert task.status == TaskStatus.RUNNING
        assert task.started_at != ""
        assert task.tool_name == "read_file"

    def test_create_task_increments_counter(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        t1 = tt.create_task("a", {})
        t2 = tt.create_task("b", {})
        t3 = tt.create_task("c", {})
        assert t1.task_id == "task_0001"
        assert t2.task_id == "task_0002"
        assert t3.task_id == "task_0003"

    def test_complete_task_success(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        task = tt.create_task("read_file", {"path": "/f.csv"})
        completed = tt.complete_task(
            task.task_id,
            success=True,
            result_summary="File read: 1000 lines",
            provenance_step_id="read_file_abc123",
        )
        assert completed is not None
        assert completed.status == TaskStatus.DONE
        assert completed.result_summary == "File read: 1000 lines"
        assert completed.provenance_step_id == "read_file_abc123"
        assert completed.completed_at != ""
        assert completed.duration_ms >= 0

    def test_complete_task_failure(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        task = tt.create_task("run_r", {"code": "bad"})
        completed = tt.complete_task(
            task.task_id,
            success=False,
            error="R script error: object 'x' not found",
        )
        assert completed is not None
        assert completed.status == TaskStatus.FAILED
        assert "object 'x' not found" in completed.error

    def test_complete_unknown_task(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        result = tt.complete_task("nonexistent", success=True)
        assert result is None

    def test_get_task(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        tt.create_task("a", {})
        tt.create_task("b", {})
        assert tt.get_task("task_0001").tool_name == "a"  # type: ignore[union-attr]
        assert tt.get_task("task_0002").tool_name == "b"  # type: ignore[union-attr]

    def test_get_tasks_returns_sorted(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        tt.create_task("c", {})
        tt.create_task("a", {})
        tt.create_task("b", {})
        tasks = tt.get_tasks()
        ids = [t.task_id for t in tasks]
        assert ids == ["task_0001", "task_0002", "task_0003"]

    def test_persistence(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        task = tt.create_task("read_file", {"path": "/f.csv"})
        tt.complete_task(task.task_id, success=True, result_summary="OK")

        tasks_path = tmp_path / "tasks.jsonl"
        assert tasks_path.exists()

        # Load into new tracker
        tt2 = TaskTracker(tmp_path)
        tt2.load_tasks()
        loaded = tt2.get_tasks()
        assert len(loaded) == 1
        assert loaded[0].task_id == "task_0001"
        assert loaded[0].status == TaskStatus.DONE
        assert loaded[0].result_summary == "OK"

    def test_load_tasks_empty_directory(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        tt.load_tasks()  # Should not raise
        assert tt.get_tasks() == []

    def test_create_task_with_step_id(self, tmp_path: Path):
        tt = TaskTracker(tmp_path)
        task = tt.create_task("deseq2", {"counts": "/c.csv"}, step_id="plan_x_step_03")
        assert task.step_id == "plan_x_step_03"

    def test_load_tasks_restores_counter(self, tmp_path: Path):
        tt1 = TaskTracker(tmp_path)
        tt1.create_task("tool1", {})
        tt1.create_task("tool2", {})

        tt2 = TaskTracker(tmp_path)
        tt2.load_tasks()
        # New tasks should continue from the highest seen number
        t3 = tt2.create_task("tool3", {})
        assert t3.task_id == "task_0003"
