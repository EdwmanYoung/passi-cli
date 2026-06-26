"""Session management for PassiAgent.

Each analysis session is a directory containing:
- wire.jsonl: communication log
- session.yaml: metadata
- checkpoint_*.json: state checkpoints
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from passi.config import PassiConfig

logger = logging.getLogger(__name__)


class SessionMeta(BaseModel):
    """Metadata for an analysis session."""

    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    domain: str = "multi-omics"
    description: str = ""
    message_count: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Result tracking
    result_id: str = ""  # unique ID for this analysis run's result directory
    # Plan Q&A
    qa_transcript: list[dict[str, Any]] = Field(default_factory=list)  # Q&A pairs from plan QA phase


class SessionManager:
    """Manages analysis sessions — create, load, list, delete."""

    def __init__(self, config: PassiConfig) -> None:
        self.config = config.session
        self._active_session: SessionMeta | None = None
        self._sessions_dir = Path(self.config.sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self, session_id: str | None = None, domain: str = "multi-omics", description: str = ""
    ) -> SessionMeta:
        """Create a new analysis session."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_id = session_id or f"session_{ts}"
        result_id = f"result_{ts}"
        session_dir = self._sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = SessionMeta(
            session_id=session_id,
            domain=domain,
            description=description,
            result_id=result_id,
        )
        self._write_meta(session_dir, meta)
        self._active_session = meta
        logger.info("Session created: %s (domain: %s)", session_id, domain)
        return meta

    def load_session(self, session_id: str) -> SessionMeta:
        """Load an existing session by ID."""
        session_dir = self._sessions_dir / session_id
        if not session_dir.exists():
            msg = f"Session not found: {session_id}"
            raise FileNotFoundError(msg)
        meta = self._read_meta(session_dir)
        self._active_session = meta
        logger.info("Session loaded: %s (%d messages)", session_id, meta.message_count)
        return meta

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all available sessions with summary."""
        sessions: list[dict[str, Any]] = []
        for session_dir in sorted(self._sessions_dir.iterdir(), reverse=True):
            if session_dir.is_dir():
                try:
                    meta = self._read_meta(session_dir)
                    sessions.append({
                        "session_id": meta.session_id,
                        "domain": meta.domain,
                        "created_at": meta.created_at,
                        "message_count": meta.message_count,
                        "description": meta.description,
                    })
                except Exception:
                    continue
        return sessions

    def delete_session(self, session_id: str) -> None:
        """Delete a session directory and all its data."""
        session_dir = self._sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info("Session deleted: %s", session_id)
        if self._active_session and self._active_session.session_id == session_id:
            self._active_session = None

    def get_session_dir(self, session_id: str | None = None) -> Path:
        """Get the directory for a session."""
        sid = session_id or (self._active_session.session_id if self._active_session else None)
        if sid is None:
            msg = "No active session."
            raise RuntimeError(msg)
        return self._sessions_dir / sid

    def touch(self) -> None:
        """Update the session's updated_at timestamp."""
        if self._active_session is None:
            return
        self._active_session.updated_at = datetime.now(timezone.utc).isoformat()
        self._active_session.message_count += 1
        session_dir = self._sessions_dir / self._active_session.session_id
        self._write_meta(session_dir, self._active_session)

    @property
    def active_session(self) -> SessionMeta | None:
        return self._active_session

    def _write_meta(self, session_dir: Path, meta: SessionMeta) -> None:
        meta_path = session_dir / "session.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(meta.model_dump(), f, allow_unicode=True)

    def _read_meta(self, session_dir: Path) -> SessionMeta:
        meta_path = session_dir / "session.yaml"
        if meta_path.exists():
            import yaml
            with open(meta_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return SessionMeta(**data)
        return SessionMeta(session_id=session_dir.name)

    def checkpoint(self, state: dict[str, Any]) -> Path:
        """Save a state checkpoint to the active session."""
        session_dir = self.get_session_dir()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        checkpoint_path = session_dir / f"checkpoint_{ts}.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str, indent=2, ensure_ascii=False)
        return checkpoint_path
