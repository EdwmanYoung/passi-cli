"""TDD-style unit tests for SessionManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from passi.config import PassiConfig, SessionConfig


class TestSessionManager:
    """Unit tests for SessionManager — create, load, list, delete, checkpoint."""

    @pytest.fixture
    def session_manager(self, tmp_path: Path):
        from passi.infra.session import SessionManager

        config = PassiConfig(
            session=SessionConfig(sessions_dir=tmp_path),
        )
        return SessionManager(config)

    def test_create_session_returns_meta(self, session_manager):
        # Act
        meta = session_manager.create_session(domain="transcriptomics")

        # Assert
        assert meta.session_id.startswith("session_")
        assert meta.domain == "transcriptomics"
        assert meta.message_count == 0
        assert (session_manager._sessions_dir / meta.session_id).exists()

    def test_create_session_with_custom_id(self, session_manager):
        # Act
        meta = session_manager.create_session(session_id="my_session")

        # Assert
        assert meta.session_id == "my_session"

    def test_list_sessions_returns_all(self, session_manager):
        # Arrange
        session_manager.create_session(session_id="s1")
        session_manager.create_session(session_id="s2")

        # Act
        sessions = session_manager.list_sessions()

        # Assert
        assert len(sessions) >= 2
        ids = [s["session_id"] for s in sessions]
        assert "s1" in ids
        assert "s2" in ids

    def test_load_session_restores_meta(self, session_manager, tmp_path):
        # Arrange
        session_manager.create_session(session_id="test_load", domain="genomics")
        session_manager.touch()

        # Act — load in fresh manager
        from passi.infra.session import SessionManager
        from passi.config import PassiConfig, SessionConfig

        new_mgr = SessionManager(PassiConfig(session=SessionConfig(sessions_dir=tmp_path)))
        meta = new_mgr.load_session("test_load")

        # Assert
        assert meta.session_id == "test_load"
        assert meta.domain == "genomics"
        assert meta.message_count == 1

    def test_load_nonexistent_session_raises(self, session_manager):
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            session_manager.load_session("ghost_session")

    def test_delete_session_removes_directory(self, session_manager):
        # Arrange
        session_manager.create_session(session_id="temp_session")

        # Act
        session_manager.delete_session("temp_session")

        # Assert
        assert not (session_manager._sessions_dir / "temp_session").exists()

    def test_touch_updates_timestamp_and_count(self, session_manager):
        # Arrange
        session_manager.create_session(session_id="s")
        old_count = session_manager.active_session.message_count

        # Act
        session_manager.touch()

        # Assert
        assert session_manager.active_session.message_count == old_count + 1

    def test_checkpoint_writes_state_file(self, session_manager):
        # Arrange
        session_manager.create_session(session_id="s")
        state = {"step": 5, "data": "important"}

        # Act
        path = session_manager.checkpoint(state)

        # Assert
        assert path.exists()
        assert path.suffix == ".json"
        assert "checkpoint_" in path.name
