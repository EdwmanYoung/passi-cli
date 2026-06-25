"""Wire Protocol — JSON-RPC communication channel.

All agent↔UI communication flows through Wire, which is persisted to
wire.jsonl for session replay and audit. Inspired by Kimi CLI's Wire protocol.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Event type constants
class EventType:
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    AGENT_THINKING = "agent_thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    SYSTEM = "system"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CHECKPOINT = "checkpoint"


class WireEvent(BaseModel):
    """A single event on the Wire."""

    id: str = Field(default_factory=lambda: _generate_event_id())
    type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WireListener(Protocol):
    """Protocol for Wire event listeners."""

    async def on_event(self, event: WireEvent) -> None:
        ...


class Wire:
    """In-process pub/sub communication channel.

    Events are published by the agent or UI and received by all listeners.
    Events are persisted to wire.jsonl for replay.
    """

    def __init__(self, wire_path: Path | None = None) -> None:
        self._listeners: list[WireListener] = []
        self._history: list[WireEvent] = []
        self._wire_path = wire_path or Path("wire.jsonl")
        self._event_counter = 0

    def subscribe(self, listener: WireListener) -> None:
        """Register a listener to receive events."""
        self._listeners.append(listener)

    def unsubscribe(self, listener: WireListener) -> None:
        """Remove a listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def publish(self, event: WireEvent) -> None:
        """Publish an event to all listeners and persist it."""
        self._history.append(event)
        for listener in self._listeners:
            try:
                await listener.on_event(event)
            except Exception:
                logger.exception("Listener failed for event %s", event.id)

    def publish_sync(self, event: WireEvent) -> None:
        """Synchronous publish for non-async contexts."""
        self._history.append(event)
        self._persist(event)

    def emit(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> WireEvent:
        """Build and publish an event synchronously. Returns the event."""
        event = WireEvent(
            type=event_type,
            session_id=session_id,
            data=data or {},
            metadata=metadata or {},
        )
        self.publish_sync(event)
        return event

    def get_history(self) -> list[WireEvent]:
        """Get all events in order."""
        return list(self._history)

    def replay(self, until_event_id: str | None = None) -> list[WireEvent]:
        """Replay events up to (and including) the given event ID."""
        if until_event_id is None:
            return list(self._history)
        events: list[WireEvent] = []
        for event in self._history:
            events.append(event)
            if event.id == until_event_id:
                break
        return events

    def _persist(self, event: WireEvent) -> None:
        """Persist event to wire.jsonl."""
        try:
            with open(self._wire_path, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except OSError:
            logger.warning("Failed to persist wire event: %s", event.id)

    def load_history(self) -> list[WireEvent]:
        """Load event history from wire.jsonl."""
        events: list[WireEvent] = []
        if self._wire_path.exists():
            with open(self._wire_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(WireEvent(**json.loads(line)))
                        except Exception:
                            continue
        return events


def _generate_event_id() -> str:
    import uuid

    return str(uuid.uuid4())[:12]
