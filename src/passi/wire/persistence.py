"""Wire persistence — replay and audit from wire.jsonl.

Enables full session replay and analysis provenance via the wire event log.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from passi.wire.protocol import WireEvent

logger = logging.getLogger(__name__)


class WirePersistence:
    """Read and write wire events for session persistence."""

    def __init__(self, wire_path: Path) -> None:
        self._path = Path(wire_path)

    def read_all(self) -> list[WireEvent]:
        """Read all events from wire.jsonl."""
        events: list[WireEvent] = []
        if not self._path.exists():
            return events
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(WireEvent(**json.loads(line)))
                    except Exception as e:
                        logger.debug("Skipping invalid wire line: %s", e)
        return events

    def read_session(self, session_id: str) -> list[WireEvent]:
        """Read events for a specific session."""
        return [e for e in self.read_all() if e.session_id == session_id]

    def export_chatlog(self, session_id: str | None = None) -> str:
        """Export events as a readable chat log (markdown format)."""
        events = self.read_session(session_id) if session_id else self.read_all()
        lines = ["# Session Chat Log", ""]
        for e in events:
            if e.type == "user_message":
                lines.append(f"**User:** {_extract_text(e.data)}")
            elif e.type == "agent_message":
                lines.append(f"**Agent:** {_extract_text(e.data)}")
            elif e.type == "tool_call":
                lines.append(f"**Tool [{e.data.get('name', '?')}]:** `{json.dumps(e.data.get('params', {}), ensure_ascii=False)}`")
            elif e.type == "tool_result":
                result = e.data.get("result", "")
                if isinstance(result, str) and len(result) > 500:
                    result = result[:500] + "..."
                lines.append(f"  → {result}")
            elif e.type == "error":
                lines.append(f"**Error:** {e.data.get('message', 'Unknown error')}")
            elif e.type == "session_start":
                lines.append(f"--- Session started: {e.session_id} ---")
            elif e.type == "session_end":
                lines.append(f"--- Session ended: {e.session_id} ---")
            lines.append("")
        return "\n".join(lines)

    def get_statistics(self, session_id: str | None = None) -> dict[str, Any]:
        """Compute statistics from wire events."""
        events = self.read_session(session_id) if session_id else self.read_all()
        tool_counts: dict[str, int] = {}
        total_user_msgs = 0
        total_agent_msgs = 0
        errors = 0

        for e in events:
            if e.type == "user_message":
                total_user_msgs += 1
            elif e.type == "agent_message":
                total_agent_msgs += 1
            elif e.type == "tool_call":
                name = e.data.get("name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1
            elif e.type == "error":
                errors += 1

        return {
            "total_events": len(events),
            "user_messages": total_user_msgs,
            "agent_messages": total_agent_msgs,
            "tool_calls": sum(tool_counts.values()),
            "tool_breakdown": tool_counts,
            "errors": errors,
        }


def _extract_text(data: dict[str, Any]) -> str:
    """Extract text content from wire data."""
    content = data.get("content", "")
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, dict)]
        return " ".join(texts)
    return str(content)
