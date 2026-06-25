"""Context window management for PassiAgent.

Manages conversation context with token-aware message windowing, compaction hints,
and checkpoint integration. Follows the same pattern as Kimi CLI's context management.
"""

from __future__ import annotations

from typing import Any

from passi.config import PassiConfig


class ContextManager:
    """Manages the conversation context window for the agent.

    Tracks messages, system prompt, tool definitions, and supports
    context compaction to prevent overflow.
    """

    # Approximate token count thresholds
    DEFAULT_MAX_MESSAGES = 100
    DEFAULT_WARNING_TOKENS = 80_000
    DEFAULT_CRITICAL_TOKENS = 160_000
    CHARS_PER_TOKEN_ESTIMATE = 3

    def __init__(self, config: PassiConfig) -> None:
        self.config = config
        self._messages: list[dict[str, Any]] = []
        self._system_prompt: str = ""
        self._tools: list[dict[str, Any]] = []
        self._token_count: int = 0
        self._compaction_index: int = 0  # messages before this are compacted

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def set_tools(self, tools: list[dict[str, Any]]) -> None:
        self._tools = tools

    def add_message(self, role: str, content: str | list[dict[str, Any]]) -> None:
        """Append a message to the context."""
        msg: dict[str, Any] = {"role": role}
        if isinstance(content, str):
            msg["content"] = content
        else:
            msg["content"] = content
        self._messages.append(msg)
        self._update_token_estimate()

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the current window."""
        return self._messages[self._compaction_index :]

    def get_full_context(self) -> dict[str, Any]:
        """Build the full context for an LLM API call."""
        return {
            "system": self._system_prompt,
            "messages": self.get_messages(),
            "tools": self._tools,
        }

    def needs_compaction(self, threshold: int | None = None) -> bool:
        """Check if the context needs compaction."""
        threshold = threshold or self.DEFAULT_WARNING_TOKENS
        return self._token_count > threshold

    def compact(self, summary_prompt: str = "") -> None:
        """Compact the context by summarizing older messages.

        Keeps the last N messages and replaces older ones with a summary.
        """
        if len(self._messages) < 4:
            return
        # Keep the most recent messages (last 20 or 30%)
        keep = max(min(len(self._messages) // 3, 20), 5)
        old_messages = self._messages[: -keep]
        self._messages = self._messages[-keep:]

        # Insert summary of compacted messages
        summary = f"[上下文压缩] 之前的 {len(old_messages)} 条消息已被压缩。"
        if summary_prompt:
            summary += f" 摘要: {summary_prompt}"
        self._messages.insert(0, {"role": "system", "content": summary})
        self._compaction_index = 1  # skip the summary
        self._update_token_estimate()

    def clear(self) -> None:
        """Clear all context (system prompt and tools preserved)."""
        self._messages.clear()
        self._compaction_index = 0
        self._token_count = 0

    def reset(self) -> None:
        """Full reset including system prompt and tools."""
        self._messages.clear()
        self._system_prompt = ""
        self._tools.clear()
        self._compaction_index = 0
        self._token_count = 0

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def estimated_tokens(self) -> int:
        return self._token_count

    def _update_token_estimate(self) -> None:
        """Rough token count estimate based on character length."""
        total_chars = len(self._system_prompt)
        for msg in self._messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total_chars += len(block["text"])
        for tool in self._tools:
            total_chars += len(str(tool))
        self._token_count = total_chars // self.CHARS_PER_TOKEN_ESTIMATE
