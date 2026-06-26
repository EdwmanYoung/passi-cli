"""Context window management for PassiAgent.

Manages conversation context with token-aware message windowing, LLM-based
compaction, and checkpoint integration.
"""

from __future__ import annotations

import logging
from typing import Any

from passi.config import PassiConfig

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages the conversation context window for the agent.

    Tracks messages, system prompt, tool definitions, and supports
    LLM-based context compaction to prevent overflow. Uses API-reported
    token counts for accuracy, falls back to character estimation.
    """

    DEFAULT_MAX_MESSAGES = 100
    DEFAULT_WARNING_TOKENS = 200_000
    DEFAULT_CRITICAL_TOKENS = 200_000
    CHARS_PER_TOKEN_ESTIMATE = 3

    def __init__(self, config: PassiConfig) -> None:
        self.config = config
        self._messages: list[dict[str, Any]] = []
        self._system_prompt: str = ""
        self._tools: list[dict[str, Any]] = []
        self._token_count: int = 0
        self._compaction_index: int = 0
        self._last_api_tokens: int = 0  # token count from last LLM API response
        self._llm_client: Any = None  # set via set_llm_client() for LLM-based compaction

    def set_llm_client(self, client: Any) -> None:
        """Set the LLM client for compaction summarization."""
        self._llm_client = client

    def update_api_tokens(self, input_tokens: int) -> None:
        """Update token estimate from LLM API response (more accurate than char estimate)."""
        self._last_api_tokens = input_tokens

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
        return self._messages[self._compaction_index:]

    def get_full_context(self) -> dict[str, Any]:
        """Build the full context for an LLM API call."""
        return {
            "system": self._system_prompt,
            "messages": self.get_messages(),
            "tools": self._tools,
        }

    def needs_compaction(self, threshold: int | None = None) -> bool:
        """Check if the context needs compaction.

        Uses the maximum of character-based estimate and last API-reported
        token count for accuracy.
        """
        threshold = threshold or self.DEFAULT_WARNING_TOKENS
        effective_tokens = max(self._token_count, self._last_api_tokens)
        return effective_tokens > threshold

    async def compact(self) -> str | None:
        """Compact the context by summarizing older messages.

        If an LLM client is available, uses it to generate an intelligent summary
        of the conversation so far. Falls back to truncation if no LLM client
        or if the LLM call fails.

        Returns the summary text if LLM-based compaction succeeded, None if
        truncation fallback was used.
        """
        if len(self._messages) < 6:
            return None

        # Keep the most recent messages
        keep = max(min(len(self._messages) // 4, 15), 5)
        old_messages = self._messages[:-keep]
        self._messages = self._messages[-keep:]

        # Try LLM-based summarization
        if self._llm_client is not None:
            try:
                summary = await self._llm_summarize(old_messages)
                if summary:
                    summary_msg = (
                        f"[Context Compaction] The conversation above has been summarized. "
                        f"Previous {len(old_messages)} messages distilled:\n\n{summary}"
                    )
                    self._messages.insert(0, {"role": "user", "content": summary_msg})
                    self._compaction_index = 1
                    self._update_token_estimate()
                    logger.info("LLM compaction complete: %d messages → summary", len(old_messages))
                    return summary
            except Exception as e:
                logger.warning("LLM compaction failed, falling back to truncation: %s", e)

        # Fallback: truncation with structured summary
        summary = (
            f"[Context Compaction] Previous {len(old_messages)} messages have been "
            f"compacted to preserve context. Key analysis context may have been "
            f"lost — if you need details from earlier in the conversation, "
            f"ask the user to repeat them."
        )
        self._messages.insert(0, {"role": "user", "content": summary})
        self._compaction_index = 1
        self._update_token_estimate()
        logger.info("Truncation compaction: %d messages removed", len(old_messages))
        return None

    async def _llm_summarize(self, messages: list[dict[str, Any]]) -> str | None:
        """Use the LLM to summarize old messages before compaction."""
        if self._llm_client is None:
            return None

        # Build the conversation text to summarize
        conversation_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        text_parts.append(f"[tool: {block.get('name', '')}]")
                content = " ".join(text_parts)
            conversation_parts.append(f"{role}: {content}")

        conversation = "\n".join(conversation_parts)

        # Load the compaction prompt template
        from pathlib import Path
        template_path = Path(__file__).resolve().parent.parent / "prompts" / "compact_summary.txt"
        if template_path.exists():
            summary_prompt = template_path.read_text(encoding="utf-8")
        else:
            summary_prompt = "Summarize this conversation concisely, preserving analysis goals, data characteristics, key decisions, intermediate results, current state, and file paths."

        try:
            response = await self._llm_client.chat(
                messages=[
                    {"role": "user", "content": f"{summary_prompt}\n\n{conversation}"}
                ],
                tools=None,
                system="You are a conversation summarizer. Output a structured summary.",
                max_tokens=2048,
            )
            text_parts = [
                b.get("text", "") for b in response.get("content", [])
                if b.get("type") == "text"
            ]
            return " ".join(text_parts) if text_parts else None
        except Exception:
            return None

    def clear(self) -> None:
        """Clear all context (system prompt and tools preserved)."""
        self._messages.clear()
        self._compaction_index = 0
        self._token_count = 0
        self._last_api_tokens = 0

    def reset(self) -> None:
        """Full reset including system prompt and tools."""
        self._messages.clear()
        self._system_prompt = ""
        self._tools.clear()
        self._compaction_index = 0
        self._token_count = 0
        self._last_api_tokens = 0

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def estimated_tokens(self) -> int:
        return max(self._token_count, self._last_api_tokens)

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
