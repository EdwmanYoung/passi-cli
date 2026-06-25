"""Mock LLM clients for agent testing without real API calls."""

from __future__ import annotations

from typing import Any

from passi.infra.llm_client import LLMClient


class FakeLLMClient(LLMClient):
    """A fake LLM client that returns pre-programmed text responses.

    Use for testing agent behavior without real LLM API calls.
    """

    def __init__(self, response_text: str = "Test response.") -> None:
        from passi.config import AnthropicConfig
        super().__init__(AnthropicConfig(api_key="fake", model="fake"))
        self._response_text = response_text
        self._tool_calls: list[dict[str, Any]] | None = None
        self._chat_history: list[dict[str, Any]] = []

    def set_response(self, text: str) -> None:
        self._response_text = text

    def set_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        """Set tool calls the fake LLM should return."""
        self._tool_calls = calls

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self._chat_history.append({
            "messages": messages,
            "tools": tools,
            "system": system,
        })
        content = [{"type": "text", "text": self._response_text}]
        tool_calls = None
        if self._tool_calls:
            for tc in self._tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", "fake-id"),
                    "name": tc["name"],
                    "input": tc.get("input", {}),
                })
            tool_calls = self._tool_calls
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "finish_reason": "end_turn",
            "model": "fake-model",
        }

    def supports_tool_use(self) -> bool:
        return True

    @property
    def chat_history(self) -> list[dict[str, Any]]:
        return self._chat_history


class FakeLLMClientWithToolSequence(LLMClient):
    """Fake LLM that returns a sequence of tool calls, then a final text response.

    Useful for testing ReAct loops with multiple tool invocations.
    """

    def __init__(self) -> None:
        from passi.config import AnthropicConfig
        super().__init__(AnthropicConfig(api_key="fake", model="fake"))
        self._tool_sequences: list[list[dict[str, Any]]] = []
        self._final_response: str = "Analysis complete."
        self._call_index: int = 0

    def set_sequence(
        self,
        tool_sequences: list[list[dict[str, Any]]],
        final_response: str = "Analysis complete.",
    ) -> None:
        """Set the sequence of tool calls to return on successive LLM calls."""
        self._tool_sequences = tool_sequences
        self._final_response = final_response
        self._call_index = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if self._call_index < len(self._tool_sequences):
            tool_calls = self._tool_sequences[self._call_index]
            self._call_index += 1
            content = [{"type": "text", "text": "Let me run a tool."}]
            for tc in tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", "fake-id"),
                    "name": tc["name"],
                    "input": tc.get("input", {}),
                })
            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "finish_reason": "tool_use",
                "model": "fake-model",
            }
        else:
            return {
                "content": [{"type": "text", "text": self._final_response}],
                "tool_calls": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "finish_reason": "end_turn",
                "model": "fake-model",
            }

    def supports_tool_use(self) -> bool:
        return True
