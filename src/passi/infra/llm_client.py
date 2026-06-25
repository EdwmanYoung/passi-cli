"""Multi-provider LLM client abstraction.

Supports Anthropic Claude, OpenAI, and Ollama local models through a unified
interface. Provider-specific implementations handle the API differences.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from passi.config import PassiConfig, LLMProviderConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client interface."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Returns a dict with keys: content, tool_calls (optional), usage, finish_reason.
        """
        ...

    @abstractmethod
    def supports_tool_use(self) -> bool:
        """Whether this provider supports native tool/function calling."""
        ...


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, config: LLMProviderConfig) -> None:
        super().__init__(config)
        import anthropic

        kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = config.model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "messages": self._convert_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        # Extended thinking (Anthropic only)
        if hasattr(self.config, "thinking_budget_tokens") and self.config.thinking_budget_tokens > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.config.thinking_budget_tokens,
            }
            kwargs["max_tokens"] = max(kwargs["max_tokens"], self.config.thinking_budget_tokens + 1024)

        response = await self._client.messages.create(**kwargs)

        content_blocks: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return {
            "content": content_blocks,
            "tool_calls": tool_calls if tool_calls else None,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "finish_reason": response.stop_reason,
            "model": response.model,
        }

    def supports_tool_use(self) -> bool:
        return True

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # System messages are handled separately in Anthropic API
                continue

            if role == "tool":
                # Anthropic: tool results go as user messages with tool_result blocks
                # Find the last assistant's tool_use id for correct pairing
                tool_use_id = _find_last_tool_use_id(converted)
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content if isinstance(content, str) else str(content),
                    }],
                })
            elif role == "tool_results":
                # Batched tool results: expand into one user message with multiple tool_result blocks
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr["tool_use_id"],
                        "content": tr["content"] if isinstance(tr["content"], str) else str(tr["content"]),
                    }
                    for tr in content
                ]
                converted.append({"role": "user", "content": blocks})
            elif isinstance(content, str):
                converted.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Content blocks already in Anthropic format
                converted.append({"role": role, "content": content})
        return converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool definitions to Anthropic format."""
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            elif "name" in tool:
                anthropic_tools.append(tool)
        return anthropic_tools


class OpenAIClient(LLMClient):
    """OpenAI API client (also works with Ollama compatible endpoints)."""

    def __init__(self, config: LLMProviderConfig) -> None:
        super().__init__(config)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
        )
        self._model = config.model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = tools
            # Enable tool calling behavior
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        content_blocks: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        if choice.message.content:
            content_blocks.append({"type": "text", "text": choice.message.content})

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        return {
            "content": content_blocks,
            "tool_calls": tool_calls if tool_calls else None,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            "finish_reason": choice.finish_reason,
            "model": response.model,
        }

    def supports_tool_use(self) -> bool:
        return True


class OllamaClient(OpenAIClient):
    """Ollama local model client (OpenAI-compatible API)."""

    def __init__(self, config: LLMProviderConfig) -> None:
        # Ollama uses an OpenAI-compatible endpoint
        from passi.config import OllamaConfig as OC

        if not isinstance(config, OC):
            config = OC(
                api_key="ollama",
                base_url=config.base_url or "http://localhost:11434/v1",
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                enabled=True,
            )
        super().__init__(config)

    def supports_tool_use(self) -> bool:
        return True


def _find_last_tool_use_id(messages: list[dict[str, Any]]) -> str:
    """Find the tool_use id from the last assistant message's content blocks."""
    for msg in reversed(messages):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
            for block in reversed(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return block.get("id", "unknown")
    return "unknown"


def create_llm_client(config: PassiConfig, provider: str | None = None) -> LLMClient:
    """Factory function to create an LLM client for the given provider."""
    provider = provider or config.default_provider
    provider_config = config.get_llm_config(provider)

    factories = {
        "anthropic": lambda: AnthropicClient(provider_config),
        "openai": lambda: OpenAIClient(provider_config),
        "ollama": lambda: OllamaClient(provider_config),
    }

    if provider not in factories:
        msg = f"Unknown LLM provider: {provider}. Available: {list(factories.keys())}"
        raise ValueError(msg)

    if not provider_config.enabled:
        msg = f"LLM provider '{provider}' is disabled in config."
        raise ValueError(msg)

    return factories[provider]()
