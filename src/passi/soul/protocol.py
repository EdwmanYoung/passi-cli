"""Soul Protocol — abstract agent interface.

All UIs depend on the Soul protocol, never directly on concrete agent implementations.
Inspired by Kimi CLI's Soul (Protocol) pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Protocol

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """A single message in the agent conversation."""

    role: str = Field(..., description="'user', 'agent', 'system', or 'tool'")
    content: str | list[dict] = Field(default="", description="Message content")
    tool_calls: list[dict] | None = Field(default=None, description="Tool calls requested")
    tool_call_id: str | None = Field(default=None, description="ID for tool result messages")
    name: str | None = Field(default=None, description="Tool name for tool result messages")
    metadata: dict = Field(default_factory=dict)


class AgentStreamEvent(BaseModel):
    """An event emitted during agent streaming."""

    type: str = Field(..., description="'text', 'tool_call', 'tool_result', 'thinking', 'error', 'done'")
    content: str | dict = Field(default="", description="Event payload")
    tool_name: str | None = Field(default=None)
    metadata: dict = Field(default_factory=dict)


class Soul(ABC):
    """Abstract agent interface — the 'Soul' protocol.

    All UI implementations (CLI, web, client SDK) depend on this interface,
    never on concrete agent implementations.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent with tools and configuration."""
        ...

    @abstractmethod
    async def chat(
        self,
        user_message: str,
        attachments: list[str] | None = None,
    ) -> AgentMessage:
        """Send a message to the agent and get a complete response.

        Non-streaming variant — returns the full response at once.
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        user_message: str,
        attachments: list[str] | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Send a message to the agent and receive a stream of events.

        Streaming variant — yields text chunks, tool calls, and results as they happen.
        """
        ...

    @abstractmethod
    async def execute_tool(self, tool_name: str, params: dict) -> AgentMessage:
        """Execute a tool directly without LLM mediation."""
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Reset the conversation context."""
        ...


class SoulFactory(Protocol):
    """Protocol for creating Soul instances."""

    async def create(self) -> Soul:
        ...
