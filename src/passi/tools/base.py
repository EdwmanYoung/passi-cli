"""Base tool classes and protocols for the PassiAgent tool system.

Similar to Kimi CLI's CallableTool2 pattern — each tool has a Pydantic params model
and an async execute method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

# Generic type for tool parameters
ParamsT = TypeVar("ParamsT", bound=BaseModel)


class CallableTool(ABC, Generic[ParamsT]):
    """Abstract base class for all PassiAgent tools.

    Each tool defines:
    - name: unique identifier
    - description: for LLM tool selection
    - params_model: Pydantic model for parameter validation
    - execute(): the actual implementation
    """

    name: str
    description: str
    params_model: type[ParamsT]

    @abstractmethod
    async def execute(self, params: ParamsT, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with validated parameters.

        Returns a dict with at minimum: {'success': bool, 'result': Any}
        May include: {'error': str, 'files': list, 'figures': list}
        """
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._build_parameters_schema(),
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert tool to Anthropic tool use schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._build_parameters_schema(),
        }

    def _build_parameters_schema(self) -> dict[str, Any]:
        """Build JSON schema from the Pydantic params model."""
        schema = self.params_model.model_json_schema()
        # Remove pydantic-specific keys not needed for LLM tool schemas
        for key in ("title", "additionalProperties"):
            schema.pop(key, None)
        # Clean up property definitions
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def validate_params(self, raw_params: dict[str, Any]) -> ParamsT:
        """Validate and parse raw parameters against the params model."""
        return self.params_model(**raw_params)
