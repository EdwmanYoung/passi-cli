"""ask_user tool — pauses the ReAct loop to ask the user a clarifying question.

Returns a dict with __ask_user__ marker that the ReAct loop detects to break
and propagate the question back to the UI layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from passi.tools.base import CallableTool


class AskUserParams(BaseModel):
    """Parameters for asking the user a question."""

    question: str = Field(..., description="The question to ask the user")
    context: str = Field(
        default="",
        description="Background explaining why this question matters for the analysis",
    )
    options: list[str] | None = Field(
        default=None,
        description="Suggested answer options for the user to choose from",
    )


class AskUserTool(CallableTool[AskUserParams]):
    """Pause analysis to ask the user a clarifying question.

    Use this when:
    - Experimental design is unclear (control vs treatment, comparison groups)
    - Data format is ambiguous (FPKM vs raw counts, reference genome build)
    - Method choice has trade-offs the user should decide
    - Critical parameters are missing

    The agent should provide context and suggested options based on
    what it observed in the data.
    """

    name = "ask_user"
    description = (
        "Pause analysis to ask the user a clarifying question. "
        "Use this when the experimental design, data format, or analysis parameters "
        "are ambiguous and need user input. Provide context on why the question "
        "matters and suggest options based on what you observed in the data."
    )
    params_model = AskUserParams

    async def execute(self, params: AskUserParams, **kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "__ask_user__": True,
            "question": params.question,
            "context": params.context,
            "options": params.options,
        }
