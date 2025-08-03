from textwrap import dedent
from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)


class ToolExecutionResult(BaseModel):
    """
    Represents the result of a tool execution within a step.

    This class captures both the suggestion that was executed and the result
    that was returned, providing a complete record of the tool interaction.
    """

    type: Literal["tool_execution_result"] = Field(
        default="tool_execution_result",
        description="Discriminator field to identify this as a tool execution suggestion.",
    )

    suggestion: ToolExecutionSuggestion
    """The tool execution suggestion that was executed."""

    result: Any
    """The result returned by the tool execution."""

    execution_time_ms: float | None = Field(default=None)
    """Time taken to execute the tool in milliseconds."""

    success: bool = Field(default=True)
    """Whether the tool execution was successful."""

    error_message: str | None = Field(default=None)
    """Error message if the tool execution failed."""

    @property
    def text(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return dedent(f"""\
        <suggestion>
        {self.suggestion}
        </suggestion>
        <result>
        {self.result}
        </result>
        """)
