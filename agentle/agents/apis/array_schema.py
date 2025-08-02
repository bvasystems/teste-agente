from collections.abc import Sequence
from typing import Any, Literal
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.parameter_schema import ParameterSchema


class ArraySchema(BaseModel):
    """Schema definition for array parameters."""

    type: Literal["array"] = Field(default="array")

    items: ParameterSchema = Field(description="Schema for array items")

    min_items: int | None = Field(default=None)
    max_items: int | None = Field(default=None)

    example: Sequence[Any] | None = Field(default=None)
