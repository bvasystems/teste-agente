from collections.abc import Sequence
from typing import Any, Literal
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class PrimitiveSchema(BaseModel):
    """Schema definition for primitive parameters."""

    type: Literal["string", "integer", "number", "boolean"] = Field(
        description="Primitive type"
    )

    format: str | None = Field(
        default=None, description="Format hint (e.g., 'date', 'email', 'uri')"
    )

    enum: Sequence[Any] | None = Field(default=None)
    minimum: float | None = Field(default=None)
    maximum: float | None = Field(default=None)
    pattern: str | None = Field(default=None)

    example: Any = Field(default=None)
