"""
Clean replacement for the existing endpoint parameter implementation.
This replaces the original classes with proper object parameter support.

Simply replace the existing EndpointParameter and related classes with these improved versions.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.parameter_schema import ParameterSchema


class ObjectSchema(BaseModel):
    """Schema definition for object parameters."""

    type: Literal["object"] = Field(default="object")

    properties: MutableMapping[str, ParameterSchema] = Field(
        default_factory=dict, description="Properties of the object with their schemas"
    )

    required: list[str] = Field(
        default_factory=list, description="List of required property names"
    )

    additional_properties: bool = Field(
        default=True,
        description="Whether additional properties beyond those defined are allowed",
    )

    example: dict[str, Any] | None = Field(
        default=None, description="Example value for the object"
    )
