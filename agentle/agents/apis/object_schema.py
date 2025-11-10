"""
Clean replacement for the existing endpoint parameter implementation.
This replaces the original classes with proper object parameter support.

Simply replace the existing EndpointParameter and related classes with these improved versions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

if TYPE_CHECKING:
    from agentle.agents.apis.array_schema import ArraySchema
    from agentle.agents.apis.primitive_schema import PrimitiveSchema


class ObjectSchema(BaseModel):
    """Schema definition for object parameters."""

    type: Literal["object"] = Field(default="object")

    properties: Mapping[str, ObjectSchema | ArraySchema | PrimitiveSchema] = Field(
        default_factory=dict, description="Properties of the object with their schemas"
    )

    required: Sequence[str] = Field(
        default_factory=list, description="List of required property names"
    )

    additional_properties: bool = Field(
        default=True,
        description="Whether additional properties beyond those defined are allowed",
    )

    example: Mapping[str, Any] | None = Field(
        default=None, description="Example value for the object"
    )

    @classmethod
    def from_json_schema(
        cls, schema: Mapping[str, Any]
    ) -> ObjectSchema | ArraySchema | PrimitiveSchema:
        """
        Recursively convert a JSON Schema definition to Agentle schema types.

        This method handles deeply nested objects, arrays, and primitives,
        making it easy to convert complex JSON Schema definitions.

        Args:
            schema: JSON Schema definition (dict with 'type', 'properties', etc.)

        Returns:
            Appropriate schema type (ObjectSchema, ArraySchema, or PrimitiveSchema)

        Example:
            ```python
            from agentle.agents.apis.object_schema import ObjectSchema

            json_schema = {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "settings": {
                                "type": "object",
                                "properties": {
                                    "theme": {"type": "string"},
                                    "notifications": {"type": "boolean"}
                                }
                            }
                        }
                    }
                }
            }

            schema = ObjectSchema.from_json_schema(json_schema)
            ```
        """
        from agentle.agents.apis.array_schema import ArraySchema
        from agentle.agents.apis.primitive_schema import PrimitiveSchema

        schema_type = schema.get("type", "string")

        if schema_type == "object":
            properties: dict[str, ObjectSchema | ArraySchema | PrimitiveSchema] = {}
            for prop_name, prop_schema in schema.get("properties", {}).items():
                properties[prop_name] = cls.from_json_schema(prop_schema)

            return cls(
                properties=properties,
                required=list(schema.get("required", [])),
                additional_properties=schema.get("additionalProperties", True),
                example=schema.get("example"),
            )

        elif schema_type == "array":
            items_schema = schema.get("items", {"type": "string"})
            return ArraySchema(
                items=cls.from_json_schema(items_schema),
                min_items=schema.get("minItems"),
                max_items=schema.get("maxItems"),
                example=schema.get("example"),
            )

        else:
            # Primitive type
            return PrimitiveSchema(
                type=cast(
                    Literal["string", "integer", "boolean", "number"], schema_type
                )
                if schema_type in ["string", "integer", "number", "boolean"]
                else "string",
                format=schema.get("format"),
                enum=schema.get("enum"),
                minimum=schema.get("minimum"),
                maximum=schema.get("maximum"),
                pattern=schema.get("pattern"),
                example=schema.get("example"),
            )
