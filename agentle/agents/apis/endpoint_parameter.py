import json
import urllib
from collections.abc import Sequence, Sized
from typing import Any, cast
import urllib.parse

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.array_schema import ArraySchema
from agentle.agents.apis.object_schema import ObjectSchema
from agentle.agents.apis.object_serialization_style import ObjectSerializationStyle
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.parameter_schema import ParameterSchema
from agentle.agents.apis.primitive_schema import PrimitiveSchema


class EndpointParameter(BaseModel):
    """
    Represents a parameter for an API endpoint with proper object support.

    Fixed version that resolves all type errors and naming conflicts.
    """

    name: str = Field(description="Name of the parameter")

    description: str = Field(description="Human-readable description")

    # For backward compatibility, support both old param_type and new parameter_schema
    param_type: str | None = Field(
        default=None, description="Simple type name (for backward compatibility)"
    )

    # Renamed from 'schema' to avoid conflict with BaseModel.schema()
    parameter_schema: ParameterSchema | None = Field(
        default=None, description="Detailed schema definition for this parameter"
    )

    location: ParameterLocation = Field(
        default=ParameterLocation.QUERY,
        description="Where to place this parameter in the HTTP request",
    )

    required: bool = Field(default=False)

    default: Any = Field(default=None)

    enum: Sequence[str] | None = Field(
        default=None, description="Allowed values (for backward compatibility)"
    )

    serialization_style: ObjectSerializationStyle | None = Field(
        default=None,
        description="How to serialize objects/arrays (auto-detected if None)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize schema from param_type if schema not provided (backward compatibility)."""
        super().model_post_init(__context)

        if self.parameter_schema is None and self.param_type:
            # Convert old param_type to new schema for backward compatibility
            if self.param_type == "object":
                self.parameter_schema = ObjectSchema()
            elif self.param_type == "array":
                self.parameter_schema = ArraySchema(
                    items=PrimitiveSchema(type="string")
                )
            elif self.param_type in ["string", "integer", "number", "boolean"]:
                self.parameter_schema = PrimitiveSchema(
                    type=self.param_type,  # type: ignore
                    enum=self.enum,
                )
            else:
                # Default to string for unknown types
                self.parameter_schema = PrimitiveSchema(type="string")

    def get_serialization_style(self) -> ObjectSerializationStyle:
        """Get the appropriate serialization style based on location and schema."""
        if self.serialization_style:
            return self.serialization_style

        # Auto-detect based on location and type
        if self.location == ParameterLocation.BODY:
            return ObjectSerializationStyle.JSON_STRING
        elif self.location == ParameterLocation.QUERY:
            if isinstance(self.parameter_schema, ObjectSchema):
                return ObjectSerializationStyle.FORM_STYLE
            elif isinstance(self.parameter_schema, ArraySchema):
                return ObjectSerializationStyle.BRACKET_NOTATION
        elif self.location == ParameterLocation.HEADER:
            return ObjectSerializationStyle.JSON_STRING

        return ObjectSerializationStyle.JSON_STRING

    def serialize_value(self, value: Any) -> str | dict[str, Any] | Any:
        """
        Serialize a value according to the parameter's schema and location.

        Args:
            value: The value to serialize

        Returns:
            Serialized value ready for HTTP request
        """
        if self.parameter_schema is None:
            return str(value)

        # Validate the value against schema
        self._validate_value(value)

        # For body parameters, return the object as-is (will be JSON serialized)
        if self.location == ParameterLocation.BODY:
            return value

        # For other locations, serialize to string
        style = self.get_serialization_style()

        if isinstance(self.parameter_schema, ObjectSchema):
            return self._serialize_object(value, style)
        elif isinstance(self.parameter_schema, ArraySchema):
            return self._serialize_array(value, style)
        else:
            # Primitive type
            return str(value)

    def _validate_value(self, value: Any) -> None:
        """Validate value against the parameter schema."""
        if self.parameter_schema is None:
            return

        if isinstance(self.parameter_schema, ObjectSchema):
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected object for parameter {self.name}, got {type(value)}"
                )

            # Check required fields
            missing_required = [
                field for field in self.parameter_schema.required if field not in value
            ]
            if missing_required:
                raise ValueError(
                    f"Missing required fields for {self.name}: {missing_required}"
                )

        elif isinstance(self.parameter_schema, ArraySchema):
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected array for parameter {self.name}, got {type(value)}"
                )

            # Validate array length
            if (
                self.parameter_schema.min_items is not None
                and len(cast(Sized, value)) < self.parameter_schema.min_items
            ):
                raise ValueError(
                    f"Array {self.name} must have at least {self.parameter_schema.min_items} items"
                )
            if (
                self.parameter_schema.max_items is not None
                and len(cast(Sized, value)) > self.parameter_schema.max_items
            ):
                raise ValueError(
                    f"Array {self.name} must have at most {self.parameter_schema.max_items} items"
                )

    def _serialize_object(
        self, obj: dict[str, Any], style: ObjectSerializationStyle
    ) -> str:
        """Serialize object based on style."""
        if style == ObjectSerializationStyle.JSON_STRING:
            return json.dumps(obj, separators=(",", ":"))

        elif style == ObjectSerializationStyle.FORM_STYLE:
            # Flatten object to key=value pairs
            return "&".join(
                [
                    f"{urllib.parse.quote(str(k))}={urllib.parse.quote(str(v))}"
                    for k, v in obj.items()
                ]
            )

        elif style == ObjectSerializationStyle.DOT_NOTATION:
            # Flatten nested objects with dot notation
            pairs: list[str] = []
            self._flatten_object_with_dots(obj, "", pairs)
            return "&".join(pairs)

        else:
            return json.dumps(obj, separators=(",", ":"))

    def _serialize_array(self, arr: list[Any], style: ObjectSerializationStyle) -> str:
        """Serialize array based on style."""
        if style == ObjectSerializationStyle.JSON_STRING:
            return json.dumps(arr, separators=(",", ":"))

        elif style == ObjectSerializationStyle.BRACKET_NOTATION:
            # items[0]=value1&items[1]=value2
            pairs = [
                f"{urllib.parse.quote(self.name)}[{i}]={urllib.parse.quote(str(val))}"
                for i, val in enumerate(arr)
            ]
            return "&".join(pairs)

        else:
            # Comma-separated values
            return ",".join(str(item) for item in arr)

    def _flatten_object_with_dots(
        self, obj: dict[str, Any], prefix: str, pairs: list[str]
    ) -> None:
        """Recursively flatten object with dot notation."""
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_object_with_dots(
                    cast(dict[str, Any], value), full_key, pairs
                )
            else:
                pairs.append(
                    f"{urllib.parse.quote(full_key)}={urllib.parse.quote(str(value))}"
                )

    def to_tool_parameter_schema(self) -> dict[str, Any]:
        """
        Convert this parameter to tool parameter format with proper object schema.

        Returns:
            Tool parameter schema dictionary
        """
        base_schema: dict[str, Any] = {
            "description": self.description,
            "required": self.required,
        }

        if self.default is not None:
            base_schema["default"] = self.default

        if self.parameter_schema is None:
            # Fallback for old param_type style
            base_schema["type"] = self.param_type or "string"
            if self.enum:
                base_schema["enum"] = list(self.enum)
            return base_schema

        # Add schema-specific information
        if isinstance(self.parameter_schema, ObjectSchema):
            properties_dict: dict[str, Any] = {}
            for name, prop_schema in self.parameter_schema.properties.items():
                properties_dict[name] = self._schema_to_tool_property(prop_schema)

            base_schema.update(
                {
                    "type": "object",
                    "properties": properties_dict,
                    "required_properties": list(self.parameter_schema.required),
                }
            )

            if self.parameter_schema.example:
                base_schema["example"] = self.parameter_schema.example

        elif isinstance(self.parameter_schema, ArraySchema):
            base_schema.update(
                {
                    "type": "array",
                    "items": self._schema_to_tool_property(self.parameter_schema.items),
                }
            )

            if self.parameter_schema.example:
                base_schema["example"] = list(self.parameter_schema.example)

        else:
            # Primitive schema
            base_schema["type"] = self.parameter_schema.type

            if hasattr(self.parameter_schema, "enum") and self.parameter_schema.enum:
                base_schema["enum"] = list(self.parameter_schema.enum)
            if (
                hasattr(self.parameter_schema, "example")
                and self.parameter_schema.example
            ):
                base_schema["example"] = self.parameter_schema.example

        return base_schema

    def _schema_to_tool_property(self, schema: ParameterSchema) -> dict[str, Any]:
        """Convert a parameter schema to tool property format."""
        if isinstance(schema, ObjectSchema):
            properties_dict: dict[str, Any] = {}
            for name, prop_schema in schema.properties.items():
                properties_dict[name] = self._schema_to_tool_property(prop_schema)

            return {"type": "object", "properties": properties_dict}
        elif isinstance(schema, ArraySchema):
            return {
                "type": "array",
                "items": self._schema_to_tool_property(schema.items),
            }
        else:
            return {"type": schema.type}
