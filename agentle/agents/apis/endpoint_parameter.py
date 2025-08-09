from __future__ import annotations

import json
import urllib
import urllib.parse
from collections.abc import Sequence, Sized
from typing import TYPE_CHECKING, Any, cast

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.object_serialization_style import ObjectSerializationStyle
from agentle.agents.apis.parameter_location import ParameterLocation

if TYPE_CHECKING:
    from agentle.agents.apis.array_schema import ArraySchema
    from agentle.agents.apis.object_schema import ObjectSchema
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
    parameter_schema: ObjectSchema | ArraySchema | PrimitiveSchema | None = Field(
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

    @classmethod
    def from_json_schema(
        cls,
        schema: dict[str, Any],
        *,
        name: str,
        description: str | None = None,
        location: ParameterLocation = ParameterLocation.QUERY,
        required: bool = False,
        default: Any = None,
        serialization_style: ObjectSerializationStyle | None = None,
        components: dict[str, Any] | None = None,
    ) -> EndpointParameter:
        """
        Create an EndpointParameter from a JSON schema.

        Args:
            schema: JSON schema dictionary defining the parameter structure
            name: Name of the parameter
            description: Description of the parameter (falls back to schema description)
            location: Where to place this parameter in the HTTP request
            required: Whether this parameter is required
            default: Default value for the parameter
            serialization_style: How to serialize objects/arrays (auto-detected if None)
            components: Components dictionary for resolving $ref (optional)

        Returns:
            EndpointParameter instance created from the JSON schema

        Example:
            ```python
            # Simple string parameter
            param = EndpointParameter.from_json_schema(
                schema={"type": "string", "enum": ["red", "green", "blue"]},
                name="color",
                description="Color selection",
                required=True
            )

            # Object parameter
            param = EndpointParameter.from_json_schema(
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer", "minimum": 0}
                    },
                    "required": ["name"]
                },
                name="user",
                description="User information",
                location=ParameterLocation.BODY
            )

            # Array parameter
            param = EndpointParameter.from_json_schema(
                schema={
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10
                },
                name="tags",
                description="List of tags"
            )
            ```
        """

        # Use schema description as fallback if no description provided
        param_description = description or schema.get("description", "")

        # Use schema default as fallback if no default provided
        param_default = default if default is not None else schema.get("default")

        # Convert JSON schema to internal schema representation
        parameter_schema = cls._convert_json_schema(schema, components or {})

        return cls(
            name=name,
            description=param_description,
            parameter_schema=parameter_schema,
            location=location,
            required=required,
            default=param_default,
            serialization_style=serialization_style,
        )

    @classmethod
    def _convert_json_schema(
        cls,
        schema: dict[str, Any],
        components: dict[str, Any],
    ) -> ObjectSchema | ArraySchema | PrimitiveSchema:
        """
        Convert a JSON schema to internal schema representation.

        Args:
            schema: JSON schema dictionary
            components: Components dictionary for resolving $ref

        Returns:
            Internal schema representation
        """
        from agentle.agents.apis.array_schema import ArraySchema
        from agentle.agents.apis.object_schema import ObjectSchema
        from agentle.agents.apis.primitive_schema import PrimitiveSchema

        # Handle $ref references
        if "$ref" in schema:
            ref_path = schema["$ref"].split("/")
            if len(ref_path) >= 4 and ref_path[1] == "components":
                component_type = ref_path[2]  # e.g., "schemas"
                component_name = ref_path[3]  # e.g., "User"
                ref_schema = components.get(component_type, {}).get(component_name, {})
                return cls._convert_json_schema(ref_schema, components)
            else:
                raise ValueError(f"Unsupported $ref format: {schema['$ref']}")

        schema_type = schema.get("type", "string")

        if schema_type == "object":
            # Handle object schemas
            properties: dict[str, ObjectSchema | ArraySchema | PrimitiveSchema] = {}

            for prop_name, prop_schema in schema.get("properties", {}).items():
                properties[prop_name] = cls._convert_json_schema(
                    prop_schema, components
                )

            return ObjectSchema(
                properties=properties,
                required=schema.get("required", []),
                additional_properties=schema.get("additionalProperties", True),
                example=schema.get("example"),
            )

        elif schema_type == "array":
            # Handle array schemas
            items_schema = schema.get("items", {"type": "string"})
            items = cls._convert_json_schema(items_schema, components)

            return ArraySchema(
                items=items,
                min_items=schema.get("minItems"),
                max_items=schema.get("maxItems"),
                example=schema.get("example"),
            )

        else:
            # Handle primitive schemas (string, integer, number, boolean)
            if schema_type not in ["string", "integer", "number", "boolean"]:
                # Default to string for unknown types
                schema_type = "string"

            return PrimitiveSchema(
                type=schema_type,  # type: ignore
                format=schema.get("format"),
                enum=schema.get("enum"),
                minimum=schema.get("minimum"),
                maximum=schema.get("maximum"),
                pattern=schema.get("pattern"),
                example=schema.get("example"),
            )

    def model_post_init(self, __context: Any) -> None:
        """Initialize schema from param_type if schema not provided (backward compatibility)."""
        from agentle.agents.apis.object_schema import ObjectSchema
        from agentle.agents.apis.array_schema import ArraySchema
        from agentle.agents.apis.primitive_schema import PrimitiveSchema

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

        Fixed version that creates proper tool parameter schemas.

        Returns:
            Tool parameter schema dictionary
        """
        from agentle.agents.apis.object_schema import ObjectSchema
        from agentle.agents.apis.array_schema import ArraySchema

        # Start with basic info
        param_schema: dict[str, Any] = {
            "description": self.description,
            "required": self.required,
        }

        if self.default is not None:
            param_schema["default"] = self.default

        # Handle parameter schema if available
        if self.parameter_schema is not None:
            if isinstance(self.parameter_schema, ObjectSchema):
                # For object parameters
                param_schema["type"] = "object"

                # Convert properties
                if self.parameter_schema.properties:
                    properties_dict: dict[str, Any] = {}
                    for (
                        prop_name,
                        prop_schema,
                    ) in self.parameter_schema.properties.items():
                        properties_dict[prop_name] = (
                            self._convert_schema_to_tool_property(prop_schema)
                        )
                    param_schema["properties"] = properties_dict

                # Add required properties
                if self.parameter_schema.required:
                    param_schema["required_properties"] = list(
                        self.parameter_schema.required
                    )

                # Add example if available
                if self.parameter_schema.example:
                    param_schema["example"] = self.parameter_schema.example

            elif isinstance(self.parameter_schema, ArraySchema):
                # For array parameters
                param_schema["type"] = "array"
                param_schema["items"] = self._convert_schema_to_tool_property(
                    self.parameter_schema.items
                )

                if self.parameter_schema.example:
                    param_schema["example"] = list(self.parameter_schema.example)

            else:
                # For primitive parameters
                param_schema["type"] = self.parameter_schema.type

                if (
                    hasattr(self.parameter_schema, "enum")
                    and self.parameter_schema.enum
                ):
                    param_schema["enum"] = list(self.parameter_schema.enum)
                if (
                    hasattr(self.parameter_schema, "example")
                    and self.parameter_schema.example
                ):
                    param_schema["example"] = self.parameter_schema.example
        else:
            # Fallback for old param_type style
            param_schema["type"] = self.param_type or "string"
            if hasattr(self, "enum") and self.enum:
                param_schema["enum"] = list(self.enum)

        return param_schema

    def _convert_schema_to_tool_property(self, schema: Any) -> dict[str, Any]:
        """Convert a parameter schema to tool property format."""
        if hasattr(schema, "type"):
            if schema.type == "object" and hasattr(schema, "properties"):
                # Object schema
                result: dict[str, Any] = {"type": "object"}
                if schema.properties:
                    properties_dict: dict[str, Any] = {}
                    for prop_name, prop_schema in schema.properties.items():
                        properties_dict[prop_name] = (
                            self._convert_schema_to_tool_property(prop_schema)
                        )
                    result["properties"] = properties_dict
                return result
            elif schema.type == "array" and hasattr(schema, "items"):
                # Array schema
                return {
                    "type": "array",
                    "items": self._convert_schema_to_tool_property(schema.items),
                }
            else:
                # Primitive schema
                result = {"type": schema.type}
                if hasattr(schema, "enum") and schema.enum:
                    result["enum"] = list(schema.enum)
                return result

        # Fallback
        return {"type": "string"}

    def _schema_to_tool_property(
        self, schema: ObjectSchema | ArraySchema | PrimitiveSchema
    ) -> dict[str, Any]:
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
