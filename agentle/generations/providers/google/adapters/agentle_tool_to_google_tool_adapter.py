"""
Adapter module for converting Agentle Tool objects to Google AI Tool format.

This module provides the AgentleToolToGoogleToolAdapter class, which transforms
Agentle's internal Tool representation into the Tool format expected by Google's
Generative AI APIs. This conversion is necessary when using Agentle tools with
Google's AI models that support function calling capabilities.

The adapter handles the mapping of Agentle tool definitions, including parameters,
types, and descriptions, to Google's schema-based function declaration format.
It includes comprehensive type mapping between Agentle's string-based types and
Google's enumerated Type values.

This adapter is typically used internally by the GoogleGenerationProvider when
preparing tool definitions to be sent to Google's API.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Mapping, TypedDict, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai import types

# Constants for validation
MAX_FUNCTION_NAME_LENGTH = 64
FUNCTION_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\.\-]*$")
MAX_PARAM_NAME_LENGTH = 64
PARAM_NAME_PATTERN = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_]*$")

# Special JSON Schema keywords that should be allowed despite not matching the normal pattern
JSON_SCHEMA_KEYWORDS = {"$schema", "$ref", "$id", "$defs", "$comment", "$vocabulary"}

# Type aliases for better readability
JSONValue = bool | int | float | str | list[Any] | dict[str, Any] | None
JSONObject = dict[str, JSONValue]


class JSONSchemaDict(TypedDict, total=False):
    """Type definition for JSON Schema dictionary."""

    type: str
    description: str | None
    default: Any
    properties: dict[str, "JSONSchemaDict"]
    items: "JSONSchemaDict" | list["JSONSchemaDict"]
    required: list[str]
    minItems: int | None
    maxItems: int | None
    minLength: int | None
    maxLength: int | None
    pattern: str | None
    minimum: float | None
    maximum: float | None
    enum: list[Any] | None
    additionalProperties: bool | None
    schema: str | None  # Using schema instead of $schema due to Python syntax
    nullable: bool | None
    anyOf: list["JSONSchemaDict"] | None
    allOf: list["JSONSchemaDict"] | None
    oneOf: list["JSONSchemaDict"] | None
    not_: JSONSchemaDict | None  # Using not_ instead of not due to Python syntax


class TypeParser:
    """Utility class for parsing complex Python type annotations."""

    @staticmethod
    def parse_type_string(type_str: str) -> dict[str, Any]:
        """
        Parse a type string and return normalized type information.

        Args:
            type_str: String representation of the type (e.g., "typing.Optional[int]")

        Returns:
            Dictionary with type information including base_type, is_optional, etc.
        """
        result: dict[str, Any] = {
            "base_type": "object",
            "is_optional": False,
            "is_list": False,
            "list_item_type": None,
            "union_types": None,
            "is_dict": False,
            "dict_value_type": None,
        }

        # Clean up the type string
        type_str = type_str.strip()

        # Handle None type
        if type_str in ("None", "NoneType", "type(None)"):
            result["base_type"] = "null"
            return result

        # Handle typing module prefixes
        type_str = type_str.replace("typing.", "")
        type_str = type_str.replace("types.", "")

        # Handle Optional[T] -> Union[T, None]
        optional_match = re.match(r"Optional\[(.*)\]", type_str)
        if optional_match:
            result["is_optional"] = True
            inner_type = optional_match.group(1)
            # Recursively parse the inner type
            inner_result = TypeParser.parse_type_string(inner_type)
            result.update(inner_result)
            return result

        # Handle Union types
        union_match = re.match(r"Union\[(.*)\]", type_str)
        if union_match:
            union_content = union_match.group(1)
            union_types = TypeParser._parse_union_types(union_content)

            # Check if it's actually Optional (Union with None)
            if "None" in union_types or "NoneType" in union_types:
                result["is_optional"] = True
                non_none_types = [
                    t for t in union_types if t not in ("None", "NoneType")
                ]

                if len(non_none_types) == 1:
                    # It's Optional[T]
                    inner_result = TypeParser.parse_type_string(non_none_types[0])
                    result.update(inner_result)
                else:
                    # It's a Union with None and multiple other types
                    result["base_type"] = (
                        "object"  # Default to object for complex unions
                    )
                    result["union_types"] = non_none_types
            else:
                # It's a Union without None
                result["base_type"] = "object"  # Default to object for unions
                result["union_types"] = union_types

            return result

        # Handle List/list types
        list_match = re.match(r"(?:List|list)\[(.*)\]", type_str, re.IGNORECASE)
        if list_match:
            result["is_list"] = True
            result["base_type"] = "array"
            item_type = list_match.group(1)
            result["list_item_type"] = TypeParser.parse_type_string(item_type)
            return result

        # Handle Dict/dict types
        dict_match = re.match(
            r"(?:Dict|dict|Mapping)\[([^,]+),\s*(.*)\]", type_str, re.IGNORECASE
        )
        if dict_match:
            result["is_dict"] = True
            result["base_type"] = "object"
            # We mainly care about the value type for schema generation
            value_type = dict_match.group(2)
            result["dict_value_type"] = TypeParser.parse_type_string(value_type)
            return result

        # Handle Tuple types (treat as array)
        tuple_match = re.match(r"(?:Tuple|tuple)\[(.*)\]", type_str, re.IGNORECASE)
        if tuple_match:
            result["is_list"] = True
            result["base_type"] = "array"
            # For simplicity, use object as item type for heterogeneous tuples
            result["list_item_type"] = {"base_type": "object"}
            return result

        # Handle Set types (treat as array)
        set_match = re.match(
            r"(?:Set|set|FrozenSet|frozenset)\[(.*)\]", type_str, re.IGNORECASE
        )
        if set_match:
            result["is_list"] = True
            result["base_type"] = "array"
            item_type = set_match.group(1)
            result["list_item_type"] = TypeParser.parse_type_string(item_type)
            return result

        # Handle Any type
        if type_str in ("Any", "any"):
            result["base_type"] = "object"
            return result

        # Handle simple types
        simple_type_map = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
            "bytes": "string",  # Treat bytes as string with base64 encoding
            "datetime": "string",  # ISO format string
            "date": "string",
            "time": "string",
            "timedelta": "string",
            "uuid": "string",
            "UUID": "string",
            "Path": "string",
            "Decimal": "number",
        }

        # Check for simple types
        for simple_type, schema_type in simple_type_map.items():
            if type_str.lower() == simple_type.lower():
                result["base_type"] = schema_type
                return result

        # If we can't parse it, default to object
        result["base_type"] = "object"
        return result

    @staticmethod
    def _parse_union_types(union_content: str) -> list[str]:
        """Parse the content of a Union type annotation."""
        types = []
        depth = 0
        current_type = ""

        for char in union_content:
            if char == "[":
                depth += 1
                current_type += char
            elif char == "]":
                depth -= 1
                current_type += char
            elif char == "," and depth == 0:
                types.append(current_type.strip())
                current_type = ""
            else:
                current_type += char

        if current_type.strip():
            types.append(current_type.strip())

        return types


class AgentleToolToGoogleToolAdapter(Adapter[Tool[Any], "types.Tool"]):
    """
    Adapter for converting Agentle Tool objects to Google AI Tool format.

    This adapter transforms Agentle's Tool objects into the FunctionDeclaration-based
    Tool format used by Google's Generative AI APIs. It handles the mapping between
    Agentle's parameter definitions and Google's schema-based format, including
    type conversion, required parameters, and default values.

    The adapter implements Agentle's provider abstraction layer pattern, which allows
    tools defined once to be used across different AI providers without modification.

    Key features:
    - Conversion of parameter types from string-based to Google's Type enum
    - Handling of complex Python typing annotations (Optional, Union, List, etc.)
    - Support for nested object structures
    - Handling of required parameters
    - Support for default values
    - Comprehensive JSON Schema support
    """

    def __init__(self) -> None:
        """Initialize the adapter with a logger."""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._type_parser = TypeParser()

    def _validate_function_name(self, name: str) -> None:
        """Validate function name according to Google's requirements."""
        if not name:
            raise ValueError("Function name cannot be empty")
        if len(name) > MAX_FUNCTION_NAME_LENGTH:
            raise ValueError(
                f"Function name cannot exceed {MAX_FUNCTION_NAME_LENGTH} characters"
            )
        if not FUNCTION_NAME_PATTERN.match(name):
            raise ValueError(
                "Function name must start with a letter or underscore and contain only "
                + "letters, numbers, underscores, dots, or dashes"
            )

    def _validate_parameter_name(self, name: str) -> None:
        """Validate parameter name according to Google's requirements."""
        if not name:
            raise ValueError("Parameter name cannot be empty")
        if len(name) > MAX_PARAM_NAME_LENGTH:
            raise ValueError(
                f"Parameter name cannot exceed {MAX_PARAM_NAME_LENGTH} characters"
            )

        # Allow special JSON Schema keywords
        if name in JSON_SCHEMA_KEYWORDS:
            return

        if not PARAM_NAME_PATTERN.match(name):
            raise ValueError(
                "Parameter name must start with a letter, underscore, or $ and contain only letters, numbers, or underscores"
            )

    def _get_google_type_from_parsed(self, parsed_type: dict[str, Any]) -> "types.Type":
        """Convert parsed type information to Google Type enum."""
        from google.genai import types

        type_mapping = {
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,
            "object": types.Type.OBJECT,
            "null": types.Type.STRING,  # Google doesn't have NULL type, use STRING
        }

        base_type = parsed_type.get("base_type", "object")
        google_type = type_mapping.get(base_type, types.Type.OBJECT)

        return google_type

    def _get_google_type(self, param_type_str: str, param_name: str) -> "types.Type":
        """Convert type string to Google Type enum, handling complex types."""
        try:
            # First try to parse as a complex type
            parsed_type = self._type_parser.parse_type_string(str(param_type_str))
            return self._get_google_type_from_parsed(parsed_type)
        except Exception as e:
            self._logger.warning(
                f"Failed to parse type '{param_type_str}' for parameter '{param_name}': {e}, "
                + "defaulting to OBJECT type"
            )
            from google.genai import types

            return types.Type.OBJECT

    def _create_schema_from_parsed_type(
        self, parsed_type: dict[str, Any], param_name: str = ""
    ) -> "types.Schema":
        """Create a Google Schema from parsed type information."""
        from google.genai import types

        google_type = self._get_google_type_from_parsed(parsed_type)

        # Create base schema
        schema = types.Schema(type=google_type)

        # Handle array types
        if parsed_type.get("is_list") and parsed_type.get("list_item_type"):
            item_type = parsed_type["list_item_type"]
            schema.items = self._create_schema_from_parsed_type(
                item_type, f"{param_name}[items]"
            )

        # Handle dict types (as object with additional properties)
        elif parsed_type.get("is_dict"):
            # For dict types, we can't specify the schema of arbitrary keys
            # But we can specify that additional properties are allowed
            schema.properties = {}
            # Note: Google's Schema might not support additionalProperties directly
            # This is a limitation we have to work with

        # Handle union types (default to object for now)
        elif parsed_type.get("union_types"):
            # Google doesn't support union types directly
            # We have to default to a more general type
            self._logger.info(
                f"Union type for '{param_name}' converted to OBJECT type. "
                + f"Original union types: {parsed_type['union_types']}"
            )

        return schema

    def _create_schema_from_json_schema(
        self, schema_dict: Mapping[str, Any], param_name: str = ""
    ) -> "types.Schema":
        """Create a Google Schema from a JSON Schema definition."""
        from google.genai import types

        # Handle null type
        if schema_dict.get("type") == "null":
            # Google doesn't have a null type, treat as optional string
            return types.Schema(type=types.Type.STRING, nullable=True)

        # Handle anyOf/oneOf/allOf (composite schemas)
        if "anyOf" in schema_dict or "oneOf" in schema_dict or "allOf" in schema_dict:
            # Google doesn't support these directly, default to object
            self._logger.info(
                f"Composite schema (anyOf/oneOf/allOf) for '{param_name}' converted to OBJECT type"
            )
            schema = types.Schema(
                type=types.Type.OBJECT,
                description=str(schema_dict.get("description"))
                if schema_dict.get("description")
                else None,
                default=schema_dict.get("default"),
            )
            return schema

        # Get the type - handle both single type and array of types
        schema_type: Any = schema_dict.get("type", "object")

        parsed_type: dict[str, Any] = {}

        # Handle array of types (JSON Schema allows type: ["string", "null"])
        if isinstance(schema_type, list):
            # Check if nullable
            is_nullable = "null" in schema_type
            non_null_types: list[str] = [t for t in schema_type if t != "null"]

            if non_null_types:
                schema_type = non_null_types[0]  # Use the first non-null type
            else:
                schema_type = "string"  # Default to string if only null
        else:
            is_nullable = schema_dict.get("nullable", False)

        # Parse the type if it's a complex string
        if isinstance(schema_type, str):
            parsed_type = self._type_parser.parse_type_string(schema_type)
            google_type = self._get_google_type_from_parsed(parsed_type)
        else:
            google_type = self._get_google_type(str(schema_type), param_name)

        # Create base schema
        schema = types.Schema(
            type=google_type,
            description=str(schema_dict.get("description"))
            if schema_dict.get("description")
            else None,
            default=schema_dict.get("default"),
        )

        # Add nullable flag if applicable (Google might not support this directly)
        if (
            is_nullable or parsed_type.get("is_optional", False)
            if isinstance(schema_type, str)
            else False
        ):
            # Mark as nullable - Google's implementation might handle this differently
            schema.nullable = True

        # Handle array type
        if google_type == types.Type.ARRAY:
            items_schema = schema_dict.get("items", {})
            if isinstance(items_schema, dict):
                schema.items = self._create_schema_from_json_schema(
                    cast(Mapping[str, Any], items_schema), f"{param_name}[items]"
                )
            elif isinstance(items_schema, list):
                # Tuple validation - use first item schema for simplicity
                if items_schema:
                    schema.items = self._create_schema_from_json_schema(
                        cast(Mapping[str, Any], items_schema[0]), f"{param_name}[items]"
                    )
                else:
                    schema.items = types.Schema(type=types.Type.STRING)
            else:
                # Default to string items if items schema is not an object
                schema.items = types.Schema(type=types.Type.STRING)

            # Add array constraints
            if "minItems" in schema_dict:
                schema.min_items = int(schema_dict["minItems"])
            if "maxItems" in schema_dict:
                schema.max_items = int(schema_dict["maxItems"])

        # Handle object type
        elif google_type == types.Type.OBJECT:
            properties = schema_dict.get("properties", {})
            schema_properties: dict[str, types.Schema] = {}

            for prop_name, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    continue

                # Skip validation for JSON Schema keywords
                if prop_name not in JSON_SCHEMA_KEYWORDS:
                    try:
                        self._validate_parameter_name(prop_name)
                    except ValueError as e:
                        self._logger.warning(
                            f"Skipping invalid property name '{prop_name}': {e}"
                        )
                        continue

                schema_properties[prop_name] = self._create_schema_from_json_schema(
                    cast(Mapping[str, Any], prop_schema),
                    f"{param_name}.{prop_name}" if param_name else prop_name,
                )

            if schema_properties:
                schema.properties = schema_properties

            # Handle required properties
            if "required" in schema_dict:
                required = schema_dict["required"]
                if isinstance(required, list):
                    schema.required = required

            # Handle additionalProperties
            if "additionalProperties" in schema_dict:
                additional = schema_dict["additionalProperties"]
                if isinstance(additional, bool):
                    # Google might not support this directly
                    pass
                elif isinstance(additional, dict):
                    # Could be a schema for additional properties
                    # Google might not support this pattern
                    pass

        # Handle string type
        elif google_type == types.Type.STRING:
            if "minLength" in schema_dict:
                schema.min_length = int(schema_dict["minLength"])
            if "maxLength" in schema_dict:
                schema.max_length = int(schema_dict["maxLength"])
            if "pattern" in schema_dict:
                schema.pattern = str(schema_dict["pattern"])
            if "format" in schema_dict:
                # Store format as a hint (Google might not use it)
                schema.format = str(schema_dict["format"])

        # Handle number/integer type
        elif google_type in (types.Type.NUMBER, types.Type.INTEGER):
            if "minimum" in schema_dict:
                schema.minimum = float(schema_dict["minimum"])
            if "maximum" in schema_dict:
                schema.maximum = float(schema_dict["maximum"])

        # Handle enums for any type
        if "enum" in schema_dict:
            enum_values = schema_dict["enum"]
            if isinstance(enum_values, list):
                schema.enum = enum_values

        # Handle const (single allowed value)
        if "const" in schema_dict:
            schema.enum = [schema_dict["const"]]

        return schema

    def _convert_agentle_params_to_json_schema(
        self, agentle_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert Agentle's flat parameter format to proper JSON Schema format.

        Agentle format:
        {
            'param1': {'type': 'str', 'required': True, 'description': '...'},
            'param2': {'type': 'typing.Optional[int]', 'required': False, 'default': 42}
        }

        JSON Schema format:
        {
            'type': 'object',
            'properties': {
                'param1': {'type': 'string', 'description': '...'},
                'param2': {'type': 'integer', 'default': 42, 'nullable': true}
            },
            'required': ['param1']
        }
        """
        # Check if this is already in JSON Schema format
        if "type" in agentle_params and "properties" in agentle_params:
            return dict(agentle_params)

        # Check if it's a $schema format
        if "$schema" in agentle_params or "properties" in agentle_params:
            # It's likely already a JSON Schema
            if "type" not in agentle_params:
                # Add default type as object if not specified
                result = {"type": "object"}
                result.update(agentle_params)
                return result
            return dict(agentle_params)

        # Convert from Agentle flat format to JSON Schema format
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param_info in agentle_params.items():
            if not isinstance(param_info, dict):
                continue

            # Skip JSON Schema keywords if they appear at this level
            if param_name in JSON_SCHEMA_KEYWORDS:
                continue

            # Validate parameter name
            try:
                self._validate_parameter_name(param_name)
            except ValueError as e:
                self._logger.warning(
                    f"Skipping invalid parameter name '{param_name}': {e}"
                )
                continue

            # Extract the parameter info
            param_type_str: str = cast(dict[str, Any], param_info).get("type", "string")
            is_required = param_info.get("required", False)

            # Parse the type
            parsed_type = self._type_parser.parse_type_string(str(param_type_str))

            # Create the property schema
            prop_schema: dict[str, Any] = {}

            # Set the base type
            if parsed_type["base_type"] == "array":
                prop_schema["type"] = "array"
                if parsed_type.get("list_item_type"):
                    item_parsed = parsed_type["list_item_type"]
                    prop_schema["items"] = {"type": item_parsed["base_type"]}
            else:
                prop_schema["type"] = parsed_type["base_type"]

            # Handle optional types
            if parsed_type.get("is_optional"):
                prop_schema["nullable"] = True
                # Don't mark optional parameters as required
                is_required = False

            # Copy over other attributes (excluding 'required' since it goes to the root level)
            for key, value in param_info.items():
                if key not in (
                    "required",
                    "type",
                ):  # Exclude these as they're handled separately
                    prop_schema[key] = cast(Any, value)

            properties[param_name] = prop_schema

            if is_required:
                required.append(param_name)

        result: dict[str, Any] = {"type": "object", "properties": properties}

        if required:
            result["required"] = required

        return result

    def adapt(self, agentle_tool: Tool[Any]) -> "types.Tool":
        """
        Convert an Agentle Tool to a Google AI Tool.

        Args:
            agentle_tool: The Agentle Tool object to convert.

        Returns:
            types.Tool: A Google AI Tool object.

        Raises:
            ValueError: If the tool name or parameters are invalid.
        """
        from google.genai import types

        try:
            # Validate function name
            self._validate_function_name(agentle_tool.name)

            # Convert parameters
            parameters_schema = None
            if agentle_tool.parameters:
                self._logger.debug(
                    f"Converting parameters for tool '{agentle_tool.name}': {agentle_tool.parameters}"
                )

                # Convert Agentle parameter format to JSON Schema format
                json_schema = self._convert_agentle_params_to_json_schema(
                    agentle_tool.parameters
                )
                self._logger.debug(f"Converted to JSON Schema: {json_schema}")

                # Create Google Schema from JSON Schema
                parameters_schema = self._create_schema_from_json_schema(json_schema)

                # Detailed logging for debugging
                if parameters_schema and parameters_schema.properties:
                    self._logger.debug(
                        f"Tool '{agentle_tool.name}' has {len(parameters_schema.properties)} parameters:"
                    )
                    for (
                        param_name,
                        param_schema,
                    ) in parameters_schema.properties.items():
                        required_params = parameters_schema.required or []
                        self._logger.debug(
                            f"  - {param_name}: type={param_schema.type}, "
                            + f"description='{param_schema.description}', "
                            + f"default={param_schema.default}, "
                            + f"required={param_name in required_params}, "
                            + f"nullable={getattr(param_schema, 'nullable', False)}"
                        )
                    self._logger.debug(
                        f"Required parameters: {parameters_schema.required or []}"
                    )
                else:
                    self._logger.debug(f"Tool '{agentle_tool.name}' has no parameters")

            # Create function declaration
            function_declaration = types.FunctionDeclaration(
                name=agentle_tool.name,
                description=agentle_tool.description or "",
                parameters=parameters_schema,
            )

            self._logger.info(
                f"Successfully created FunctionDeclaration for '{agentle_tool.name}' with "
                + f"{len(parameters_schema.properties) if parameters_schema and parameters_schema.properties else 0} parameters"
            )

            # Create and return tool
            return types.Tool(function_declarations=[function_declaration])

        except Exception as e:
            self._logger.error(
                f"Failed to adapt tool '{agentle_tool.name}': {e}", exc_info=True
            )
            raise
