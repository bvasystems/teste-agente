# Adapter for Agentle tool to OpenRouter tool
"""
Adapter for converting Agentle Tool definitions to OpenRouter tool format.

This module handles the conversion of Agentle's Tool objects into
the OpenRouter API tool definition format.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, override, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.json.json_schema_builder import JsonSchemaBuilder
from agentle.generations.tools.tool import Tool
from agentle.generations.providers.openrouter._types import (
    OpenRouterTool,
    OpenRouterToolFunction,
    OpenRouterToolFunctionParameters,
)

logger = logging.getLogger(__name__)


class AgentleToolToOpenRouterToolAdapter(Adapter[Tool, OpenRouterTool]):
    """
    Adapter for converting Agentle Tool objects to OpenRouter format.

    Converts tool definitions including name, description, and parameters
    to the format expected by OpenRouter's API.

    This adapter handles both flat parameter format (from Tool.from_callable)
    and JSON Schema format. It also handles complex types like BaseModel,
    TypedDict, dataclasses, etc. by expanding them to their JSON schemas.
    """

    def _is_complex_type(self, type_annotation: Any) -> bool:
        """
        Check if a type annotation represents a complex type that needs expansion.

        Args:
            type_annotation: The type annotation to check.

        Returns:
            True if the type is complex (BaseModel, TypedDict, dataclass, etc.).
        """
        if not inspect.isclass(type_annotation):
            return False

        # Check for common complex types
        try:
            # Check for BaseModel (Pydantic)
            if hasattr(type_annotation, "model_fields") or hasattr(
                type_annotation, "__fields__"
            ):
                return True

            # Check for TypedDict
            if hasattr(type_annotation, "__annotations__") and hasattr(
                type_annotation, "__required_keys__"
            ):
                return True

            # Check for dataclass
            if hasattr(type_annotation, "__dataclass_fields__"):
                return True

        except Exception:
            pass

        return False

    def _expand_complex_type(self, type_annotation: Any) -> dict[str, Any]:
        """
        Expand a complex type to its JSON schema representation.

        Args:
            type_annotation: The complex type to expand.

        Returns:
            JSON schema representation of the type.
        """
        try:
            schema = JsonSchemaBuilder(
                type_annotation,
                clean_output=True,
                use_defs_instead_of_definitions=True,
            ).build(dereference=True)

            # Remove the $defs key if present since we dereferenced
            schema.pop("$defs", None)
            schema.pop("definitions", None)

            logger.debug(
                f"Expanded complex type {type_annotation.__name__} to JSON schema"
            )
            return schema

        except Exception as e:
            logger.warning(
                f"Failed to expand complex type {type_annotation}: {e}. "
                + "Falling back to generic object type."
            )
            return {"type": "object"}

    def _resolve_type_annotation(self, type_str: str, tool: Tool) -> Any | None:
        """
        Resolve a type string to the actual type object.

        Args:
            type_str: String representation of the type.
            tool: The tool object (to access the callable's scope if needed).

        Returns:
            The resolved type object, or None if it cannot be resolved.
        """
        # Try to get the callable's module for type resolution
        if not tool.callable_ref:
            return None

        try:
            # Get the signature to access parameter annotations
            sig = inspect.signature(tool.callable_ref)
            for _, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    # Check if this parameter's annotation matches our type string
                    annotation_str = (
                        str(param.annotation).replace("<class '", "").replace("'>", "")
                    )
                    if (
                        annotation_str == type_str
                        or param.annotation.__name__ == type_str.split(".")[-1]
                    ):
                        return param.annotation
        except Exception as e:
            logger.debug(f"Could not resolve type annotation {type_str}: {e}")

        return None

    def _convert_to_json_schema(
        self, agentle_params: dict[str, Any], tool: Tool
    ) -> dict[str, Any]:
        """
        Convert Agentle's flat parameter format to proper JSON Schema format.

        Agentle format:
        {
            'param1': {'type': 'str', 'required': True, 'description': '...'},
            'param2': {'type': 'int', 'required': False, 'default': 42}
        }

        JSON Schema format:
        {
            'type': 'object',
            'properties': {
                'param1': {'type': 'string', 'description': '...'},
                'param2': {'type': 'integer', 'default': 42}
            },
            'required': ['param1']
        }

        This method also handles complex types like BaseModel, TypedDict, etc.
        by expanding them to their full JSON schema representation.

        Args:
            agentle_params: Parameters in Agentle's flat format or JSON Schema format.
            tool: The tool object (for resolving complex type annotations).

        Returns:
            Parameters in JSON Schema format.
        """
        # Check if this is already in JSON Schema format
        if "type" in agentle_params and "properties" in agentle_params:
            return agentle_params

        # Check if it's a $schema format (also JSON Schema)
        if "$schema" in agentle_params or "properties" in agentle_params:
            if "type" not in agentle_params:
                result = {"type": "object"}
                result.update(agentle_params)
                return result
            return agentle_params

        # Convert from Agentle flat format to JSON Schema format
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Type mapping from Python types to JSON Schema types
        type_mapping = {
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
            "none": "null",
            "nonetype": "null",
            "null": "null",
        }

        for param_name, param_info in agentle_params.items():
            if not isinstance(param_info, dict):
                continue

            # Extract the parameter info
            param_type_str: str = cast(str, param_info.get("type", "string"))
            is_required = param_info.get("required", False)

            # Create the property schema
            prop_schema: dict[str, Any] = {}

            # Check if this is a complex type that needs expansion
            type_annotation = self._resolve_type_annotation(param_type_str, tool)

            if type_annotation and self._is_complex_type(type_annotation):
                # Expand the complex type to its full JSON schema
                logger.debug(
                    f"Expanding complex type for parameter '{param_name}': {param_type_str}"
                )
                prop_schema = self._expand_complex_type(type_annotation)
            else:
                # Map the type to JSON Schema type
                if "|" in param_type_str:
                    # Handle union types like "str | None"
                    parts = [p.strip() for p in param_type_str.split("|")]
                    json_types: list[str] = []
                    
                    for part in parts:
                        mapped = type_mapping.get(part.lower(), part)
                        if mapped not in json_types:
                            json_types.append(mapped)
                    
                    if len(json_types) == 1:
                        prop_schema["type"] = json_types[0]
                    else:
                        prop_schema["type"] = json_types
                else:
                    json_type = type_mapping.get(param_type_str.lower(), param_type_str)
                    prop_schema["type"] = json_type

            # Copy over other attributes (excluding 'required' and 'type')
            for key, value in param_info.items():
                if key not in ("required", "type"):
                    # Don't overwrite if already set by complex type expansion
                    if key not in prop_schema:
                        prop_schema[key] = value

            properties[param_name] = prop_schema

            if is_required:
                required.append(param_name)

        result: dict[str, Any] = {"type": "object", "properties": properties}

        if required:
            result["required"] = required

        return result

    @override
    def adapt(self, tool: Tool) -> OpenRouterTool:
        """
        Convert an Agentle Tool to OpenRouter format.

        Args:
            tool: The Agentle Tool to convert.

        Returns:
            The corresponding OpenRouter tool definition.
        """
        # Convert parameters to JSON Schema format
        json_schema_params = self._convert_to_json_schema(tool.parameters, tool)

        return OpenRouterTool(
            type="function",
            function=OpenRouterToolFunction(
                name=tool.name,
                description=tool.description or "",
                parameters=OpenRouterToolFunctionParameters(
                    type="object",
                    properties=json_schema_params.get("properties", {}),
                    required=json_schema_params.get("required", []),
                ),
            ),
        )
