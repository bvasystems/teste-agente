"""
JsonSchemaExtractor - Convert Python callables to OpenAI JSON Schema format
"""

import inspect
from types import EllipsisType
import typing
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NotRequired,
    Optional,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
    Literal,
    Tuple,
    Set,
    ForwardRef,
)
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import re
from pydantic import BaseModel


@dataclass
class JsonSchemaConfig:
    """Configuration for JSON Schema generation"""

    ensure_additional_properties: bool = True  # Always set additionalProperties: false
    include_descriptions: bool = True  # Include docstring descriptions
    strict_mode: bool = True  # Generate strict schemas for Structured Outputs
    max_enum_values: int = 1000  # Maximum enum values allowed
    max_nesting_depth: int = 10  # Maximum object nesting depth
    make_all_required: bool = (
        True  # Make all fields required (can use Optional for nullable)
    )
    dereference: bool = False  # Inline all $ref definitions (no references)


class FunctionExtractionConfig(TypedDict):
    extract_return_type: NotRequired[bool]


class JsonSchemaExtractor:
    """
    Extracts JSON Schema from Python callables (functions, methods, classes)
    Compatible with OpenAI Structured Outputs requirements.
    """

    def __init__(self, config: Optional[JsonSchemaConfig] = None):
        self.config = config or JsonSchemaConfig()
        self._definitions: Dict[str, Any] = {}
        self._seen_types: Set[type] = set()
        self._current_depth = 0
        self._type_schemas: Dict[type, Dict[str, Any]] = {}  # Cache for dereferencing
        self._building_types: Dict[
            type, Dict[str, Any]
        ] = {}  # Types currently being built (for circular refs)
        self._param_descriptions: Dict[
            str, str
        ] = {}  # Parameter descriptions from docstring

    def extract(
        self,
        target: Callable[..., Any] | type[BaseModel],
        *,
        function_extraction_config: FunctionExtractionConfig | None = None,
    ) -> Dict[str, Any]:
        """
        Extract JSON Schema from a callable or Pydantic BaseModel.

        Args:
            target: A callable (function, method, or class) or a Pydantic BaseModel class
            extract_return_type: If True, extract schema from return type instead of parameters

        Returns:
            Dictionary containing the JSON Schema
        """
        self._definitions = {}
        self._seen_types = set()
        self._current_depth = 0
        self._type_schemas = {}
        self._building_types = {}
        self._param_descriptions = {}

        # Check if target is a Pydantic BaseModel
        if inspect.isclass(target) and issubclass(target, BaseModel):
            return self._extract_from_pydantic(target)

        # Otherwise, it's a callable - use existing logic
        func = target

        # Get function signature
        sig = inspect.signature(func)

        # Get function name and description
        func_name = func.__name__
        description = self._extract_description(func)

        # Extract parameter descriptions from docstring
        self._param_descriptions = self._extract_param_descriptions(func)

        function_extraction_config = function_extraction_config or {}

        if function_extraction_config.get("extract_return_type"):
            # Extract schema from return type
            return_annotation = sig.return_annotation

            if return_annotation == inspect.Signature.empty:
                raise ValueError(f"Function {func_name} has no return type annotation")

            # Get the schema for the return type
            schema = self._get_type_schema(return_annotation, func_name)

            # If it's a reference to a definition, unwrap it
            if "$ref" in schema:
                ref_name = schema["$ref"].split("/")[-1]
                if ref_name in self._definitions:
                    schema = self._definitions[ref_name].copy()

            # Add definitions if any (unless dereferencing)
            if self._definitions and not self.config.dereference:
                schema["$defs"] = self._definitions
            elif self.config.dereference:
                # Dereference all $ref in the schema
                schema = self._dereference_schema(schema)
        else:
            # Build properties from parameters
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                # Skip self, cls, *args, **kwargs
                if param_name in ("self", "cls") or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                param_schema = self._get_type_schema(
                    param.annotation, param_name, param.default
                )

                properties[param_name] = param_schema

                # Determine if required
                if (
                    self.config.make_all_required
                    or param.default == inspect.Parameter.empty
                ):
                    required.append(param_name)

            # Build the schema
            schema: dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

            if self.config.ensure_additional_properties:
                schema["additionalProperties"] = False

            # Add definitions if any (unless dereferencing)
            if self._definitions and not self.config.dereference:
                schema["$defs"] = self._definitions
            elif self.config.dereference:
                # Dereference all $ref in the schema
                schema = self._dereference_schema(schema)

        # Build complete schema with metadata
        complete_schema = {
            "name": func_name,
            "strict": self.config.strict_mode,
            "schema": schema,
        }

        if description and self.config.include_descriptions:
            complete_schema["description"] = description

        return complete_schema

    def _extract_from_pydantic(self, model_class: type[BaseModel]) -> Dict[str, Any]:
        """
        Extract JSON Schema from a Pydantic BaseModel.

        Args:
            model_class: A Pydantic BaseModel class

        Returns:
            Dictionary containing the JSON Schema
        """
        # Get the base schema from Pydantic
        if self.config.dereference:
            # Use mode='serialization' to get a clean schema
            base_schema = model_class.model_json_schema(
                mode="serialization", ref_template="{model}"
            )
        else:
            base_schema = model_class.model_json_schema(mode="serialization")

        # Extract the main schema and definitions
        schema = base_schema.copy()
        definitions = schema.pop("$defs", {})

        # Apply our configuration settings
        schema = self._apply_config_to_pydantic_schema(schema, definitions)

        # Handle dereferencing if needed
        if self.config.dereference:
            # Store definitions temporarily for dereferencing
            self._definitions = definitions
            schema = self._dereference_schema(schema)
            # Clear definitions after dereferencing since they're inlined
            if "$defs" in schema:
                del schema["$defs"]
        else:
            # Add definitions back if not dereferencing
            if definitions:
                schema["$defs"] = definitions

        # Build complete schema with metadata
        model_name = model_class.__name__
        description = model_class.__doc__

        complete_schema = {
            "name": model_name,
            "strict": self.config.strict_mode,
            "schema": schema,
        }

        if description and self.config.include_descriptions:
            complete_schema["description"] = description.strip()

        return complete_schema

    def _apply_config_to_pydantic_schema(
        self, schema: Dict[str, Any], definitions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply JsonSchemaConfig settings to a Pydantic-generated schema.

        Args:
            schema: The main schema object
            definitions: The definitions ($defs) object

        Returns:
            Modified schema with config applied
        """
        # Apply to main schema
        schema = self._apply_config_to_schema_object(schema)

        # Apply to all definitions
        for def_name, def_schema in definitions.items():
            definitions[def_name] = self._apply_config_to_schema_object(def_schema)

        return schema

    def _apply_config_to_schema_object(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively apply config to a schema object.

        Args:
            schema: A schema object (can be nested)

        Returns:
            Modified schema
        """
        # Make a copy to avoid modifying the original
        schema = schema.copy()

        # Handle additionalProperties
        if self.config.ensure_additional_properties and schema.get("type") == "object":
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = False

        # Handle descriptions
        if not self.config.include_descriptions:
            if "description" in schema:
                del schema["description"]

        # Handle required fields based on make_all_required
        if self.config.make_all_required and schema.get("type") == "object":
            # Get all properties
            properties = schema.get("properties", {})
            # Make all properties required
            if properties:
                schema["required"] = list(properties.keys())

        # Recursively apply to nested schemas
        if "properties" in schema:
            schema["properties"] = {
                key: self._apply_config_to_schema_object(value)
                for key, value in schema["properties"].items()
            }

        if "items" in schema:
            schema["items"] = self._apply_config_to_schema_object(schema["items"])

        if "additionalProperties" in schema and isinstance(
            schema["additionalProperties"], dict
        ):
            schema["additionalProperties"] = self._apply_config_to_schema_object(
                cast(Dict[str, Any], schema["additionalProperties"])
            )

        if "anyOf" in schema:
            schema["anyOf"] = [
                self._apply_config_to_schema_object(s) for s in schema["anyOf"]
            ]

        if "allOf" in schema:
            schema["allOf"] = [
                self._apply_config_to_schema_object(s) for s in schema["allOf"]
            ]

        if "oneOf" in schema:
            schema["oneOf"] = [
                self._apply_config_to_schema_object(s) for s in schema["oneOf"]
            ]

        return schema

    def _extract_description(self, func: Callable[..., Any]) -> Optional[str]:
        """Extract description from docstring"""
        doc = inspect.getdoc(func)
        if doc:
            # Get first line or paragraph
            lines = doc.strip().split("\n")
            description: list[str] = []
            for line in lines:
                line = line.strip()
                if not line:
                    break
                description.append(line)
            return " ".join(description)
        return None

    def _extract_param_descriptions(self, func: Callable[..., Any]) -> Dict[str, str]:
        """Extract parameter descriptions from docstring (supports Google, NumPy, Sphinx styles)"""
        doc = inspect.getdoc(func)
        if not doc:
            return {}

        param_descriptions = {}
        lines = doc.split("\n")

        # Try different docstring formats
        in_params_section = False
        current_param = None
        current_desc_lines = []

        for line in lines:
            stripped = line.strip()

            # Detect parameter sections (Google/NumPy style)
            if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
                in_params_section = True
                continue

            # End of parameter section
            if in_params_section and stripped and not line.startswith((" ", "\t")):
                # Save last param if any
                if current_param and current_desc_lines:
                    param_descriptions[current_param] = " ".join(
                        current_desc_lines
                    ).strip()
                in_params_section = False
                current_param = None
                current_desc_lines = []
                continue

            if in_params_section:
                # Google style: param_name: description or param_name (type): description
                if ":" in stripped and not stripped.startswith(":"):
                    # Save previous param
                    if current_param and current_desc_lines:
                        param_descriptions[current_param] = " ".join(
                            current_desc_lines
                        ).strip()

                    # Parse new param
                    parts = stripped.split(":", 1)
                    param_part = parts[0].strip()

                    # Remove type annotation if present: "param_name (type)" -> "param_name"
                    if "(" in param_part:
                        param_part = param_part.split("(")[0].strip()

                    current_param = param_part
                    current_desc_lines = [parts[1].strip()] if len(parts) > 1 else []
                elif current_param and stripped:
                    # Continuation of previous description
                    current_desc_lines.append(stripped)

            # Sphinx style: :param param_name: description
            if stripped.startswith(":param "):
                # Save previous param
                if current_param and current_desc_lines:
                    param_descriptions[current_param] = " ".join(
                        current_desc_lines
                    ).strip()

                # Parse Sphinx format
                match = re.match(r":param\s+(\w+):\s*(.*)$", stripped)
                if match:
                    current_param = match.group(1)
                    current_desc_lines = [match.group(2)] if match.group(2) else []

        # Save last param
        if current_param and current_desc_lines:
            param_descriptions[current_param] = " ".join(current_desc_lines).strip()

        return param_descriptions

    def _resolve_forward_ref(
        self, type_hint: Any, context_globals: Optional[dict[str, Any]] = None
    ) -> Optional[type]:
        """
        Resolve a forward reference (string type annotation) to the actual type.
        Always returns the resolved class - circular reference detection happens elsewhere.
        """
        if isinstance(type_hint, str):
            # It's a string annotation
            ref_name = type_hint
        elif isinstance(type_hint, ForwardRef):
            # It's a ForwardRef object
            ref_name = type_hint.__forward_arg__
        else:
            return type_hint

        # First, check if it's in our currently known types (for circular refs)
        # Check _seen_types, _building_types, and _type_schemas
        all_known_types = (
            list(self._seen_types)
            + list(self._building_types.keys())
            + list(self._type_schemas.keys())
        )

        for type_cls in all_known_types:
            if hasattr(type_cls, "__name__") and type_cls.__name__ == ref_name:
                # Found it! Return the class itself
                # The circular reference will be detected in _handle_dataclass
                return type_cls

        # Try context globals if provided
        if context_globals and ref_name in context_globals:
            resolved = context_globals[ref_name]
            if inspect.isclass(resolved):
                return resolved

        # Try to find the type in the calling frame's globals

        frame = inspect.currentframe()
        for _ in range(15):  # Look up to 15 frames
            if frame is None:
                break
            frame = frame.f_back
            if frame is None:
                break

            # Check the frame's globals and locals
            if ref_name in frame.f_globals:
                resolved = frame.f_globals[ref_name]
                if inspect.isclass(resolved):
                    return resolved

            if ref_name in frame.f_locals:
                resolved = frame.f_locals[ref_name]
                if inspect.isclass(resolved):
                    return resolved

        return None

    def _get_type_schema(
        self,
        type_hint: Any,
        field_name: str = "",
        default: Any = inspect.Parameter.empty,
    ) -> Dict[str, Any]:
        """Convert a type hint to JSON Schema"""

        # Handle missing annotation
        if type_hint == inspect.Parameter.empty:
            return {"type": "string", "description": f"Parameter {field_name}"}

        # Handle ForwardRef (string annotations like 'ClassName')
        # Check if it's a string first (most common case for forward refs in dataclasses)
        if isinstance(type_hint, str):
            # Try to resolve the forward reference
            resolved = self._resolve_forward_ref(type_hint)
            if resolved is not None:
                type_hint = resolved
            else:
                # Can't resolve, treat as string
                return {"type": "string", "description": f"Parameter {field_name}"}
        elif isinstance(type_hint, ForwardRef):
            # Try to resolve the forward reference
            resolved = self._resolve_forward_ref(type_hint)
            if resolved is not None:
                type_hint = resolved
            else:
                # Can't resolve, treat as string
                return {"type": "string", "description": f"Parameter {field_name}"}

        # Handle primitive types first (before get_origin)
        if type_hint in (str, int, float, bool, type(None)):
            return self._handle_primitive(type_hint, field_name)

        # Get origin and args for generic types
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle Union types (including Optional)
        if origin is Union:
            return self._handle_union(args, field_name)

        # Handle Literal
        if origin is Literal:
            return self._handle_literal(args, field_name)

        # Handle List
        if origin is list or origin is List:
            return self._handle_list(args, field_name)

        # Handle Dict
        if origin is dict or origin is Dict:
            return self._handle_dict(args, field_name)

        # Handle Tuple
        if origin is tuple or origin is Tuple:
            return self._handle_tuple(args, field_name)

        # Handle Enum (check before dataclass as enums can be dataclass-like)
        if inspect.isclass(type_hint) and issubclass(type_hint, Enum):
            return self._handle_enum(type_hint, field_name)

        # Handle dataclass
        if inspect.isclass(type_hint) and is_dataclass(type_hint):
            return self._handle_dataclass(type_hint, field_name)

        # Handle custom classes (try to extract as object)
        if inspect.isclass(type_hint):
            return self._handle_class(type_hint, field_name)

        # Fallback
        return {"type": "string", "description": f"Parameter {field_name}"}

    def _handle_primitive(self, type_hint: type, field_name: str) -> Dict[str, Any]:
        """Handle primitive types"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
        }

        json_type = type_map.get(type_hint, "string")
        schema = {"type": json_type}

        if field_name and self.config.include_descriptions:
            # Use docstring description if available, otherwise use default
            schema["description"] = self._param_descriptions.get(
                field_name, f"The {field_name} parameter"
            )

        return schema

    def _handle_union(self, args: tuple[type, ...], field_name: str) -> Dict[str, Any]:
        """Handle Union types including Optional"""

        # Resolve any forward references in the args first
        resolved_args: list[type] = []
        for arg in args:
            if isinstance(arg, (str, ForwardRef)):
                resolved = self._resolve_forward_ref(arg)
                if resolved is not None:
                    resolved_args.append(resolved)
                else:
                    resolved_args.append(arg)
            else:
                resolved_args.append(arg)

        # Check if it's Optional (Union with None)
        non_none_types: list[type] = [
            arg for arg in resolved_args if arg is not type(None)
        ]
        has_none: bool = len(non_none_types) < len(resolved_args)

        if len(non_none_types) == 1:
            # It's Optional[X]
            schema: dict[str, Any] = self._get_type_schema(
                non_none_types[0], field_name
            )

            # Make it nullable
            if has_none:
                if isinstance(schema.get("type"), str):
                    schema["type"] = [schema["type"], "null"]
                elif isinstance(schema.get("type"), list):
                    if "null" not in schema["type"]:
                        schema["type"].append("null")
                else:
                    # Schema might not have a "type" field (e.g., it's an object reference)
                    # In this case, we need to wrap it
                    schema = {"anyOf": [schema, {"type": "null"}]}

            # Add docstring description if available and not already present
            if (
                field_name
                and self.config.include_descriptions
                and "description" not in schema
            ):
                if field_name in self._param_descriptions:
                    schema["description"] = self._param_descriptions[field_name]

            return schema

        # Multiple non-None types - use anyOf
        schemas: list[dict[str, Any]] = [
            self._get_type_schema(arg, field_name) for arg in non_none_types
        ]

        result: dict[str, Any] = {"anyOf": schemas}

        if has_none:
            result["anyOf"].append({"type": "null"})

        if field_name and self.config.include_descriptions:
            result["description"] = self._param_descriptions.get(
                field_name, f"The {field_name} parameter"
            )

        return result

    def _handle_literal(
        self, args: tuple[type | int | float | str | bool, ...], field_name: str
    ) -> Dict[str, Any]:
        """Handle Literal types as enums"""

        # Determine the type
        if all(isinstance(arg, str) for arg in args):
            type_name = "string"
        elif all(isinstance(arg, int) for arg in args):
            type_name = "integer"
        elif all(isinstance(arg, (int, float)) for arg in args):
            type_name = "number"
        else:
            type_name = "string"
            args = tuple(str(arg) for arg in args)

        schema = {"type": type_name, "enum": list(args)}

        if field_name and self.config.include_descriptions:
            schema["description"] = self._param_descriptions.get(
                field_name, f"The {field_name} parameter"
            )

        return schema

    def _handle_list(self, args: tuple[type, ...], field_name: str) -> Dict[str, Any]:
        """Handle List types"""

        schema: dict[str, Any] = {"type": "array"}

        if args:
            # Resolve forward references in list item types
            item_type = args[0]

            # The item type might be a string forward reference
            if isinstance(item_type, (ForwardRef, str)):
                resolved = self._resolve_forward_ref(item_type)
                if resolved is not None:
                    item_type = resolved

            if item_type:
                item_schema = self._get_type_schema(item_type, f"{field_name}_item")
                schema["items"] = item_schema
            else:
                schema["items"] = {}
        else:
            schema["items"] = {}

        if field_name and self.config.include_descriptions:
            schema["description"] = self._param_descriptions.get(
                field_name, f"Array of {field_name}"
            )

        return schema

    def _handle_dict(self, args: tuple[type, ...], field_name: str) -> Dict[str, Any]:
        """Handle Dict types"""

        schema: dict[str, Any] = {"type": "object"}

        if len(args) >= 2:
            # Dict[str, ValueType]
            value_schema = self._get_type_schema(args[1], f"{field_name}_value")
            schema["additionalProperties"] = value_schema
        else:
            schema["additionalProperties"] = True

        if field_name and self.config.include_descriptions:
            schema["description"] = self._param_descriptions.get(
                field_name, f"Dictionary for {field_name}"
            )

        return schema

    def _handle_tuple(
        self, args: tuple[Any, ...] | EllipsisType, field_name: str
    ) -> Dict[str, Any]:
        """Handle Tuple types - represented as arrays with specific item schemas"""

        if not args:
            return {
                "type": "array",
                "description": f"Tuple for {field_name}" if field_name else "Tuple",
            }

        # For fixed-length tuples, we'll use array with items as array of schemas
        # Note: JSON Schema prefixItems is more precise but may not be supported
        # Using a simpler array representation
        item_schemas: list[Dict[str, Any]] = []
        if isinstance(args, tuple):
            for i, arg in enumerate(args):
                if arg is Ellipsis:
                    continue
                item_schemas.append(self._get_type_schema(arg, f"{field_name}_{i}"))

        schema: dict[str, Any] = {
            "type": "array",
            "items": item_schemas[0]
            if len(item_schemas) == 1
            else {"anyOf": item_schemas},
        }

        if field_name and self.config.include_descriptions:
            schema["description"] = self._param_descriptions.get(
                field_name, f"Tuple for {field_name}"
            )

        return schema

    def extract_type(self, type_cls: type) -> Dict[str, Any]:
        """Public helper to extract schema directly from a Python type.

        This avoids accessing protected internals from callers and mirrors the
        behavior used when extracting from return annotations.
        """
        # Reset state
        self._definitions = {}
        self._seen_types = set()
        self._current_depth = 0
        self._type_schemas = {}
        self._building_types = {}
        self._param_descriptions = {}

        schema = self._get_type_schema(type_cls, getattr(type_cls, "__name__", ""))

        # If it's a reference, unwrap it
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            if ref_name in self._definitions:
                schema = self._definitions[ref_name].copy()

        # Add definitions if any (unless dereferencing)
        if self._definitions and not self.config.dereference:
            schema["$defs"] = self._definitions

        result: Dict[str, Any] = {
            "name": getattr(type_cls, "__name__", "type").lower(),
            "strict": True if self.config.strict_mode else False,
            "schema": schema,
        }

        if getattr(type_cls, "__doc__", None):
            result["description"] = cast(str, type_cls.__doc__).strip()

        return result

    def _handle_enum(self, enum_class: type[Enum], field_name: str) -> Dict[str, Any]:
        """Handle Enum types"""

        values: list[Any] = [e.value for e in enum_class]

        # Validate enum size
        if len(values) > self.config.max_enum_values:
            raise ValueError(
                f"Enum {enum_class.__name__} has {len(values)} values, "
                + f"exceeding maximum of {self.config.max_enum_values}"
            )

        # Determine type
        if all(isinstance(v, str) for v in values):
            type_name = "string"
        elif all(isinstance(v, int) for v in values):
            type_name = "integer"
        elif all(isinstance(v, (int, float)) for v in values):
            type_name = "number"
        else:
            type_name = "string"
            values = [str(v) for v in values]

        schema = {"type": type_name, "enum": values}

        if field_name and self.config.include_descriptions:
            desc = enum_class.__doc__ or f"Enum for {field_name}"
            schema["description"] = desc.strip()

        return schema

    def _handle_dataclass(
        self, dataclass_type: type, field_name: str
    ) -> Dict[str, Any]:
        """Handle dataclass types"""

        # If dereferencing, check if we've already completed processing this type
        if self.config.dereference and dataclass_type in self._type_schemas:
            # Return a deep copy to avoid mutations
            return self._deep_copy_schema(self._type_schemas[dataclass_type])

        # Check for circular reference - this is the CRITICAL check
        if dataclass_type in self._seen_types:
            if self.config.dereference:
                # We're in a circular reference during dereferencing
                # For circular refs, we MUST use $ref even in dereference mode
                # because JSON cannot represent true circular structures
                def_name = dataclass_type.__name__

                # Make sure it's in definitions
                if (
                    def_name not in self._definitions
                    and dataclass_type in self._building_types
                ):
                    # Store the schema being built
                    self._definitions[def_name] = self._building_types[dataclass_type]

                return {"$ref": f"#/$defs/{def_name}"}
            # Use reference (non-dereferenced mode)
            def_name = dataclass_type.__name__
            return {"$ref": f"#/$defs/{def_name}"}

        # Check depth
        self._current_depth += 1
        if self._current_depth > self.config.max_nesting_depth:
            self._current_depth -= 1
            # Don't raise error, just return a placeholder for deep nesting
            return {
                "type": "object",
                "description": f"Object of type {dataclass_type.__name__} (max depth reached)",
            }

        self._seen_types.add(dataclass_type)

        try:
            # Create the schema object that will be filled in
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            # If dereferencing, register this schema as being built (for circular refs)
            if self.config.dereference:
                self._building_types[dataclass_type] = schema

            properties = {}
            required = []

            # DON'T use get_type_hints() as it causes recursion with forward refs
            # Instead, directly process field.type
            for field in fields(dataclass_type):
                # Use field.type directly - it may contain forward references as strings
                field_type = field.type

                field_schema = self._get_type_schema(
                    field_type,
                    field.name,
                    field.default
                    if field.default is not dataclass_type
                    else inspect.Parameter.empty,
                )
                properties[field.name] = field_schema

                # Check if required
                has_default = (
                    field.default is not dataclass_type
                    and field.default_factory is not dataclass_type
                )

                if self.config.make_all_required or not has_default:
                    required.append(field.name)

            # Update the schema with actual properties
            schema["properties"] = properties
            schema["required"] = required

            if self.config.ensure_additional_properties:
                schema["additionalProperties"] = False

            # Add description
            if dataclass_type.__doc__ and self.config.include_descriptions:
                schema["description"] = dataclass_type.__doc__.strip()

            if self.config.dereference:
                # Store the complete schema for this type
                self._type_schemas[dataclass_type] = schema
                # Also store in definitions for circular refs
                def_name = dataclass_type.__name__
                self._definitions[def_name] = schema
                # Remove from building set
                if dataclass_type in self._building_types:
                    del self._building_types[dataclass_type]
                return schema
            else:
                # Store in definitions and return reference
                def_name = dataclass_type.__name__
                self._definitions[def_name] = schema
                return {"$ref": f"#/$defs/{def_name}"}

        finally:
            self._current_depth -= 1
            self._seen_types.discard(dataclass_type)

    def _deep_copy_schema(self, schema: Any) -> Any:
        """
        Deep copy a schema, but preserve object identity for circular references.
        """
        if isinstance(schema, dict):
            # Create new dict with copied values
            return {k: self._deep_copy_schema(v) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._deep_copy_schema(item) for item in schema]
        else:
            return schema

    def _handle_class(self, class_type: type, field_name: str) -> Dict[str, Any]:
        """Handle regular classes by inspecting __init__ or annotations"""

        # Try to get type hints from __init__
        try:
            init_method = class_type.__init__
            sig = inspect.signature(init_method)
            hints = typing.get_type_hints(init_method)

            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = hints.get(param_name, inspect.Parameter.empty)
                param_schema = self._get_type_schema(
                    param_type, param_name, param.default
                )

                properties[param_name] = param_schema

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            if properties:
                schema: dict[str, Any] = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }

                if self.config.ensure_additional_properties:
                    schema["additionalProperties"] = False

                if class_type.__doc__ and self.config.include_descriptions:
                    schema["description"] = class_type.__doc__.strip()

                return schema

        except Exception:
            pass

        # Fallback to generic object
        return {
            "type": "object",
            "description": f"Object of type {class_type.__name__}",
        }

    def _dereference_schema(
        self, schema: Any, visiting: Optional[Set[str]] = None
    ) -> Any:
        """
        Recursively replace all $ref with their actual definitions.
        Preserves $ref for circular references to avoid infinite recursion.

        Args:
            schema: The schema to dereference
            visiting: Set of definition names currently being visited (for cycle detection)

        Returns:
            The dereferenced schema
        """
        if visiting is None:
            visiting = set()

        if isinstance(schema, dict):
            # Handle $ref
            if "$ref" in schema:
                ref_path: str = cast(str, schema["$ref"])
                # Extract definition name from path like "#/$defs/Person"
                if ref_path.startswith("#/$defs/"):
                    def_name: str = ref_path.split("/")[-1]

                    # Check if we're already visiting this definition (circular reference)
                    if def_name in visiting:
                        # Keep the $ref to avoid infinite recursion
                        return schema

                    if def_name in self._definitions:
                        # Mark as visiting
                        visiting.add(def_name)
                        try:
                            # Recursively dereference the definition itself
                            result = self._dereference_schema(
                                self._definitions[def_name].copy(), visiting
                            )
                            return result
                        finally:
                            # Unmark after processing
                            visiting.discard(def_name)
                return schema

            # Recursively dereference all values
            return {
                k: self._dereference_schema(v, visiting)
                for k, v in schema.items()
                if k != "$defs"
            }

        elif isinstance(schema, list):
            return [self._dereference_schema(item, visiting) for item in schema]

        else:
            return schema


# Test cases
if __name__ == "__main__":
    from dataclasses import dataclass
    from enum import Enum
    import json

    print("=" * 80)
    print("JSON SCHEMA EXTRACTOR TESTS")
    print("=" * 80)

    # Test 1: Simple function with primitive types
    print("\n\n1. Simple function with primitives:")
    print("-" * 80)

    def get_weather(location: str, unit: Literal["F", "C"] = "F") -> dict[str, Any]:
        """Fetches the weather in the given location"""
        return {}

    extractor = JsonSchemaExtractor()
    schema = extractor.extract(get_weather)
    print(json.dumps(schema, indent=2))

    # Test 2: Function with complex types
    print("\n\n2. Function with lists and optional parameters:")
    print("-" * 80)

    def create_user(
        name: str,
        age: int,
        email: str,
        tags: List[str],
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Create a new user with the given information"""
        pass

    schema = extractor.extract(create_user)
    print(json.dumps(schema, indent=2))

    # Test 3: Function with Enum
    print("\n\n3. Function with Enum:")
    print("-" * 80)

    class Status(Enum):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"

    def update_status(user_id: int, status: Status, reason: Optional[str] = None):
        """Update the status of a user"""
        pass

    schema = extractor.extract(update_status)
    print(json.dumps(schema, indent=2))

    # Test 4: Function with dataclass
    print("\n\n4. Function with dataclass (with $ref):")
    print("-" * 80)

    @dataclass
    class Address:
        """Address information"""

        street: str
        city: str
        country: str
        postal_code: Optional[str] = None

    @dataclass
    class Person:
        """Person information"""

        name: str
        age: int
        address: Address
        emails: List[str]

    def register_person(person: Person, send_confirmation: bool = True):
        """Register a new person in the system"""
        pass

    schema = extractor.extract(register_person)
    print(json.dumps(schema, indent=2))

    # Test 4b: Same but dereferenced
    print("\n\n4b. Function with dataclass (dereferenced - no $ref):")
    print("-" * 80)

    config_deref = JsonSchemaConfig(dereference=True)
    extractor_deref = JsonSchemaExtractor(config_deref)
    schema = extractor_deref.extract(register_person)
    print(json.dumps(schema, indent=2))

    # Test 5: Function with Union types
    print("\n\n5. Function with Union types:")
    print("-" * 80)

    def process_data(
        data: Union[str, int, List[str]], mode: Literal["fast", "accurate"] = "fast"
    ):
        """Process data in different formats"""
        pass

    schema = extractor.extract(process_data)
    print(json.dumps(schema, indent=2))

    # Test 6: Math reasoning example (from docs)
    print("\n\n6. Math reasoning (chain of thought) - Parameters:")
    print("-" * 80)

    @dataclass
    class Step:
        """A single step in the solution"""

        explanation: str
        output: str

    @dataclass
    class MathReasoning:
        """Step-by-step math solution"""

        steps: List[Step]
        final_answer: str

    def solve_math_problem(problem: str) -> MathReasoning:
        """Solve a math problem step by step"""
        return MathReasoning(
            steps=[],
            final_answer="",
        )

    schema = extractor.extract(solve_math_problem)
    print(json.dumps(schema, indent=2))

    print("\n\n6b. Math reasoning - Return Type (for response format):")
    print("-" * 80)
    schema = extractor.extract(
        solve_math_problem, function_extraction_config={"extract_return_type": True}
    )
    print(json.dumps(schema, indent=2))

    # Test 7: Recursive schema (UI components)
    print("\n\n7. Recursive schema (simplified UI):")
    print("-" * 80)

    @dataclass
    class UIAttribute:
        """UI component attribute"""

        name: str
        value: str

    @dataclass
    class UIComponent:
        """Recursive UI component"""

        type: Literal["div", "button", "header", "section", "field", "form"]
        label: str
        attributes: List[UIAttribute]
        # Note: For true recursion, you'd need forward references
        # children: List['UIComponent']

    def generate_ui(description: str) -> UIComponent:
        """Generate UI components from description"""
        return UIComponent(
            type="div",
            label="",
            attributes=[],
        )

    schema = extractor.extract(generate_ui)
    print(json.dumps(schema, indent=2))

    # Test 8: Direct type extraction (for Pydantic-style models)
    print("\n\n8. Direct type extraction from dataclass:")
    print("-" * 80)

    @dataclass
    class CalendarEvent:
        """A calendar event"""

        name: str
        date: str
        participants: List[str]

    # You can also extract directly from a type
    def extract_from_type(type_cls: type) -> Dict[str, Any]:
        """Helper to extract schema directly from a type"""
        ext = JsonSchemaExtractor()
        return ext.extract_type(type_cls)

    schema = extract_from_type(CalendarEvent)
    print(json.dumps(schema, indent=2))

    # Test 9: Custom configuration
    print("\n\n9. Custom configuration (no descriptions, non-strict):")
    print("-" * 80)

    config = JsonSchemaConfig(
        include_descriptions=False,
        strict_mode=False,
        ensure_additional_properties=False,
    )
    extractor_custom = JsonSchemaExtractor(config)

    def simple_func(x: int, y: str):
        pass

    schema = extractor_custom.extract(simple_func)
    print(json.dumps(schema, indent=2))

    # Test 10: Dereferenced return type
    print("\n\n10. Dereferenced return type (completely inlined):")
    print("-" * 80)

    @dataclass
    class StepDetailed:
        """A detailed step"""

        explanation: str
        output: str
        confidence: float

    @dataclass
    class DetailedMathReasoning:
        """Detailed step-by-step solution"""

        steps: List[StepDetailed]
        final_answer: str
        total_confidence: float

    def solve_detailed(problem: str) -> DetailedMathReasoning:
        """Solve with detailed reasoning"""
        return DetailedMathReasoning(
            steps=[],
            final_answer="",
            total_confidence=0.0,
        )

    config_deref = JsonSchemaConfig(dereference=True)
    extractor_deref = JsonSchemaExtractor(config_deref)
    schema = extractor_deref.extract(
        solve_detailed, function_extraction_config={"extract_return_type": True}
    )
    print(json.dumps(schema, indent=2))

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

    # Test 11: Circular/Recursive schemas with dereferencing
    print("\n\n11. CIRCULAR REFERENCE TEST - Linked List (dereferenced):")
    print("    Note: Circular refs must use $ref even in dereference mode")
    print("-" * 80)

    @dataclass
    class LinkedListNode:
        """A node in a linked list"""

        value: int
        next: Optional["LinkedListNode"] = None

    def create_linked_list(head: LinkedListNode):
        """Create a linked list"""
        pass

    try:
        config_deref = JsonSchemaConfig(dereference=True)
        extractor_deref = JsonSchemaExtractor(config_deref)
        schema = extractor_deref.extract(create_linked_list)
        print("✓ Successfully handled circular reference!")
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"✗ Failed: {e}")
        # import traceback
        # traceback.print_exc()

    # Test 12: Tree structure (circular)
    print("\n\n12. CIRCULAR REFERENCE TEST - Tree (dereferenced):")
    print("-" * 80)

    @dataclass
    class TreeNode:
        """A node in a tree"""

        value: str
        children: List["TreeNode"]

    def build_tree(root: TreeNode):
        """Build a tree structure"""
        pass

    try:
        config_deref = JsonSchemaConfig(dereference=True)
        extractor_deref = JsonSchemaExtractor(config_deref)
        schema = extractor_deref.extract(build_tree)
        print("✓ Successfully handled circular reference!")
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 13: Mutual recursion
    print("\n\n13. MUTUAL RECURSION TEST (dereferenced):")
    print("-" * 80)

    @dataclass
    class Author:
        """An author"""

        name: str
        books: List["Book"]

    @dataclass
    class Book:
        """A book"""

        title: str
        author: Author

    def register_author(author: Author):
        """Register an author"""
        pass

    try:
        config_deref = JsonSchemaConfig(dereference=True)
        extractor_deref = JsonSchemaExtractor(config_deref)
        schema = extractor_deref.extract(register_author)
        print("✓ Successfully handled mutual recursion!")
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"✗ Failed: {e}")

    # NEW TESTS FOR PYDANTIC BASEMODEL
    print("\n\n" + "=" * 80)
    print("PYDANTIC BASEMODEL TESTS")
    print("=" * 80)

    # Test 14: Simple Pydantic model
    print("\n\n14. Simple Pydantic BaseModel:")
    print("-" * 80)

    class UserModel(BaseModel):
        """A user model"""

        name: str
        age: int
        email: str

    extractor = JsonSchemaExtractor()
    schema = extractor.extract(UserModel)
    print(json.dumps(schema, indent=2))

    # Test 15: Pydantic model with optional fields
    print("\n\n15. Pydantic model with optional fields:")
    print("-" * 80)

    class ProductModel(BaseModel):
        """A product model"""

        name: str
        price: float
        description: Optional[str] = None
        tags: List[str] = []

    schema = extractor.extract(ProductModel)
    print(json.dumps(schema, indent=2))

    # Test 16: Nested Pydantic models
    print("\n\n16. Nested Pydantic models:")
    print("-" * 80)

    class AddressModel(BaseModel):
        """Address information"""

        street: str
        city: str
        country: str

    class PersonModel(BaseModel):
        """Person information"""

        name: str
        age: int
        address: AddressModel
        emails: List[str]

    schema = extractor.extract(PersonModel)
    print(json.dumps(schema, indent=2))

    # Test 17: Dereferenced Pydantic model
    print("\n\n17. Dereferenced Pydantic model:")
    print("-" * 80)

    config_deref = JsonSchemaConfig(dereference=True)
    extractor_deref = JsonSchemaExtractor(config_deref)
    schema = extractor_deref.extract(PersonModel)
    print(json.dumps(schema, indent=2))

    # Test 18: Pydantic model with Literal and Enum
    print("\n\n18. Pydantic model with Literal and Enum:")
    print("-" * 80)

    class Priority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class TaskModel(BaseModel):
        """A task model"""

        title: str
        priority: Priority
        status: Literal["pending", "in_progress", "done"]

    schema = extractor.extract(TaskModel)
    print(json.dumps(schema, indent=2))

    # Test 19: Pydantic with custom config (no descriptions)
    print("\n\n19. Pydantic with custom config (no descriptions):")
    print("-" * 80)

    config_no_desc = JsonSchemaConfig(include_descriptions=False)
    extractor_no_desc = JsonSchemaExtractor(config_no_desc)
    schema = extractor_no_desc.extract(UserModel)
    print(json.dumps(schema, indent=2))

    # Test 20: Pydantic with make_all_required=False
    print("\n\n20. Pydantic with make_all_required=False:")
    print("-" * 80)

    class FlexibleModel(BaseModel):
        """A flexible model"""

        required_field: str
        optional_field: Optional[str] = None

    config_flexible = JsonSchemaConfig(make_all_required=False)
    extractor_flexible = JsonSchemaExtractor(config_flexible)
    schema = extractor_flexible.extract(FlexibleModel)
    print(json.dumps(schema, indent=2))

    print("\n\n" + "=" * 80)
    print("ALL PYDANTIC TESTS COMPLETED")
    print("=" * 80)
