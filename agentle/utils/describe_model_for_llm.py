# type: ignore
from typing import (
    Type,
    Optional,
    Union,
    Any,
    get_origin,
    get_args,
    List,
    Dict,
    Set,
    FrozenSet,
    Tuple,
)
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
import inspect


def describe_model_for_llm(
    model_class: Type[BaseModel],
    include_examples: bool = True,
    max_depth: int = 3,
    inline_nested: bool = True,
    _current_depth: int = 0,
) -> str:
    """
    Generate an LLM-friendly description of a Pydantic BaseModel structure.

    Args:
        model_class: The Pydantic BaseModel class to describe
        include_examples: Whether to include example values in the description
        max_depth: Maximum recursion depth for nested models (prevents infinite recursion)
        inline_nested: If True, describe nested objects inline rather than separately
        _current_depth: Internal parameter for tracking recursion depth

    Returns:
        A formatted string description suitable for LLM prompts

    Example:
        class User(BaseModel):
            name: str = Field(description="User's full name")
            age: int = Field(ge=0, le=150)

        description = describe_model_for_llm(User, inline_nested=True)
        # Returns a detailed description with nested objects described inline
    """

    def format_type_description(
        field_type: Type, depth: int = 0, for_inline: bool = False
    ) -> str:
        """Format a type annotation into a human-readable description."""
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Union types (including Optional)
        if origin is Union:
            if type(None) in args:
                # Optional type
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    inner_desc = format_type_description(
                        non_none_args[0], depth, for_inline
                    )
                    return f"optional {inner_desc}"
                else:
                    # Multiple non-None types
                    type_descriptions = [
                        format_type_description(arg, depth, for_inline)
                        for arg in non_none_args
                    ]
                    return f"optional union of ({' | '.join(type_descriptions)})"
            else:
                # Regular Union
                type_descriptions = [
                    format_type_description(arg, depth, for_inline) for arg in args
                ]
                return f"union of ({' | '.join(type_descriptions)})"

        # Handle List types
        elif origin in (list, List):
            inner_type = args[0] if args else Any
            inner_desc = format_type_description(inner_type, depth, for_inline)
            return f"array of {inner_desc}"

        # Handle Dict types
        elif origin in (dict, Dict):
            if len(args) >= 2:
                key_desc = format_type_description(args[0], depth, for_inline)
                value_desc = format_type_description(args[1], depth, for_inline)
                return f"object with {key_desc} keys and {value_desc} values"
            else:
                return "object"

        # Handle Set types
        elif origin in (set, Set):
            inner_type = args[0] if args else Any
            inner_desc = format_type_description(inner_type, depth, for_inline)
            return f"array of unique {inner_desc}"

        # Handle FrozenSet types
        elif origin in (frozenset, FrozenSet):
            inner_type = args[0] if args else Any
            inner_desc = format_type_description(inner_type, depth, for_inline)
            return f"array of unique {inner_desc}"

        # Handle Tuple types
        elif origin in (tuple, Tuple):
            if args:
                if len(args) == 2 and args[1] is Ellipsis:
                    # Variable length tuple
                    inner_desc = format_type_description(args[0], depth, for_inline)
                    return f"array of {inner_desc}"
                else:
                    # Fixed length tuple
                    type_descriptions = [
                        format_type_description(arg, depth, for_inline) for arg in args
                    ]
                    return f"array with exactly [{', '.join(type_descriptions)}]"
            else:
                return "array"

        # Handle nested BaseModel classes
        elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            if for_inline and depth < max_depth:
                return f"object"
            else:
                return f"{field_type.__name__} object"

        # Handle basic types
        elif field_type == str:
            return "string"
        elif field_type == int:
            return "integer"
        elif field_type == float:
            return "number"
        elif field_type == bool:
            return "boolean"
        elif field_type == Any:
            return "any type"
        else:
            # For other types, use the type name
            type_name = getattr(field_type, "__name__", str(field_type))
            return type_name.lower()

    def format_constraints(field_info: FieldInfo) -> List[str]:
        """Extract and format field constraints."""
        constraints = []

        # Handle common constraints from field_info
        if hasattr(field_info, "constraints") and field_info.constraints:
            for constraint, value in field_info.constraints.items():
                if constraint == "min_length" and value is not None:
                    constraints.append(f"minimum length: {value}")
                elif constraint == "max_length" and value is not None:
                    constraints.append(f"maximum length: {value}")
                elif constraint == "ge" and value is not None:
                    constraints.append(f"greater than or equal to: {value}")
                elif constraint == "gt" and value is not None:
                    constraints.append(f"greater than: {value}")
                elif constraint == "le" and value is not None:
                    constraints.append(f"less than or equal to: {value}")
                elif constraint == "lt" and value is not None:
                    constraints.append(f"less than: {value}")
                elif constraint == "regex" and value is not None:
                    constraints.append(f"must match pattern: {value}")

        # Check for individual constraint attributes (Pydantic v2 style)
        constraint_attrs = [
            ("min_length", "minimum length"),
            ("max_length", "maximum length"),
            ("ge", "greater than or equal to"),
            ("gt", "greater than"),
            ("le", "less than or equal to"),
            ("lt", "less than"),
            ("pattern", "must match pattern"),
            ("regex", "must match pattern"),
        ]

        for attr, description in constraint_attrs:
            if hasattr(field_info, attr):
                value = getattr(field_info, attr)
                if value is not None:
                    constraints.append(f"{description}: {value}")

        return constraints

    def generate_example_value(
        field_type: Type, field_info: FieldInfo, depth: int = 0
    ) -> str:
        """Generate a realistic example value for a field type."""
        origin = get_origin(field_type)

        # Check if there's a default value
        if field_info.default is not None and field_info.default != ...:
            if isinstance(field_info.default, str):
                return f'"{field_info.default}"'
            elif isinstance(field_info.default, bool):
                return "true" if field_info.default else "false"
            else:
                return str(field_info.default)

        # Handle Union/Optional types
        if origin is Union:
            args = get_args(field_type)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                return generate_example_value(non_none_args[0], FieldInfo(), depth)

        # Generate examples based on type
        if field_type == str or origin is str:
            return '"example_string"'
        elif field_type == int or origin is int:
            return "42"
        elif field_type == float or origin is float:
            return "3.14"
        elif field_type == bool or origin is bool:
            return "true"
        elif origin in (list, List):
            inner_type = get_args(field_type)[0] if get_args(field_type) else str
            inner_example = generate_example_value(inner_type, FieldInfo(), depth + 1)
            return f"[{inner_example}]"
        elif origin in (dict, Dict):
            args = get_args(field_type)
            if len(args) >= 2:
                key_type, value_type = args[0], args[1]
                if key_type == str:
                    value_example = generate_example_value(
                        value_type, FieldInfo(), depth + 1
                    )
                    return f'{{"key": {value_example}}}'
            return '{"key": "value"}'
        elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            if depth < 2:  # Limit nesting in examples
                # Generate a compact example of the nested object
                nested_fields = []
                for name, info in field_type.model_fields.items():
                    example_val = generate_example_value(
                        info.annotation, info, depth + 1
                    )
                    nested_fields.append(f'"{name}": {example_val}')
                return (
                    "{"
                    + ", ".join(nested_fields[:3])
                    + ("..." if len(field_type.model_fields) > 3 else "")
                    + "}"
                )
            else:
                return f"{{/* {field_type.__name__} object */}}"
        else:
            return '"example_value"'

    # Start building the description
    lines = []
    indent = "  " * _current_depth

    # Model header
    if _current_depth == 0:
        lines.append(f"# {model_class.__name__} Model Structure")
        lines.append("")
        if model_class.__doc__:
            lines.append(f"Model Description: {model_class.__doc__.strip()}")
            lines.append("")
    else:
        lines.append(f"{indent}## {model_class.__name__} (nested object)")
        lines.append("")

    # Check if we've hit max depth
    if _current_depth >= max_depth:
        lines.append(
            f"{indent}(Maximum nesting depth reached - see separate {model_class.__name__} definition)"
        )
        return "\n".join(lines)

    # Field descriptions
    lines.append(f"{indent}**Fields:**")
    lines.append("")

    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation

        # Check if this field is a nested BaseModel
        nested_model = None
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            nested_model = field_type
        elif get_origin(field_type) is Union:
            # Check if it's Optional[BaseModel]
            args = get_args(field_type)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if (
                len(non_none_args) == 1
                and inspect.isclass(non_none_args[0])
                and issubclass(non_none_args[0], BaseModel)
            ):
                nested_model = non_none_args[0]

        # Build field description
        field_lines = []

        if nested_model and inline_nested and _current_depth < max_depth:
            # Inline nested object description
            is_optional = get_origin(field_type) is Union and type(None) in get_args(
                field_type
            )
            optional_text = "optional " if is_optional else ""
            field_lines.append(
                f"{indent}- **{field_name}** ({optional_text}object with the following structure):"
            )

            if field_info.description:
                field_lines.append(f"{indent}  - Description: {field_info.description}")

            # Add nested object fields inline
            for (
                nested_field_name,
                nested_field_info,
            ) in nested_model.model_fields.items():
                nested_type_desc = format_type_description(
                    nested_field_info.annotation, _current_depth + 1, True
                )
                nested_required = (
                    nested_field_info.default is ...
                    and nested_field_info.default_factory is None
                )
                required_text = " (required)" if nested_required else ""

                nested_desc = f"{indent}    - **{nested_field_name}** ({nested_type_desc}){required_text}"
                if nested_field_info.description:
                    nested_desc += f": {nested_field_info.description}"

                # Add constraints for nested fields
                nested_constraints = format_constraints(nested_field_info)
                if nested_constraints:
                    nested_desc += f" [{', '.join(nested_constraints)}]"

                field_lines.append(nested_desc)

            # Add overall field info
            is_required = (
                field_info.default is ... and field_info.default_factory is None
            )
            field_lines.append(
                f"{indent}  - Required: {'Yes' if is_required else 'No'}"
            )

            # Add example with full nested structure
            if include_examples:
                example = generate_example_value(field_type, field_info, _current_depth)
                field_lines.append(f"{indent}  - Example: {example}")

        else:
            # Regular field description
            type_desc = format_type_description(field_type, _current_depth + 1, False)
            field_lines.append(f"{indent}- **{field_name}** ({type_desc})")

            # Add description if available
            if field_info.description:
                field_lines.append(f"{indent}  - Description: {field_info.description}")

            # Add constraints
            constraints = format_constraints(field_info)
            if constraints:
                field_lines.append(f"{indent}  - Constraints: {', '.join(constraints)}")

            # Add required/optional status
            is_required = (
                field_info.default is ... and field_info.default_factory is None
            )
            field_lines.append(
                f"{indent}  - Required: {'Yes' if is_required else 'No'}"
            )

            # Add default value if not required
            if not is_required:
                if field_info.default is not None and field_info.default != ...:
                    default_val = (
                        f'"{field_info.default}"'
                        if isinstance(field_info.default, str)
                        else str(field_info.default)
                    )
                    field_lines.append(f"{indent}  - Default: {default_val}")
                elif field_info.default_factory is not None:
                    field_lines.append(
                        f"{indent}  - Default: Generated by factory function"
                    )
                else:
                    field_lines.append(f"{indent}  - Default: None")

            # Add example if requested
            if include_examples:
                example = generate_example_value(field_type, field_info, _current_depth)
                field_lines.append(f"{indent}  - Example: {example}")

        lines.extend(field_lines)
        lines.append("")

    # Handle nested models recursively (only if not inlined)
    if not inline_nested:
        nested_models = []
        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation

            # Check for nested BaseModel in the type
            def find_nested_models(t):
                origin = get_origin(t)
                args = get_args(t)

                if inspect.isclass(t) and issubclass(t, BaseModel):
                    return [t]
                elif origin is Union:
                    models = []
                    for arg in args:
                        models.extend(find_nested_models(arg))
                    return models
                elif origin in (list, List, set, Set, frozenset, FrozenSet):
                    if args:
                        return find_nested_models(args[0])
                elif origin in (dict, Dict):
                    if len(args) >= 2:
                        return find_nested_models(args[1])
                elif origin in (tuple, Tuple):
                    models = []
                    for arg in args:
                        if arg is not Ellipsis:
                            models.extend(find_nested_models(arg))
                    return models
                return []

            nested_models.extend(find_nested_models(field_type))

        # Add nested model descriptions
        if nested_models and _current_depth < max_depth:
            lines.append(f"{indent}**Nested Model Definitions:**")
            lines.append("")

            # Remove duplicates while preserving order
            seen = set()
            unique_nested = []
            for model in nested_models:
                if model not in seen:
                    seen.add(model)
                    unique_nested.append(model)

            for nested_model in unique_nested:
                nested_desc = describe_model_for_llm(
                    nested_model,
                    include_examples=include_examples,
                    max_depth=max_depth,
                    inline_nested=inline_nested,
                    _current_depth=_current_depth + 1,
                )
                lines.append(nested_desc)
                lines.append("")

    return "\n".join(lines)


# Example usage and test cases
if __name__ == "__main__":
    from pydantic import Field
    from typing import Optional
    from enum import Enum

    class UserRole(str, Enum):
        ADMIN = "admin"
        USER = "user"
        GUEST = "guest"

    class Address(BaseModel):
        """Physical address information"""

        street: str = Field(description="Street address", min_length=1, max_length=200)
        city: str = Field(description="City name", min_length=1)
        postal_code: str = Field(
            description="Postal/ZIP code", pattern=r"^\d{5}(-\d{4})?$"
        )
        country: str = Field(default="USA", description="Country code")

    class ContactInfo(BaseModel):
        """Contact information for a user"""

        email: str = Field(description="Email address", pattern=r"^[^@]+@[^@]+\.[^@]+$")
        phone: Optional[str] = Field(
            None, description="Phone number", pattern=r"^\+?[\d\s\-\(\)]+$"
        )

    class User(BaseModel):
        """User account information"""

        id: int = Field(description="Unique user identifier", ge=1)
        name: str = Field(
            description="Full name of the user", min_length=1, max_length=100
        )
        age: Optional[int] = Field(None, description="User's age", ge=0, le=150)
        role: UserRole = Field(
            default=UserRole.USER, description="User's role in the system"
        )
        address: Address = Field(description="User's primary address")
        contact: ContactInfo = Field(description="Contact information")
        tags: List[str] = Field(
            default_factory=list, description="User tags for categorization"
        )
        metadata: Dict[str, Any] = Field(
            default_factory=dict, description="Additional user metadata"
        )
        is_active: bool = Field(
            default=True, description="Whether the user account is active"
        )

    # Generate description with inline nested objects (recommended for LLMs)
    description_inline = describe_model_for_llm(
        User, include_examples=True, inline_nested=True
    )
    print("=== INLINE NESTED OBJECTS (RECOMMENDED) ===")
    print(description_inline)

    print("\n" + "=" * 80 + "\n")

    # Generate description with separate nested object definitions
    description_separate = describe_model_for_llm(
        User, include_examples=True, inline_nested=False
    )
    print("=== SEPARATE NESTED DEFINITIONS ===")
    print(description_separate)

    print("\n" + "=" * 80 + "\n")

    # Example of using the description in an LLM prompt
    prompt = f"""
Please generate a JSON object that matches this exact structure:

{description_inline}

Generate a realistic example with a software engineer named John Doe from Boston.
Make sure the JSON is valid and includes all required fields.
"""

    print("Example LLM Prompt:")
    print(prompt)
