from enum import StrEnum


class ObjectSerializationStyle(StrEnum):
    """Different ways to serialize objects for different parameter locations."""

    JSON_STRING = "json_string"  # {"key": "value"} -> '{"key":"value"}'
    FORM_STYLE = "form_style"  # {"key": "value"} -> "key=value"
    DOT_NOTATION = "dot_notation"  # {"user": {"name": "John"}} -> "user.name=John"
    BRACKET_NOTATION = (
        "bracket_notation"  # {"items": ["a", "b"]} -> "items[0]=a&items[1]=b"
    )
