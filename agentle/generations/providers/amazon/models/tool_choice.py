from collections.abc import Mapping
from typing import TypedDict, NotRequired, Any


class ToolChoice(TypedDict):
    auto: NotRequired[Mapping[str, Any]]  # Empty dict for auto mode
    any: NotRequired[Mapping[str, Any]]  # Empty dict for any tool mode
    tool: NotRequired[Any]  # For specific tool selection
