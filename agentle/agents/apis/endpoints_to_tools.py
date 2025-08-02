# Utility function for easy tool creation from endpoints
from collections.abc import MutableMapping, MutableSequence, Sequence
from typing import Any
from agentle.agents.apis.api import API
from agentle.agents.apis.endpoint import Endpoint
from agentle.generations.tools.tool import Tool


def endpoints_to_tools(
    endpoints: Sequence[Endpoint | API],
    base_url: str | None = None,
    global_headers: MutableMapping[str, str] | None = None,
) -> Sequence[Tool[Any]]:
    """
    Convert a list of endpoints and APIs to tools.

    Args:
        endpoints: List of Endpoint and/or API instances
        **kwargs: Additional arguments passed to endpoint.to_tool()

    Returns:
        List of Tool instances
    """
    tools: MutableSequence[Tool[Any]] = []

    for item in endpoints:
        if isinstance(item, Endpoint):
            tools.append(item.to_tool(base_url=base_url, global_headers=global_headers))
            continue

        tools.extend(item.to_tools())

    return tools
