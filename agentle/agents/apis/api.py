from collections.abc import MutableMapping, Sequence
from typing import Any
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.request_config import RequestConfig
from agentle.generations.tools.tool import Tool


class API(BaseModel):
    """
    Represents a collection of related API endpoints with shared configuration.

    This class groups multiple endpoints that share common settings like base URL,
    authentication headers, and request configuration. It provides a convenient
    way to define complete APIs that can be used by agents.
    """

    name: str = Field(description="Name of the API")

    description: str | None = Field(
        description="Description of what this API provides", default=None
    )

    base_url: str = Field(description="Base URL for all endpoints in this API")

    headers: MutableMapping[str, str] = Field(
        description="Common headers for all endpoints (e.g., authentication)",
        default_factory=dict,
    )

    request_config: RequestConfig = Field(
        description="Default request configuration for all endpoints",
        default_factory=RequestConfig,
    )

    endpoints: Sequence[Endpoint] = Field(
        description="List of endpoints in this API", default_factory=list
    )

    def add_endpoint(self, endpoint: Endpoint) -> None:
        """
        Add an endpoint to this API.

        Args:
            endpoint: Endpoint to add
        """
        if not isinstance(self.endpoints, list):
            self.endpoints = list(self.endpoints)
        self.endpoints.append(endpoint)

    def get_endpoint(self, name: str) -> Endpoint | None:
        """
        Get an endpoint by name.

        Args:
            name: Name of the endpoint to find

        Returns:
            Endpoint if found, None otherwise
        """
        for endpoint in self.endpoints:
            if endpoint.name == name:
                return endpoint
        return None

    def to_tools(self) -> Sequence[Tool[Any]]:
        """
        Convert all endpoints in this API to Tool instances.

        Returns:
            List of Tool instances for all endpoints
        """
        tools: list[Tool[Any]] = []

        for endpoint in self.endpoints:
            # Merge API-level and endpoint-level configurations
            merged_headers = dict(self.headers)
            merged_headers.update(endpoint.headers)

            # Use endpoint's request config or fall back to API's
            if endpoint.request_config == RequestConfig():
                endpoint.request_config = self.request_config

            tool = endpoint.to_tool(
                base_url=self.base_url, global_headers=merged_headers
            )
            tools.append(tool)

        return tools
