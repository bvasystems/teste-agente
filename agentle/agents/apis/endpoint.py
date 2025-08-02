"""
API endpoint integration for Agentle framework.

This module provides classes for defining HTTP API endpoints that can be automatically
converted to tools for use by AI agents. This allows users to integrate with REST APIs
without writing HTTP request functions manually.

Example:
```python
from agentle.apis.endpoint import Endpoint, API, HTTPMethod, ParameterLocation, EndpointParameter
from agentle.agents.agent import Agent

# Define individual endpoints
weather_endpoint = Endpoint(
    name="get_weather",
    description="Get current weather for a location",
    call_condition="when user asks about weather, current conditions, or temperature",
    url="https://api.weather.com/v1/current",
    method=HTTPMethod.GET,
    parameters=[
        EndpointParameter(
            name="location",
            description="City name or coordinates",
            param_type="string",
            location=ParameterLocation.QUERY,
            required=True
        ),
        EndpointParameter(
            name="units",
            description="Temperature units",
            param_type="string",
            location=ParameterLocation.QUERY,
            default="metric"
        )
    ]
)

# Or define an API with multiple endpoints
weather_api = API(
    name="WeatherAPI",
    base_url="https://api.weather.com/v1",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    endpoints=[
        Endpoint(
            name="get_current_weather",
            description="Get current weather conditions",
            call_condition="when user asks about current weather",
            path="/current",
            method=HTTPMethod.GET,
            parameters=[
                EndpointParameter("location", "Location to get weather for", "string",
                                ParameterLocation.QUERY, required=True)
            ]
        ),
        Endpoint(
            name="get_forecast",
            description="Get weather forecast",
            call_condition="when user asks about weather forecast or future weather",
            path="/forecast",
            method=HTTPMethod.GET,
            parameters=[
                EndpointParameter("location", "Location for forecast", "string",
                                ParameterLocation.QUERY, required=True),
                EndpointParameter("days", "Number of days", "integer",
                                ParameterLocation.QUERY, default=5)
            ]
        )
    ]
)

# Use with an agent
agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.5-flash",
    instructions="You are a weather assistant.",
    tools=[weather_endpoint],  # Individual endpoint
    apis=[weather_api]         # Full API with multiple endpoints
)
```
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping, Sequence
from typing import Any, Literal

import aiohttp
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.request_config import RequestConfig
from agentle.generations.tools.tool import Tool

logger = logging.getLogger(__name__)


class Endpoint(BaseModel):
    """
    Represents a single HTTP API endpoint that can be called by an agent.

    This class encapsulates all the information needed to make HTTP requests
    to a specific API endpoint, including URL, method, parameters, and conditions
    for when the agent should use this endpoint.
    """

    name: str = Field(description="Unique name for this endpoint")

    description: str = Field(
        description="Human-readable description of what this endpoint does"
    )

    call_condition: str | None = Field(
        description="Condition or context when the agent should call this endpoint",
        default=None,
    )

    url: str | None = Field(
        description="Complete URL for the endpoint (if not using base_url + path)",
        default=None,
    )

    path: str | None = Field(
        description="Path to append to base URL (when used with API class)",
        default=None,
    )

    method: HTTPMethod = Field(
        description="HTTP method for this endpoint", default=HTTPMethod.GET
    )

    parameters: Sequence[EndpointParameter] = Field(
        description="Parameters that can be passed to this endpoint",
        default_factory=list,
    )

    headers: MutableMapping[str, str] = Field(
        description="Additional headers for this endpoint", default_factory=dict
    )

    request_config: RequestConfig = Field(
        description="Request configuration for this endpoint",
        default_factory=RequestConfig,
    )

    response_format: Literal["json", "text", "bytes"] = Field(
        description="Expected response format", default="json"
    )

    def get_full_url(self, base_url: str | None = None) -> str:
        """
        Get the complete URL for this endpoint.

        Args:
            base_url: Base URL to prepend to the path

        Returns:
            Complete URL for the endpoint

        Raises:
            ValueError: If neither url nor (base_url + path) is available
        """
        if self.url:
            return self.url

        if base_url and self.path:
            return f"{base_url.rstrip('/')}/{self.path.lstrip('/')}"

        raise ValueError(
            f"Endpoint '{self.name}' must have either 'url' or 'path' (with base_url)"
        )

    def get_enhanced_description(self) -> str:
        """
        Get description enhanced with call condition.

        Returns:
            Description with call condition appended if available
        """
        base_desc = self.description

        if self.call_condition:
            return f"{base_desc}\n\nCall this endpoint: {self.call_condition}"

        return base_desc

    def to_tool_parameters(self) -> dict[str, object]:
        """
        Convert endpoint parameters to tool parameter format.

        Returns:
            Dictionary of parameters in tool format
        """
        tool_params: dict[str, object] = {}

        for param in self.parameters:
            param_info: dict[str, object] = {
                "type": param.param_type,
                "description": param.description,
                "required": param.required,
            }

            if param.default is not None:
                param_info["default"] = param.default

            if param.enum:
                param_info["enum"] = list(param.enum)

            tool_params[param.name] = param_info

        return tool_params

    async def _make_request(
        self,
        base_url: str | None = None,
        global_headers: MutableMapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Internal method to make the HTTP request.

        Args:
            base_url: Base URL if using path-based endpoint
            global_headers: Global headers to merge with endpoint headers
            **kwargs: Parameters for the request

        Returns:
            Response data based on response_format
        """
        url = self.get_full_url(base_url)

        # Separate parameters by location
        query_params: dict[str, Any] = {}
        body_params: dict[str, Any] = {}
        header_params: dict[str, str] = {}
        path_params: dict[str, Any] = {}

        for param in self.parameters:
            param_name = param.name

            # Use provided value or default
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param.default is not None:
                value = param.default
            elif param.required:
                raise ValueError(f"Required parameter '{param_name}' not provided")
            else:
                continue

            # Place parameter in appropriate location
            if param.location == ParameterLocation.QUERY:
                query_params[param_name] = value
            elif param.location == ParameterLocation.BODY:
                body_params[param_name] = value
            elif param.location == ParameterLocation.HEADER:
                header_params[param_name] = str(value)
            elif param.location == ParameterLocation.PATH:
                path_params[param_name] = value

        # Replace path parameters in URL
        for param_name, value in path_params.items():
            url = url.replace(f"{{{param_name}}}", str(value))

        # Merge headers
        headers = {}
        if global_headers:
            headers.update(global_headers)
        headers.update(self.headers)
        headers.update(header_params)

        # Prepare request data
        request_kwargs: dict[str, Any] = {
            "timeout": aiohttp.ClientTimeout(total=self.request_config.timeout),
            "headers": headers,
            "allow_redirects": self.request_config.follow_redirects,
        }

        if query_params:
            request_kwargs["params"] = query_params

        if body_params and self.method in [
            HTTPMethod.POST,
            HTTPMethod.PUT,
            HTTPMethod.PATCH,
        ]:
            request_kwargs["json"] = body_params
            if "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"

        # Make request with retries
        last_exception = None

        for attempt in range(self.request_config.max_retries + 1):
            try:
                logger.debug(
                    f"Making {self.method} request to {url} (attempt {attempt + 1})"
                )

                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=self.method.value, url=url, **request_kwargs
                    ) as response:
                        # Check for HTTP errors
                        if response.status >= 400:
                            error_text = await response.text()
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"HTTP {response.status}: {error_text}",
                            )

                        # Parse response based on format
                        if self.response_format == "json":
                            return await response.json()
                        elif self.response_format == "text":
                            return await response.text()
                        elif self.response_format == "bytes":
                            return await response.read()
                        else:
                            return await response.text()

            except Exception as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")

                if attempt < self.request_config.max_retries:
                    import asyncio

                    await asyncio.sleep(self.request_config.retry_delay)
                else:
                    break

        # If we get here, all attempts failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("All request attempts failed")

    def to_tool(
        self,
        base_url: str | None = None,
        global_headers: MutableMapping[str, str] | None = None,
    ) -> Tool[Any]:
        """
        Convert this endpoint to a Tool instance.

        Args:
            base_url: Base URL for path-based endpoints
            global_headers: Global headers to include in requests

        Returns:
            Tool instance that can be used by agents
        """

        async def endpoint_callable(**kwargs: Any) -> Any:
            """Callable function for the tool."""
            return await self._make_request(
                base_url=base_url, global_headers=global_headers, **kwargs
            )

        return Tool.from_callable(
            endpoint_callable,
            name=self.name,
            description=self.get_enhanced_description(),
        )
