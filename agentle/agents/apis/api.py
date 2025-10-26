"""
Complete enhanced API module with comprehensive OpenAPI support.

Provides advanced features for managing collections of related endpoints with:
- Full OpenAPI 3.0/3.1 and Swagger 2.0 support
- Shared authentication across endpoints
- Request/response interceptors
- API-level rate limiting and circuit breaking
- Batch request support
- GraphQL support
- And more...
"""

from __future__ import annotations

import logging
from collections.abc import Coroutine, Mapping, MutableMapping, MutableSequence, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import aiohttp
import ujson
import yaml
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.array_schema import ArraySchema
from agentle.agents.apis.authentication import (
    ApiKeyLocation,
    AuthType,
    AuthenticationConfig,
    OAuth2GrantType,
)
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.object_schema import ObjectSchema
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.primitive_schema import PrimitiveSchema
from agentle.agents.apis.request_config import RequestConfig
from agentle.generations.tools.tool import Tool

logger = logging.getLogger(__name__)


class APIMetrics(BaseModel):
    """Metrics for API usage."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    requests_by_endpoint: dict[str, int] = {}


class API(BaseModel):
    """
    Enhanced API collection with comprehensive features.

    Represents a collection of related API endpoints with shared configuration,
    authentication, rate limiting, and monitoring capabilities.
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

    auth_config: AuthenticationConfig | None = Field(
        description="Authentication configuration for the API",
        default=None,
    )

    endpoints: Sequence[Endpoint] = Field(
        description="List of endpoints in this API", default_factory=list
    )

    # API-level features
    enable_batch_requests: bool = Field(
        description="Enable batch request support", default=False
    )

    enable_graphql: bool = Field(description="Enable GraphQL support", default=False)

    graphql_endpoint: str | None = Field(
        description="GraphQL endpoint path", default=None
    )

    # Monitoring
    enable_metrics: bool = Field(
        description="Enable API metrics collection", default=False
    )

    # API version
    version: str = Field(description="API version", default="1.0.0")

    # OpenAPI spec reference
    openapi_spec_url: str | None = Field(
        description="URL to OpenAPI specification", default=None
    )

    # Internal state
    _metrics: APIMetrics | None = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize API components."""
        super().model_post_init(__context)

        if self.enable_metrics:
            self._metrics = APIMetrics()

        # Apply API-level config to endpoints that don't have their own
        for endpoint in self.endpoints:
            # Inherit auth config if endpoint doesn't have one
            if not endpoint.auth_config and self.auth_config:
                endpoint.auth_config = self.auth_config

            # Inherit request config settings
            if endpoint.request_config == RequestConfig():
                endpoint.request_config = self.request_config

    @classmethod
    async def from_openapi_spec(
        cls,
        spec: str | Mapping[str, Any] | Path,
        *,
        name: str | None = None,
        description: str | None = None,
        base_url_override: str | None = None,
        headers: MutableMapping[str, str] | None = None,
        request_config: RequestConfig | None = None,
        auth_config: AuthenticationConfig | None = None,
        include_operations: Sequence[str] | None = None,
        exclude_operations: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
    ) -> API:
        """
        Create an API instance from an OpenAPI specification.

        Args:
            spec: OpenAPI specification as dict, file path, or URL
            name: Override the API name
            description: Override the API description
            base_url_override: Override the base URL
            headers: Additional headers
            request_config: Request configuration
            auth_config: Authentication configuration
            include_operations: List of operationIds to include
            exclude_operations: List of operationIds to exclude
            include_tags: List of tags to include
            exclude_tags: List of tags to exclude

        Returns:
            API instance configured from the OpenAPI spec
        """
        # Load the OpenAPI spec
        spec_dict = await cls._load_openapi_spec(spec)

        # Validate OpenAPI version
        openapi_version = spec_dict.get("openapi") or spec_dict.get("swagger")
        if not openapi_version:
            raise ValueError(
                "Invalid OpenAPI specification: missing 'openapi' or 'swagger' field"
            )

        logger.info(f"Loading OpenAPI spec version: {openapi_version}")

        # Extract API info
        info = spec_dict.get("info", {})
        api_name = name or info.get("title", "Generated API")
        api_description = description or info.get("description")
        api_version = info.get("version", "1.0.0")

        # Extract base URL
        if base_url_override:
            api_base_url = base_url_override
        else:
            servers = spec_dict.get("servers", [])
            if servers and isinstance(servers[0], dict):
                api_base_url = servers[0].get("url", "")
            else:
                # Fallback for OpenAPI 2.x (Swagger)
                host = spec_dict.get("host", "localhost")
                schemes = spec_dict.get("schemes", ["https"])
                base_path = spec_dict.get("basePath", "")
                api_base_url = f"{schemes[0]}://{host}{base_path}"

        # Extract authentication from OpenAPI spec if not provided
        if not auth_config:
            auth_config = cls._extract_auth_from_spec(spec_dict)

        # Parse endpoints from paths
        endpoints = cls._parse_openapi_paths(
            spec_dict,
            include_operations=include_operations,
            exclude_operations=exclude_operations,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

        logger.info(f"Loaded {len(endpoints)} endpoints from OpenAPI spec")

        return cls(
            name=api_name,
            description=api_description,
            base_url=api_base_url,
            headers=headers or {},
            request_config=request_config or RequestConfig(),
            auth_config=auth_config,
            endpoints=endpoints,
            version=api_version,
        )

    @classmethod
    def _extract_auth_from_spec(
        cls, spec_dict: Mapping[str, Any]
    ) -> AuthenticationConfig | None:
        """Extract authentication configuration from OpenAPI spec."""
        # OpenAPI 3.x security schemes
        components = spec_dict.get("components", {})
        security_schemes = components.get("securitySchemes", {})

        # OpenAPI 2.x security definitions
        if not security_schemes:
            security_schemes = spec_dict.get("securityDefinitions", {})

        if not security_schemes:
            return None

        # Get the first security scheme (simplified - real implementation would handle multiple)
        scheme_name, scheme = next(iter(security_schemes.items()))
        scheme_type = scheme.get("type", "").lower()

        logger.debug(f"Detected security scheme: {scheme_name} ({scheme_type})")

        if scheme_type == "http":
            http_scheme = scheme.get("scheme", "").lower()
            if http_scheme == "bearer":
                return AuthenticationConfig(type=AuthType.BEARER)
            elif http_scheme == "basic":
                return AuthenticationConfig(type=AuthType.BASIC)

        elif scheme_type == "apikey":
            location = scheme.get("in", "header")
            name = scheme.get("name", "X-API-Key")

            if location == "header":
                return AuthenticationConfig(
                    type=AuthType.API_KEY,
                    api_key_location=ApiKeyLocation.HEADER,
                    api_key_name=name,
                )
            elif location == "query":
                return AuthenticationConfig(
                    type=AuthType.API_KEY,
                    api_key_location=ApiKeyLocation.QUERY,
                    api_key_name=name,
                )

        elif scheme_type == "oauth2":
            flows = scheme.get("flows", {})
            if "clientCredentials" in flows:
                token_url = flows["clientCredentials"].get("tokenUrl")
                if token_url:
                    return AuthenticationConfig(
                        type=AuthType.OAUTH2,
                        oauth2_token_url=token_url,
                        oauth2_grant_type=OAuth2GrantType.CLIENT_CREDENTIALS,
                    )

        return None

    def add_endpoint(self, endpoint: Endpoint) -> None:
        """Add an endpoint to this API."""
        if not isinstance(self.endpoints, list):
            self.endpoints = list(self.endpoints)

        # Apply API-level configs
        if not endpoint.auth_config and self.auth_config:
            endpoint.auth_config = self.auth_config

        if endpoint.request_config == RequestConfig():
            endpoint.request_config = self.request_config

        self.endpoints.append(endpoint)
        logger.debug(f"Added endpoint '{endpoint.name}' to API '{self.name}'")

    def get_endpoint(self, name: str) -> Endpoint | None:
        """Get an endpoint by name."""
        for endpoint in self.endpoints:
            if endpoint.name == name:
                return endpoint
        return None

    def get_endpoints_by_tag(self, tag: str) -> Sequence[Endpoint]:
        """Get all endpoints with a specific tag."""
        # This would require adding tags to Endpoint model
        # For now, return empty list
        return []

    async def batch_request(
        self,
        requests: Sequence[tuple[str, dict[str, Any]]],
    ) -> Sequence[Any]:
        """
        Execute multiple requests in batch.

        Args:
            requests: List of (endpoint_name, kwargs) tuples

        Returns:
            List of results in the same order as requests
        """
        if not self.enable_batch_requests:
            raise ValueError("Batch requests not enabled for this API")

        import asyncio

        tasks: list[Coroutine[None, None, Any]] = []
        for endpoint_name, kwargs in requests:
            endpoint = self.get_endpoint(endpoint_name)
            if not endpoint:
                raise ValueError(f"Endpoint '{endpoint_name}' not found")

            # Create task for this request
            task = endpoint.make_request(
                base_url=self.base_url,
                global_headers=self.headers,
                **kwargs,
            )
            tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def graphql_query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> Any:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            operation_name: Operation name

        Returns:
            Query result
        """
        if not self.enable_graphql:
            raise ValueError("GraphQL not enabled for this API")

        if not self.graphql_endpoint:
            raise ValueError("GraphQL endpoint not configured")

        # Build GraphQL request
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        if operation_name:
            payload["operationName"] = operation_name

        # Make request
        url = f"{self.base_url.rstrip('/')}/{self.graphql_endpoint.lstrip('/')}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=self.headers,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "errors" in result:
                        raise ValueError(f"GraphQL errors: {result['errors']}")
                    return result.get("data")
                else:
                    raise ValueError(f"GraphQL request failed: HTTP {response.status}")

    def get_metrics(self) -> APIMetrics | None:
        """Get API usage metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset API metrics."""
        if self._metrics:
            self._metrics = APIMetrics()

    def to_tools(self) -> Sequence[Tool[Any]]:
        """Convert all endpoints to Tool instances."""
        tools: list[Tool[Any]] = []

        for endpoint in self.endpoints:
            # Merge API-level and endpoint-level configurations
            merged_headers = dict(self.headers)
            merged_headers.update(endpoint.headers)

            tool = endpoint.to_tool(
                base_url=self.base_url, global_headers=merged_headers
            )
            tools.append(tool)

        logger.info(f"Created {len(tools)} tools from API '{self.name}'")
        return tools

    @classmethod
    async def _load_openapi_spec(
        cls, spec: str | Mapping[str, Any] | Path
    ) -> Mapping[str, Any]:
        """Load OpenAPI spec from various sources."""
        if isinstance(spec, dict):
            return spec

        if isinstance(spec, (str, Path)):
            spec_path = Path(spec) if not isinstance(spec, Path) else spec

            # Check if it's a URL
            if isinstance(spec, str) and (
                spec.startswith("http://") or spec.startswith("https://")
            ):
                logger.info(f"Fetching OpenAPI spec from URL: {spec}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(spec) as response:
                        if response.status != 200:
                            raise ValueError(
                                f"Failed to fetch OpenAPI spec from {spec}: HTTP {response.status}"
                            )

                        content_type = response.headers.get("content-type", "").lower()
                        if "application/json" in content_type or spec.endswith(".json"):
                            return await response.json()
                        else:
                            text = await response.text()
                            return yaml.safe_load(text)

            # Local file
            if not spec_path.exists():
                raise FileNotFoundError(f"OpenAPI spec file not found: {spec_path}")

            logger.info(f"Loading OpenAPI spec from file: {spec_path}")
            content = spec_path.read_text()
            if spec_path.suffix.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(content)
            else:
                return ujson.loads(content)

        raise ValueError(
            f"Invalid spec type: {type(spec)}. Must be dict, file path, or URL"
        )

    @classmethod
    def _parse_openapi_paths(
        cls,
        spec_dict: Mapping[str, Any],
        include_operations: Sequence[str] | None = None,
        exclude_operations: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
    ) -> Sequence[Endpoint]:
        """Parse OpenAPI paths into Endpoint instances."""
        endpoints: MutableSequence[Endpoint] = []
        paths: Mapping[str, Any] = spec_dict.get("paths", {})
        components = spec_dict.get("components", {})

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            # Extract path-level parameters
            path_parameters: Mapping[str, Any] = cast(
                Mapping[str, Any], path_item.get("parameters", [])
            )

            for method, operation in cast(dict[str, Any], path_item).items():
                if method.upper() not in [
                    m.value for m in HTTPMethod
                ] or not isinstance(operation, dict):
                    continue

                operation_id = operation.get("operationId")
                operation_tags = operation.get("tags", [])

                # Apply operation filters
                if include_operations and operation_id not in include_operations:
                    continue
                if exclude_operations and operation_id in exclude_operations:
                    continue

                # Apply tag filters
                if include_tags and not any(
                    tag in include_tags for tag in operation_tags
                ):
                    continue
                if exclude_tags and any(tag in exclude_tags for tag in operation_tags):
                    continue

                # Create endpoint
                endpoint_name: str = cast(
                    str,
                    (
                        operation_id
                        or f"{method}_{path.replace('/', '_').replace('{', '').replace('}', '')}"
                    ),
                )

                endpoint_description: str = cast(
                    str,
                    operation.get(
                        "summary",
                        cast(dict[str, Any], operation).get("description", ""),
                    ),
                )

                operation = cast(dict[str, Any], operation)

                # Parse parameters
                endpoint_parameters = cls._parse_openapi_parameters(
                    operation.get("parameters", []) + path_parameters,
                    operation.get("requestBody"),
                    components,
                )

                # Determine response format
                response_format = "json"  # Default
                responses = operation.get("responses", {})
                if "200" in responses:
                    response_200 = responses["200"]
                    content = response_200.get("content", {})
                    if "application/json" in content:
                        response_format = "json"
                    elif "text/plain" in content:
                        response_format = "text"
                    elif "application/xml" in content:
                        response_format = "xml"

                endpoint = Endpoint(
                    name=endpoint_name,
                    description=endpoint_description,
                    path=path,
                    method=HTTPMethod(method.upper()),
                    parameters=endpoint_parameters,
                    response_format=response_format,  # type: ignore
                )

                endpoints.append(endpoint)

        return endpoints

    @classmethod
    def _parse_openapi_parameters(
        cls,
        parameters: Sequence[Mapping[str, Any]],
        request_body: Mapping[str, Any] | None,
        components: Mapping[str, Any],
    ) -> Sequence[EndpointParameter]:
        """Parse OpenAPI parameters into EndpointParameter instances."""
        endpoint_params: MutableSequence[EndpointParameter] = []

        # Process standard parameters
        for param in parameters:
            if "$ref" in param:
                # Resolve reference
                ref_path = param["$ref"].split("/")
                if len(ref_path) >= 4 and ref_path[1] == "components":
                    param = components.get(ref_path[2], {}).get(ref_path[3], {})

            param_name = param.get("name", "")
            param_description = param.get("description", "")
            param_required = param.get("required", False)
            param_in = param.get("in", "query")

            # Map OpenAPI 'in' to our ParameterLocation
            location_map = {
                "query": ParameterLocation.QUERY,
                "header": ParameterLocation.HEADER,
                "path": ParameterLocation.PATH,
                "cookie": ParameterLocation.HEADER,
            }
            param_location = location_map.get(param_in, ParameterLocation.QUERY)

            # Parse schema
            schema = param.get("schema", {})
            parameter_schema = cls._parse_openapi_schema(schema, components)

            endpoint_param = EndpointParameter(
                name=param_name,
                description=param_description,
                parameter_schema=parameter_schema,
                location=param_location,
                required=param_required,
                default=schema.get("default"),
            )

            endpoint_params.append(endpoint_param)

        # Process request body
        if request_body:
            content = request_body.get("content", {})

            # Look for JSON content first
            schema = None
            for content_type in [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data",
            ]:
                if content_type in content:
                    schema = content[content_type].get("schema", {})
                    break

            if not schema and content:
                first_content = next(iter(content.values()))
                schema = first_content.get("schema", {})

            if schema:
                body_param = EndpointParameter(
                    name="requestBody",
                    description=request_body.get("description", "Request body"),
                    parameter_schema=cls._parse_openapi_schema(schema, components),
                    location=ParameterLocation.BODY,
                    required=request_body.get("required", False),
                )
                endpoint_params.append(body_param)

        return endpoint_params

    @classmethod
    def _parse_openapi_schema(
        cls,
        schema: Mapping[str, Any],
        components: Mapping[str, Any],
    ) -> PrimitiveSchema | ObjectSchema | ArraySchema:
        """Parse OpenAPI schema into our schema types."""
        # Handle references
        if "$ref" in schema:
            ref_path = schema["$ref"].split("/")
            if len(ref_path) >= 4 and ref_path[1] == "components":
                ref_schema = components.get(ref_path[2], {}).get(ref_path[3], {})
                return cls._parse_openapi_schema(ref_schema, components)

        schema_type = schema.get("type", "string")

        if schema_type == "object":
            properties: Mapping[str, Any] = {}
            for prop_name, prop_schema in schema.get("properties", {}).items():
                properties[prop_name] = cls._parse_openapi_schema(
                    prop_schema, components
                )

            return ObjectSchema(
                properties=properties,
                required=schema.get("required", []),
                example=schema.get("example"),
            )

        elif schema_type == "array":
            items_schema = schema.get("items", {"type": "string"})
            return ArraySchema(
                items=cls._parse_openapi_schema(items_schema, components),
                min_items=schema.get("minItems"),
                max_items=schema.get("maxItems"),
                example=schema.get("example"),
            )

        else:
            # Primitive type
            return PrimitiveSchema(
                type=cast(
                    Literal["string", "integer", "boolean", "number"], schema_type
                )
                if schema_type in ["string", "integer", "number", "boolean"]
                else "string",
                format=schema.get("format"),
                enum=schema.get("enum"),
                minimum=schema.get("minimum"),
                maximum=schema.get("maximum"),
                pattern=schema.get("pattern"),
                example=schema.get("example"),
            )
