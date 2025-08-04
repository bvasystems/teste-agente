from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import aiohttp
import ujson
import yaml
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.array_schema import ArraySchema
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.object_schema import ObjectSchema
from agentle.agents.apis.primitive_schema import PrimitiveSchema
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
        include_operations: Sequence[str] | None = None,
        exclude_operations: Sequence[str] | None = None,
    ) -> API:
        """
        Create an API instance from an OpenAPI specification.

        Args:
            spec: OpenAPI specification as dict, file path, or URL
            name: Override the API name (uses info.title from spec if not provided)
            description: Override the API description (uses info.description from spec if not provided)
            base_url_override: Override the base URL (uses first server from spec if not provided)
            headers: Additional headers to include with all requests
            request_config: Request configuration for all endpoints
            include_operations: List of operationIds to include (if None, includes all)
            exclude_operations: List of operationIds to exclude

        Returns:
            API instance configured from the OpenAPI spec

        Example:
            ```python
            # From URL
            api = await API.from_openapi_spec("https://petstore.swagger.io/v2/swagger.json")

            # From local file
            api = await API.from_openapi_spec(Path("./api-spec.yaml"))

            # From dict with custom settings
            api = await API.from_openapi_spec(
                spec_dict,
                name="Custom Pet Store",
                base_url_override="https://api.example.com",
                headers={"Authorization": "Bearer token"},
                include_operations=["getPetById", "updatePet"]
            )
            ```
        """
        # Load the OpenAPI spec
        spec_dict = await cls._load_openapi_spec(spec)

        # Validate OpenAPI version
        openapi_version = spec_dict.get("openapi") or spec_dict.get("swagger")
        if not openapi_version:
            raise ValueError(
                "Invalid OpenAPI specification: missing 'openapi' or 'swagger' field"
            )

        # Extract API info
        info = spec_dict.get("info", {})
        api_name = name or info.get("title", "Generated API")
        api_description = description or info.get("description")

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

        # Parse endpoints from paths
        endpoints = cls._parse_openapi_paths(
            spec_dict,
            include_operations=include_operations,
            exclude_operations=exclude_operations,
        )

        return cls(
            name=api_name,
            description=api_description,
            base_url=api_base_url,
            headers=headers or {},
            request_config=request_config or RequestConfig(),
            endpoints=endpoints,
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
    ) -> Sequence[Endpoint]:
        """Parse OpenAPI paths into Endpoint instances."""
        from agentle.agents.apis.endpoint import Endpoint
        from agentle.agents.apis.http_method import HTTPMethod

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

                # Apply include/exclude filters
                if include_operations and operation_id not in include_operations:
                    continue
                if exclude_operations and operation_id in exclude_operations:
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

                endpoint = Endpoint(
                    name=endpoint_name,
                    description=endpoint_description,
                    path=path,
                    method=HTTPMethod(method.upper()),
                    parameters=endpoint_parameters,
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
        from agentle.agents.apis.endpoint_parameter import EndpointParameter
        from agentle.agents.apis.parameter_location import ParameterLocation

        endpoint_params: MutableSequence[EndpointParameter] = []

        # Process standard parameters
        for param in parameters:
            if "$ref" in param:
                # Resolve reference (simplified - doesn't handle complex nested refs)
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
                "cookie": ParameterLocation.HEADER,  # Treat cookies as headers
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

            # Look for JSON content first, then any other content type
            schema = None
            for content_type in [
                "application/json",
                "application/x-www-form-urlencoded",
            ]:
                if content_type in content:
                    schema = content[content_type].get("schema", {})
                    break

            if not schema and content:
                # Take the first available content type
                first_content = next(iter(content.values()))
                schema = first_content.get("schema", {})

            if schema:
                # For request body, create a single parameter representing the body
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
        from agentle.agents.apis.array_schema import ArraySchema
        from agentle.agents.apis.object_schema import ObjectSchema
        from agentle.agents.apis.primitive_schema import PrimitiveSchema

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
