# Import all classes for public API
from agentle.agents.apis.api import API
from agentle.agents.apis.api_metrics import APIMetrics
from agentle.agents.apis.authentication import (
    ApiKeyLocation,
    AuthType,
    AuthenticationConfig,
    AuthenticationBase,
    NoAuthentication,
    BearerAuthentication,
    BasicAuthentication,
    ApiKeyAuthentication,
    OAuth2Authentication,
    HMACAuthentication,
    OAuth2GrantType,
)
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.file_upload import FileUpload
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.object_serialization_style import ObjectSerializationStyle
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.request_config import (
    CacheStrategy,
    RequestConfig,
    RetryStrategy,
)
from agentle.agents.apis.circuit_breaker import CircuitBreaker
from agentle.agents.apis.rate_limiter import RateLimiter
from agentle.agents.apis.response_cache import ResponseCache
from agentle.agents.apis.circuit_breaker_error import CircuitBreakerError
from agentle.agents.apis.rate_limit_error import RateLimitError
from agentle.agents.apis.request_hook import RequestHook
from agentle.agents.apis.object_schema import ObjectSchema
from agentle.agents.apis.array_schema import ArraySchema
from agentle.agents.apis.primitive_schema import PrimitiveSchema
from agentle.agents.apis.endpoints_to_tools import endpoints_to_tools

_types = {
    "ObjectSchema": ObjectSchema,
    "ArraySchema": ArraySchema,
    "PrimitiveSchema": PrimitiveSchema,
}

# Rebuild models for proper type resolution
ObjectSchema.model_rebuild(_types_namespace=_types)
ArraySchema.model_rebuild(_types_namespace=_types)
PrimitiveSchema.model_rebuild(_types_namespace=_types)
EndpointParameter.model_rebuild(_types_namespace=_types)

__all__ = [
    # API classes
    "API",
    "APIMetrics",
    # Endpoint classes
    "Endpoint",
    "EndpointParameter",
    "FileUpload",
    "RequestHook",
    "HTTPMethod",
    "ParameterLocation",
    "ObjectSerializationStyle",
    # Authentication classes
    "AuthType",
    "ApiKeyLocation",
    "OAuth2GrantType",
    "AuthenticationBase",
    "NoAuthentication",
    "BearerAuthentication",
    "BasicAuthentication",
    "ApiKeyAuthentication",
    "OAuth2Authentication",
    "HMACAuthentication",
    "AuthenticationConfig",
    # Request configuration classes
    "RequestConfig",
    "RetryStrategy",
    "CacheStrategy",
    "CircuitBreaker",
    "RateLimiter",
    "ResponseCache",
    "CircuitBreakerError",
    "RateLimitError",
    # Schema classes
    "ObjectSchema",
    "ArraySchema",
    "PrimitiveSchema",
    # Utilities
    "endpoints_to_tools",
]
