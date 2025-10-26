"""
Enhanced request configuration with advanced features.

Includes timeouts, retries, circuit breakers, rate limiting, caching, and more.
"""

from __future__ import annotations

from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.cache_strategy import CacheStrategy
from agentle.agents.apis.retry_strategy import RetryStrategy


class RequestConfig(BaseModel):
    """
    Enhanced configuration for HTTP requests.

    This configuration can be set at both API-level and per-endpoint level.
    Endpoint-level configs override API-level configs.

    Example:
        ```python
        # API-level config (applies to all endpoints)
        api_config = RequestConfig(
            timeout=30.0,
            max_retries=3,
            enable_caching=True
        )

        # Per-endpoint override
        endpoint_config = RequestConfig(
            timeout=60.0,  # Override for this specific endpoint
            max_retries=5,
            cache_ttl=600.0
        )
        ```
    """

    # Timeouts (in seconds)
    timeout: float = Field(description="Total request timeout in seconds", default=30.0)
    connect_timeout: float | None = Field(
        description="Connection timeout in seconds", default=None
    )
    read_timeout: float | None = Field(
        description="Read timeout in seconds", default=None
    )

    # Retry configuration
    max_retries: int = Field(
        description="Maximum number of retries for failed requests", default=3
    )
    retry_delay: float = Field(
        description="Base delay between retries in seconds", default=1.0
    )
    retry_strategy: RetryStrategy = Field(
        description="Strategy for calculating retry delays",
        default=RetryStrategy.EXPONENTIAL,
    )
    retry_on_status_codes: Sequence[int] = Field(
        description="HTTP status codes that should trigger retries",
        default_factory=lambda: [408, 429, 500, 502, 503, 504],
    )
    retry_on_exceptions: bool = Field(
        description="Whether to retry on network exceptions", default=True
    )

    # Circuit breaker configuration
    enable_circuit_breaker: bool = Field(
        description="Enable circuit breaker pattern", default=False
    )
    circuit_breaker_failure_threshold: int = Field(
        description="Number of failures before opening circuit", default=5
    )
    circuit_breaker_recovery_timeout: float = Field(
        description="Seconds to wait before attempting recovery", default=60.0
    )
    circuit_breaker_success_threshold: int = Field(
        description="Number of successes in half-open state to close circuit", default=2
    )

    # Rate limiting
    enable_rate_limiting: bool = Field(
        description="Enable rate limiting", default=False
    )
    rate_limit_calls: int = Field(
        description="Maximum number of calls per period", default=100
    )
    rate_limit_period: float = Field(
        description="Rate limit period in seconds", default=60.0
    )
    respect_retry_after: bool = Field(
        description="Respect Retry-After header from server", default=True
    )

    # Caching
    enable_caching: bool = Field(description="Enable response caching", default=False)
    cache_strategy: CacheStrategy = Field(
        description="Caching strategy to use", default=CacheStrategy.MEMORY
    )
    cache_ttl: float = Field(description="Cache TTL in seconds", default=300.0)
    cache_only_get: bool = Field(description="Only cache GET requests", default=True)

    # Request/Response hooks
    enable_request_logging: bool = Field(
        description="Enable request logging", default=False
    )
    enable_response_logging: bool = Field(
        description="Enable response logging", default=False
    )
    enable_metrics: bool = Field(description="Enable metrics collection", default=False)

    # Connection settings
    follow_redirects: bool = Field(
        description="Whether to follow HTTP redirects", default=True
    )
    max_redirects: int = Field(
        description="Maximum number of redirects to follow", default=10
    )
    verify_ssl: bool = Field(description="Verify SSL certificates", default=True)
    ssl_cert_path: str | None = Field(
        description="Path to custom SSL certificate", default=None
    )

    # Proxy configuration
    proxy_url: str | None = Field(
        description="Proxy URL (http://host:port)", default=None
    )
    proxy_auth: tuple[str, str] | None = Field(
        description="Proxy authentication (username, password)", default=None
    )

    # Content handling
    compress_request: bool = Field(
        description="Enable request compression", default=False
    )
    decompress_response: bool = Field(
        description="Enable response decompression", default=True
    )
