"""
Enhanced request configuration with advanced features.

Includes timeouts, retries, circuit breakers, rate limiting, caching, and more.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class RetryStrategy(StrEnum):
    """Retry strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


class CircuitBreakerState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CacheStrategy(StrEnum):
    """Cache strategies."""

    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    CUSTOM = "custom"


class RequestConfig(BaseModel):
    """Enhanced configuration for HTTP requests."""

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


class CircuitBreaker:
    """Circuit breaker implementation for resilient API calls."""

    def __init__(self, config: RequestConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[[], Any]) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                # Check if we should transition to half-open
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed >= self.config.circuit_breaker_recovery_timeout:
                        self.state = CircuitBreakerState.HALF_OPEN
                        self.success_count = 0
                    else:
                        raise CircuitBreakerError(
                            f"Circuit breaker is OPEN. Retry after {self.config.circuit_breaker_recovery_timeout - elapsed:.1f}s"
                        )

        # Execute the function
        try:
            result = await func()
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.circuit_breaker_success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                # Failure in half-open state reopens circuit
                self.state = CircuitBreakerState.OPEN
            elif self.state == CircuitBreakerState.CLOSED:
                # Check if we've hit the failure threshold
                if self.failure_count >= self.config.circuit_breaker_failure_threshold:
                    self.state = CircuitBreakerState.OPEN


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, config: RequestConfig):
        self.config = config
        self.calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire rate limit slot, waiting if necessary."""
        async with self._lock:
            now = time.time()

            # Remove old calls outside the window
            cutoff = now - self.config.rate_limit_period
            self.calls = [t for t in self.calls if t > cutoff]

            # Check if we're at the limit
            if len(self.calls) >= self.config.rate_limit_calls:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = self.config.rate_limit_period - (now - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Recursively try again
                    return await self.acquire()

            # Record this call
            self.calls.append(now)


class ResponseCache:
    """Simple in-memory response cache."""

    def __init__(self, config: RequestConfig):
        self.config = config
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, url: str, params: dict[str, Any]) -> str:
        """Create cache key from URL and params."""
        import hashlib
        import json

        key_str = f"{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get(self, url: str, params: dict[str, Any]) -> Any | None:
        """Get cached response if available and not expired."""
        async with self._lock:
            key = self._make_key(url, params)
            if key in self._cache:
                response, timestamp = self._cache[key]
                if time.time() - timestamp < self.config.cache_ttl:
                    return response
                else:
                    # Expired, remove it
                    del self._cache[key]
            return None

    async def set(self, url: str, params: dict[str, Any], response: Any) -> None:
        """Cache a response."""
        async with self._lock:
            key = self._make_key(url, params)
            self._cache[key] = (response, time.time())

    async def clear(self) -> None:
        """Clear all cached responses."""
        async with self._lock:
            self._cache.clear()


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass
