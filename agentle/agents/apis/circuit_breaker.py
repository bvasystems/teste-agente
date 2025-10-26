"""Circuit breaker for resilient API calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from agentle.agents.apis.circuit_breaker_error import CircuitBreakerError
from agentle.agents.apis.circuit_breaker_state import CircuitBreakerState
from agentle.agents.apis.request_config import RequestConfig


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
