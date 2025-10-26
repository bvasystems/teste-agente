"""Rate limiter for API calls."""

from __future__ import annotations

import asyncio
import time

from agentle.agents.apis.request_config import RequestConfig


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
