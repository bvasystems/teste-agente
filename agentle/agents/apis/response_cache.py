"""Simple in-memory response cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from agentle.agents.apis.request_config import RequestConfig


class ResponseCache:
    """Simple in-memory response cache."""

    def __init__(self, config: RequestConfig):
        self.config = config
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, url: str, params: dict[str, Any]) -> str:
        """Create cache key from URL and params."""
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
