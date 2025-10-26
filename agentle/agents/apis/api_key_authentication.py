"""API key authentication."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import aiohttp
from agentle.agents.apis.api_key_location import ApiKeyLocation
from agentle.agents.apis.authentication_base import AuthenticationBase


class ApiKeyAuthentication(AuthenticationBase):
    """API Key authentication."""

    def __init__(
        self,
        api_key: str,
        location: ApiKeyLocation = ApiKeyLocation.HEADER,
        key_name: str = "X-API-Key",
    ):
        self.api_key = api_key
        self.location = location
        self.key_name = key_name

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """Add API key to the appropriate location."""
        if self.location == ApiKeyLocation.HEADER:
            headers[self.key_name] = self.api_key
        elif self.location == ApiKeyLocation.QUERY:
            params[self.key_name] = self.api_key
        elif self.location == ApiKeyLocation.COOKIE:
            headers["Cookie"] = f"{self.key_name}={self.api_key}"

    async def refresh_if_needed(self) -> bool:
        """No refresh needed for API key."""
        return False
