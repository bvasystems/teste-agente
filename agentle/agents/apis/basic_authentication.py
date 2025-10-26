"""Basic authentication."""

from __future__ import annotations

import base64
from collections.abc import MutableMapping
from typing import Any

import aiohttp
from agentle.agents.apis.authentication_base import AuthenticationBase


class BasicAuthentication(AuthenticationBase):
    """HTTP Basic authentication."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """Add Basic auth to Authorization header."""
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"

    async def refresh_if_needed(self) -> bool:
        """No refresh needed for Basic auth."""
        return False
