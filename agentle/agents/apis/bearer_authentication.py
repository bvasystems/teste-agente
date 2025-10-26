"""Bearer token authentication."""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from agentle.agents.apis.authentication_base import AuthenticationBase


class BearerAuthentication(AuthenticationBase):
    """Bearer token authentication."""

    def __init__(self, token: str, auto_refresh: bool = False):
        self.token = token
        self.auto_refresh = auto_refresh
        self._token_expiry: datetime | None = None

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """Add Bearer token to Authorization header."""
        headers["Authorization"] = f"Bearer {self.token}"

    async def refresh_if_needed(self) -> bool:
        """Check if token needs refresh."""
        if not self.auto_refresh:
            return False

        if self._token_expiry and datetime.now() >= self._token_expiry:
            # Token expired - subclass should implement refresh logic
            return False

        return False

    def set_token(self, token: str, expires_in: int | None = None) -> None:
        """
        Update the token.

        Args:
            token: New token
            expires_in: Token expiry in seconds
        """
        self.token = token
        if expires_in:
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
