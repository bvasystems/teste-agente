"""OAuth2 authentication."""

from __future__ import annotations

import asyncio
from collections.abc import MutableMapping
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from agentle.agents.apis.authentication_base import AuthenticationBase
from agentle.agents.apis.oauth2_grant_type import OAuth2GrantType


class OAuth2Authentication(AuthenticationBase):
    """OAuth2 authentication with token refresh."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        grant_type: OAuth2GrantType = OAuth2GrantType.CLIENT_CREDENTIALS,
        scope: str | None = None,
        refresh_token: str | None = None,
        scopes: list[str] | None = None,
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.grant_type = grant_type
        # Support both single scope and multiple scopes
        # If scopes list is provided, use it; otherwise fall back to single scope
        self.scopes = scopes
        self.scope = scope
        self.refresh_token_value = refresh_token

        self.access_token: str | None = None
        self.token_expiry: datetime | None = None
        self._refresh_lock = asyncio.Lock()

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """Add OAuth2 token to Authorization header."""
        # Ensure we have a valid token
        await self.refresh_if_needed()

        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

    async def refresh_if_needed(self) -> bool:
        """Refresh token if expired or missing."""
        # Check if token needs refresh
        if self.access_token and self.token_expiry:
            # Add 60 second buffer before expiry
            if datetime.now() < self.token_expiry - timedelta(seconds=60):
                return False

        # Use lock to prevent concurrent refreshes
        async with self._refresh_lock:
            # Double-check after acquiring lock
            if self.access_token and self.token_expiry:
                if datetime.now() < self.token_expiry - timedelta(seconds=60):
                    return False

            # Refresh the token
            await self._fetch_token()
            return True

    async def _fetch_token(self) -> None:
        """Fetch a new access token."""
        data: dict[str, str] = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": self.grant_type.value,
        }

        # Handle scopes - prefer scopes list over single scope
        if self.scopes:
            data["scope"] = " ".join(self.scopes)
        elif self.scope:
            data["scope"] = self.scope

        if (
            self.grant_type == OAuth2GrantType.REFRESH_TOKEN
            and self.refresh_token_value
        ):
            data["refresh_token"] = self.refresh_token_value

        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]

                    # Calculate expiry
                    expires_in = token_data.get("expires_in", 3600)
                    self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

                    # Update refresh token if provided
                    if "refresh_token" in token_data:
                        self.refresh_token_value = token_data["refresh_token"]
                else:
                    raise ValueError(
                        f"Failed to fetch OAuth2 token: HTTP {response.status}"
                    )
