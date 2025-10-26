"""
Authentication support for API endpoints.

Provides various authentication methods including Bearer, Basic, OAuth2, API Key, and custom schemes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import aiohttp
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class AuthType(StrEnum):
    """Types of authentication supported."""

    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM = "custom"
    AWS_SIGNATURE = "aws_signature"
    HMAC = "hmac"


class ApiKeyLocation(StrEnum):
    """Where to place API key."""

    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"


class OAuth2GrantType(StrEnum):
    """OAuth2 grant types."""

    CLIENT_CREDENTIALS = "client_credentials"
    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"
    PASSWORD = "password"


class AuthenticationBase(ABC):
    """Base class for authentication handlers."""

    @abstractmethod
    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """
        Apply authentication to the request.

        Args:
            session: aiohttp session
            url: Request URL
            headers: Request headers (will be modified)
            params: Request parameters (will be modified)
        """
        pass

    @abstractmethod
    async def refresh_if_needed(self) -> bool:
        """
        Refresh authentication if needed (e.g., expired tokens).

        Returns:
            True if refresh was performed, False otherwise
        """
        pass


class NoAuthentication(AuthenticationBase):
    """No authentication."""

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """No authentication to apply."""
        pass

    async def refresh_if_needed(self) -> bool:
        """No refresh needed."""
        return False


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
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.grant_type = grant_type
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

        if self.scope:
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


class HMACAuthentication(AuthenticationBase):
    """HMAC signature authentication."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "sha256",
        header_name: str = "X-Signature",
        include_timestamp: bool = True,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.header_name = header_name
        self.include_timestamp = include_timestamp

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """Add HMAC signature to headers."""
        # Build signature string
        timestamp = str(int(time.time()))
        signature_string = url

        if self.include_timestamp:
            signature_string = f"{timestamp}:{signature_string}"
            headers["X-Timestamp"] = timestamp

        # Calculate HMAC
        hash_func = getattr(hashlib, self.algorithm)
        signature = hmac.new(
            self.secret_key.encode(), signature_string.encode(), hash_func
        ).hexdigest()

        headers[self.header_name] = signature

    async def refresh_if_needed(self) -> bool:
        """No refresh needed for HMAC."""
        return False


class AuthenticationConfig(BaseModel):
    """Configuration for API authentication."""

    type: AuthType = Field(default=AuthType.NONE)

    # Bearer token
    bearer_token: str | None = Field(default=None)

    # Basic auth
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)

    # API Key
    api_key: str | None = Field(default=None)
    api_key_location: ApiKeyLocation = Field(default=ApiKeyLocation.HEADER)
    api_key_name: str = Field(default="X-API-Key")

    # OAuth2
    oauth2_token_url: str | None = Field(default=None)
    oauth2_client_id: str | None = Field(default=None)
    oauth2_client_secret: str | None = Field(default=None)
    oauth2_grant_type: OAuth2GrantType = Field(
        default=OAuth2GrantType.CLIENT_CREDENTIALS
    )
    oauth2_scope: str | None = Field(default=None)
    oauth2_refresh_token: str | None = Field(default=None)

    # HMAC
    hmac_secret_key: str | None = Field(default=None)
    hmac_algorithm: str = Field(default="sha256")
    hmac_header_name: str = Field(default="X-Signature")

    # Custom
    custom_headers: MutableMapping[str, str] = Field(default_factory=dict)

    def create_handler(self) -> AuthenticationBase:
        """Create authentication handler from config."""
        if self.type == AuthType.NONE:
            return NoAuthentication()

        elif self.type == AuthType.BEARER:
            if not self.bearer_token:
                raise ValueError("Bearer token required for Bearer authentication")
            return BearerAuthentication(self.bearer_token)

        elif self.type == AuthType.BASIC:
            if not self.username or not self.password:
                raise ValueError(
                    "Username and password required for Basic authentication"
                )
            return BasicAuthentication(self.username, self.password)

        elif self.type == AuthType.API_KEY:
            if not self.api_key:
                raise ValueError("API key required for API Key authentication")
            return ApiKeyAuthentication(
                self.api_key, self.api_key_location, self.api_key_name
            )

        elif self.type == AuthType.OAUTH2:
            if not all(
                [
                    self.oauth2_token_url,
                    self.oauth2_client_id,
                    self.oauth2_client_secret,
                ]
            ):
                raise ValueError(
                    "OAuth2 credentials required for OAuth2 authentication"
                )
            return OAuth2Authentication(
                self.oauth2_token_url,  # type: ignore
                self.oauth2_client_id,  # type: ignore
                self.oauth2_client_secret,  # type: ignore
                self.oauth2_grant_type,
                self.oauth2_scope,
                self.oauth2_refresh_token,
            )

        elif self.type == AuthType.HMAC:
            if not self.hmac_secret_key:
                raise ValueError("HMAC secret key required for HMAC authentication")
            return HMACAuthentication(
                self.hmac_secret_key, self.hmac_algorithm, self.hmac_header_name
            )

        else:
            return NoAuthentication()
