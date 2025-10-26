"""Authentication configuration."""

from __future__ import annotations

from collections.abc import MutableMapping

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.apis.api_key_authentication import ApiKeyAuthentication
from agentle.agents.apis.api_key_location import ApiKeyLocation
from agentle.agents.apis.auth_type import AuthType
from agentle.agents.apis.authentication_base import AuthenticationBase
from agentle.agents.apis.basic_authentication import BasicAuthentication
from agentle.agents.apis.bearer_authentication import BearerAuthentication
from agentle.agents.apis.hmac_authentication import HMACAuthentication
from agentle.agents.apis.no_authentication import NoAuthentication
from agentle.agents.apis.oauth2_authentication import OAuth2Authentication
from agentle.agents.apis.oauth2_grant_type import OAuth2GrantType


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
    oauth2_scope: str | None = Field(
        default=None,
        description="Single scope string (deprecated, use oauth2_scopes for multiple)",
    )
    oauth2_scopes: list[str] | None = Field(
        default=None,
        description="List of OAuth2 scopes to request (e.g., ['read', 'write', 'admin'])",
    )
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
                self.oauth2_scopes,
            )

        elif self.type == AuthType.HMAC:
            if not self.hmac_secret_key:
                raise ValueError("HMAC secret key required for HMAC authentication")
            return HMACAuthentication(
                self.hmac_secret_key, self.hmac_algorithm, self.hmac_header_name
            )

        else:
            return NoAuthentication()
