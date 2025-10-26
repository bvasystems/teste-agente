"""OAuth2 grant types."""

from enum import StrEnum


class OAuth2GrantType(StrEnum):
    """OAuth2 grant types."""

    CLIENT_CREDENTIALS = "client_credentials"
    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"
    PASSWORD = "password"
