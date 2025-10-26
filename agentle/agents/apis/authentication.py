"""
Authentication support for API endpoints.

Provides various authentication methods including Bearer, Basic, OAuth2, API Key, and custom schemes.
"""

# Import and re-export for backward compatibility
from agentle.agents.apis.api_key_location import ApiKeyLocation
from agentle.agents.apis.api_key_authentication import ApiKeyAuthentication
from agentle.agents.apis.auth_type import AuthType
from agentle.agents.apis.authentication_base import AuthenticationBase
from agentle.agents.apis.authentication_config import AuthenticationConfig
from agentle.agents.apis.basic_authentication import BasicAuthentication
from agentle.agents.apis.bearer_authentication import BearerAuthentication
from agentle.agents.apis.hmac_authentication import HMACAuthentication
from agentle.agents.apis.no_authentication import NoAuthentication
from agentle.agents.apis.oauth2_authentication import OAuth2Authentication
from agentle.agents.apis.oauth2_grant_type import OAuth2GrantType

__all__ = [
    "AuthType",
    "ApiKeyLocation",
    "OAuth2GrantType",
    "AuthenticationBase",
    "NoAuthentication",
    "BearerAuthentication",
    "BasicAuthentication",
    "ApiKeyAuthentication",
    "OAuth2Authentication",
    "HMACAuthentication",
    "AuthenticationConfig",
]
