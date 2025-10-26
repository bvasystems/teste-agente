"""Authentication types."""

from enum import StrEnum


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
