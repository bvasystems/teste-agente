"""API key location types."""

from enum import StrEnum


class ApiKeyLocation(StrEnum):
    """Where to place API key."""

    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"
