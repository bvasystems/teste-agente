from enum import StrEnum


class ParameterLocation(StrEnum):
    """Where to place parameters in the HTTP request."""

    QUERY = "query"  # URL query parameters
    BODY = "body"  # Request body (JSON)
    HEADER = "header"  # HTTP headers
    PATH = "path"  # URL path parameters
