"""Cache strategies."""

from enum import StrEnum


class CacheStrategy(StrEnum):
    """Cache strategies."""

    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    CUSTOM = "custom"
