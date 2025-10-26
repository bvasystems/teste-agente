"""Retry strategies."""

from enum import StrEnum


class RetryStrategy(StrEnum):
    """Retry strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"
