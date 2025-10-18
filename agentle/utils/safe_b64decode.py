"""Utility function for safe base64 decoding.

This module provides a utility to safely decode base64 data with automatic
padding fix, handling common encoding issues that can occur during transmission
or storage.
"""

import base64

from agentle.utils.fix_base64_padding import fix_base64_padding


def safe_b64decode(data: str) -> bytes:
    """
    Safely decode base64 data with automatic padding fix.

    This function attempts to decode base64 data, and if it fails due to
    padding issues, it will automatically fix the padding and try again.

    Args:
        data: Base64 encoded string to decode

    Returns:
        Decoded bytes

    Raises:
        ValueError: If the data cannot be decoded even after padding fix

    Example:
        >>> safe_b64decode("SGVsbG8gV29ybGQ")
        b'Hello World'
    """
    try:
        # Try to decode as-is first
        return base64.b64decode(data)
    except Exception:
        # If that fails, try fixing the padding
        try:
            fixed_data = fix_base64_padding(data)
            return base64.b64decode(fixed_data)
        except Exception as e:
            raise ValueError(
                f"Failed to decode base64 data: {e}. Data might be corrupted or not properly encoded."
            ) from e
