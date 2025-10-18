"""Utility function for fixing base64 padding issues.

This module provides a utility to handle common base64 encoding/decoding issues,
particularly padding problems that can occur when base64 data is transmitted
or stored without proper padding characters.
"""


def fix_base64_padding(data: str) -> str:
    """
    Fix base64 padding by adding missing padding characters.

    Base64 strings must be a multiple of 4 characters in length. If padding
    characters ('=') are missing, this function will add them to make the
    string valid for decoding.

    Args:
        data: Base64 encoded string that may have incorrect padding

    Returns:
        Base64 string with correct padding

    Example:
        >>> fix_base64_padding("SGVsbG8gV29ybGQ")
        'SGVsbG8gV29ybGQ='
        >>> import base64
        >>> base64.b64decode(fix_base64_padding("SGVsbG8gV29ybGQ"))
        b'Hello World'
    """
    # Remove any whitespace
    data = data.strip()
    # Add padding if needed
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    return data
