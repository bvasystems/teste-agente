"""Utility functions for fixing base64 padding issues.

This module provides utilities to handle common base64 encoding/decoding issues,
particularly padding problems that can occur when base64 data is transmitted
or stored without proper padding characters.
"""

import base64


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
        >>> base64.b64decode(fix_base64_padding("SGVsbG8gV29ybGQ"))
        b'Hello World'
    """
    # Remove any whitespace
    data = data.strip()
    # Add padding if needed
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    return data


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