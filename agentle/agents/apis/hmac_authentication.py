"""HMAC signature authentication."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import MutableMapping
from typing import Any

import aiohttp
from agentle.agents.apis.authentication_base import AuthenticationBase


class HMACAuthentication(AuthenticationBase):
    """HMAC signature authentication."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "sha256",
        header_name: str = "X-Signature",
        include_timestamp: bool = True,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.header_name = header_name
        self.include_timestamp = include_timestamp

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """Add HMAC signature to headers."""
        # Build signature string
        timestamp = str(int(time.time()))
        signature_string = url

        if self.include_timestamp:
            signature_string = f"{timestamp}:{signature_string}"
            headers["X-Timestamp"] = timestamp

        # Calculate HMAC
        hash_func = getattr(hashlib, self.algorithm)
        signature = hmac.new(
            self.secret_key.encode(), signature_string.encode(), hash_func
        ).hexdigest()

        headers[self.header_name] = signature

    async def refresh_if_needed(self) -> bool:
        """No refresh needed for HMAC."""
        return False
