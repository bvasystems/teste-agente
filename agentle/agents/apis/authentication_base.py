"""Base authentication handler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any

import aiohttp


class AuthenticationBase(ABC):
    """Base class for authentication handlers."""

    @abstractmethod
    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """
        Apply authentication to the request.

        Args:
            session: aiohttp session
            url: Request URL
            headers: Request headers (will be modified)
            params: Request parameters (will be modified)
        """
        pass

    @abstractmethod
    async def refresh_if_needed(self) -> bool:
        """
        Refresh authentication if needed (e.g., expired tokens).

        Returns:
            True if refresh was performed, False otherwise
        """
        pass
