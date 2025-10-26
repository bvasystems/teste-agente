"""No authentication handler."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import aiohttp
from agentle.agents.apis.authentication_base import AuthenticationBase


class NoAuthentication(AuthenticationBase):
    """No authentication."""

    async def apply_auth(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: MutableMapping[str, str],
        params: MutableMapping[str, Any],
    ) -> None:
        """No authentication to apply."""
        pass

    async def refresh_if_needed(self) -> bool:
        """No refresh needed."""
        return False
