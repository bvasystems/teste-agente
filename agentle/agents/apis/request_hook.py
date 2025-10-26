"""Request hooks for endpoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class RequestHook(BaseModel):
    """Hook for request/response interception."""

    name: str
    callback: Callable[[dict[str, Any]], Any] | None = None
