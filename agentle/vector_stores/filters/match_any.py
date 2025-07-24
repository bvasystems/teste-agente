from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.vector_stores.filters.any_variants import AnyVariants

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Match as QdrantMatch


class MatchAny(BaseModel):
    """
    Exact match on any of the given values
    """

    any: AnyVariants = Field(..., description="Exact match on any of the given values")

    def to_qdrant_match(self) -> QdrantMatch:
        from qdrant_client.http.models.models import MatchAny as QdrantMatchAny

        return QdrantMatchAny(any=self.any)
