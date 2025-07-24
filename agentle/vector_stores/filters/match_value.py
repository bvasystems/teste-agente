from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.vector_stores.filters.value_variants import ValueVariants

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Match as QdrantMatch


class MatchValue(BaseModel):
    """
    Exact match of the given value
    """

    value: ValueVariants = Field(..., description="Exact match of the given value")

    def to_qdrant_match(self) -> QdrantMatch:
        from qdrant_client.http.models.models import MatchValue as QdrantMatchValue

        return QdrantMatchValue(value=self.value)
