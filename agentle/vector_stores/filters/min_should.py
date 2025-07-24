from __future__ import annotations
from typing import TYPE_CHECKING

from agentle.vector_stores.filters.condition import Condition
from rsb.models.base_model import BaseModel

if TYPE_CHECKING:
    from qdrant_client.http.models.models import MinShould as QdrantMinShould


class MinShould(BaseModel):
    conditions: list[Condition]
    min_count: int

    def to_qdrant_min_should(self) -> QdrantMinShould:
        from qdrant_client.http.models.models import MinShould as QdrantMinShould

        return QdrantMinShould(
            conditions=[c.to_qdrant_condition() for c in self.conditions],
            min_count=self.min_count,
        )
