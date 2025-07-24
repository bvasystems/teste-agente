from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.vector_stores.filters.any_variants import AnyVariants

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Match as QdrantMatch


class MatchExcept(BaseModel):
    """
    Should have at least one value not matching the any given values
    """

    except_: AnyVariants = Field(
        ...,
        description="Should have at least one value not matching the any given values",
        alias="except",
    )

    def to_qdrant_match(self) -> QdrantMatch:
        from qdrant_client.http.models.models import MatchExcept as QdrantMatchExcept

        return QdrantMatchExcept.model_validate({"except": self.except_})
