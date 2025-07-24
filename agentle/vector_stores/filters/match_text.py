from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Match as QdrantMatch


class MatchText(BaseModel):
    """
    Full-text match of the strings.
    """

    text: str = Field(..., description="Full-text match of the strings.")

    def to_qdrant_match(self) -> QdrantMatch:
        from qdrant_client.http.models.models import MatchText as QdrantMatchText

        return QdrantMatchText(text=self.text)
