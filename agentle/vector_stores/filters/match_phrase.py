from __future__ import annotations
from typing import TYPE_CHECKING

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Match as QdrantMatch


class MatchPhrase(BaseModel, extra="forbid"):
    """
    Full-text phrase match of the string.
    """

    phrase: str = Field(..., description="Full-text phrase match of the string.")

    def to_qdrant_match(self) -> QdrantMatch:
        from qdrant_client.http.models.models import MatchPhrase as QdrantMatchPhrase

        return QdrantMatchPhrase(phrase=self.phrase)
