from __future__ import annotations
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agentle.vector_stores.filters.match import Match
from agentle.vector_stores.filters.range_interface import RangeInterface

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Condition


class FieldCondition(BaseModel, extra="forbid"):
    key: str = Field(..., description="Payload key")

    match: Match | None = Field(
        default=None, description="Check if point has field with a given value"
    )

    range: RangeInterface | None = Field(
        default=None, description="Check if points value lies in a given range"
    )

    is_empty: bool | None = Field(
        default=None,
        description="Check that the field is empty, alternative syntax for `is_empty: 'field_name'`",
    )

    is_null: bool | None = Field(
        default=None,
        description="Check that the field is null, alternative syntax for `is_null: 'field_name'`",
    )

    def to_qdrant_condition(self) -> Condition:
        from qdrant_client.http.models.models import (
            FieldCondition as QdrantFieldCondition,
        )

        return QdrantFieldCondition(
            key=self.key,
            match=self.match.to_qdrant_match() if self.match else None,
            range=self.range.to_qdrant_range() if self.range else None,
            is_empty=self.is_empty,
            is_null=self.is_null,
        )
