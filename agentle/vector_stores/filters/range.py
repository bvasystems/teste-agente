from __future__ import annotations
from typing import TYPE_CHECKING

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

if TYPE_CHECKING:
    from qdrant_client.http.models.models import Range as QdrantRange


class Range(BaseModel):
    """
    Range filter request
    """

    lt: float | None = Field(default=None, description="point.key &lt; range.lt")
    gt: float | None = Field(default=None, description="point.key &gt; range.gt")
    gte: float | None = Field(default=None, description="point.key &gt;= range.gte")
    lte: float | None = Field(default=None, description="point.key &lt;= range.lte")

    def to_qdrant_range(self) -> QdrantRange:
        from qdrant_client.http.models.models import Range as QdrantRange

        return QdrantRange(lt=self.lt, gt=self.gt, gte=self.gte, lte=self.lte)
