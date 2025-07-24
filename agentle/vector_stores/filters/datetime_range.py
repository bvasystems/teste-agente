from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

if TYPE_CHECKING:
    from qdrant_client.http.models.models import DatetimeRange as QdrantDatetimeRange


class DatetimeRange(BaseModel):
    """
    Range filter request
    """

    lt: datetime | date | None = Field(
        default=None, description="point.key &lt; range.lt"
    )
    gt: datetime | date | None = Field(
        default=None, description="point.key &gt; range.gt"
    )
    gte: datetime | date | None = Field(
        default=None, description="point.key &gt;= range.gte"
    )
    lte: datetime | date | None = Field(
        default=None, description="point.key &lt;= range.lte"
    )

    def to_qdrant_range(self) -> QdrantDatetimeRange:
        from qdrant_client.http.models.models import (
            DatetimeRange as QdrantDatetimeRange,
        )

        return QdrantDatetimeRange(lt=self.lt, gt=self.gt, gte=self.gte, lte=self.lte)
