from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.vector_stores.filters.condition import Condition
from agentle.vector_stores.filters.min_should import MinShould


class Filter(BaseModel):
    should: Condition | list[Condition] | None = Field(default=None)
    min_should: MinShould | None = Field(default=None)
    must: Condition | list[Condition] | None = Field(default=None)
    must_not: Condition | list[Condition] | None = Field(default=None)
