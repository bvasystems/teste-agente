from typing import Literal
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

class CacheControlType(BaseModel):
    type: Literal["ephemeral"] = Field(default="ephemeral")
