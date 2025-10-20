from typing import Literal

from rsb.models import BaseModel, Field


class Wait(BaseModel):
    type: Literal["wait"]
    milliseconds: int = Field(..., description="Number of milliseconds to wait")
    selector: str = Field(..., description="Query selector to find the element by")
