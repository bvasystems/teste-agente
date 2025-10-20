from typing import Literal

from rsb.models import BaseModel, Field


class Viewport(BaseModel):
    type: Literal["viewport"]
    width: int = Field(..., description="The width of the viewport in pixels")
    height: int = Field(..., description="The height of the viewport in pixels")
