from typing import Literal
from rsb.models import BaseModel, Field


class PressAKey(BaseModel):
    type: Literal["press"] = Field(
        default="press", description="Press a key on the keyboard."
    )
    key: str = Field(
        ...,
        description="The key to press.",
        examples=["Enter", "Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"],
    )
