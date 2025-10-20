from typing import Literal
from rsb.models import BaseModel, Field


class Click(BaseModel):
    type: Literal["click"] = Field(default="click")
    selector: str = Field(
        ...,
        description="Query selector to find the element by",
        examples=["#load-more-button"],
    )
    all: bool = Field(
        default=False,
        description="Clicks all elements matched by the selector, not just the first one. Does not throw an error if no elements match the selector.",
    )
