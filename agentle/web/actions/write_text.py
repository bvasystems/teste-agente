from typing import Literal
from rsb.models import BaseModel, Field


class WriteText(BaseModel):
    type: Literal["write"] = Field(
        default="write",
        description="Write text into an input field, text area, or contenteditable element. Note: You must first focus the element using a 'click' action before writing. The text will be typed character by character to simulate keyboard input.",
    )
    text: str = Field(
        ...,
        description="Text to write into the element.",
        examples=["Hello, world!"],
    )
