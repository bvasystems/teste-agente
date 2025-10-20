from typing import Literal
from rsb.models import BaseModel, Field


class ExecuteJavascript(BaseModel):
    type: Literal["execute_javascript"] = Field(
        default="execute_javascript",
        description="Execute a JavaScript code on the current page.",
    )

    script: str = Field(
        ...,
        description="The JavaScript code to execute.",
        examples=["document.querySelector('.button').click();"],
    )
