from typing import Literal
from rsb.models import BaseModel, Field


class GeneratePdf(BaseModel):
    type: Literal["pdf"] = Field(
        default="pdf",
        description="Generate a PDF of the current page. The PDF will be returned in the actions.pdfs array of the response.",
    )
    format: Literal[
        "A0", "A1", "A2", "A3", "A4", "A5", "A6", "Letter", "Legal", "Tabloid", "Ledger"
    ] = Field(
        default="Letter",
        description="The format of the PDF to generate.",
        examples=[
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "Letter",
            "Legal",
            "Tabloid",
            "Ledger",
        ],
    )

    landscape: bool = Field(
        default=False,
        description="Whether to generate the PDF in landscape orientation.",
    )

    scale: float = Field(
        default=1.0,
        description="The scale multiplier of the resulting PDF.",
        ge=0.1,
        le=10.0,
        examples=[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
