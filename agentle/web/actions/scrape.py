from typing import Literal
from rsb.models import BaseModel, Field


class Scrape(BaseModel):
    type: Literal["scrape"] = Field(
        default="scrape",
        description="Scrape the current page content, returns the url and the html.",
    )
