from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Pricing(BaseModel):
    input_pricing: float | None = Field(default=None)
    output_pricing: float | None = Field(default=None)
    total_pricing: float | None = Field(default=None)
