from collections.abc import MutableSequence

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict


class FieldSpec(BaseModel):
    """Specification for a field to collect"""

    name: str
    type: str  # "string", "integer", "float", "boolean", "date", "email", etc.
    description: str
    required: bool = True
    validation: str | None = None  # Custom validation message
    examples: MutableSequence[str] | None = None

    model_config = ConfigDict(frozen=True)
