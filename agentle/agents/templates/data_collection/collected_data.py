from collections.abc import MutableMapping, MutableSequence
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class CollectedData(BaseModel):
    """Represents the current state of collected data"""

    fields: MutableMapping[str, Any] = Field(default_factory=dict)
    pending_fields: MutableSequence[str] = Field(default_factory=list)
    completed: bool = False

    model_config = ConfigDict(frozen=False)
