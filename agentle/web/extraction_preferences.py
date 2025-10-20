from collections.abc import Sequence
from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.web.actions.action import Action
from agentle.web.location import Location


class ExtractionPreferences(BaseModel):
    only_main_content: bool = Field(default=True)
    include_tags: Sequence[str] | None = Field(default=None)
    exclude_tags: Sequence[str] | None = Field(default=None)
    headers: dict[str, str] | None = Field(default=None)
    wait_for_ms: float | None = Field(default=None)
    mobile: bool | None = Field(default=None)
    skip_tls_verification: bool = Field(default=True)
    timeout_ms: float | None = Field(default=None)
    actions: Sequence[Action] | None = Field(default=None)
    location: Location | None = Field(default=None)
    remove_base_64_images: bool = Field(default=True)
    block_ads: bool = Field(default=True)
    proxy: Literal["basic", "stealth", "auto"] = Field(default="auto")
