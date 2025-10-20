from collections.abc import Sequence
from rsb.models import BaseModel

from agentle.web.extraction_preferences import ExtractionPreferences


class ExtractionResult[T: BaseModel](BaseModel):
    urls: Sequence[str]
    html: str
    markdown: str
    extraction_preferences: ExtractionPreferences
    result: T
