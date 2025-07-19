from collections.abc import Sequence
from rsb.models.base_model import BaseModel


class UpsertedFile(BaseModel):
    chunk_ids: Sequence[str]
