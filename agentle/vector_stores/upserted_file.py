from __future__ import annotations

from collections.abc import Sequence

from rsb.models.base_model import BaseModel


class UpsertedFile(BaseModel):
    chunk_ids: Sequence[str]

    def __add__(self, other: UpsertedFile) -> UpsertedFile:
        return UpsertedFile(chunk_ids=list(self.chunk_ids) + list(other.chunk_ids))

    def __radd__(self, other: int | UpsertedFile) -> UpsertedFile:
        if other == 0:  # Handle sum()'s default start value
            return self
        elif isinstance(other, UpsertedFile):
            return UpsertedFile(chunk_ids=list(other.chunk_ids) + list(self.chunk_ids))
        else:
            return NotImplemented
