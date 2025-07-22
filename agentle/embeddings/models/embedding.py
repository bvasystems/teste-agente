from collections.abc import Mapping, Sequence
from typing import Any
import uuid

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Embedding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4))
    value: Sequence[float]
    original_text: str | None = Field(default=None)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """Returns (num_embeddings, embedding_dim)"""
        return len(self.value), len(self.value) if self.value else 0

    def __len__(self) -> int:
        return len(self.value)
