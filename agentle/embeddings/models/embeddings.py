from collections.abc import Sequence
from rsb.models.base_model import BaseModel


class Embeddings(BaseModel):
    value: Sequence[float]

    @property
    def shape(self) -> tuple[int, int]:
        """Returns (num_embeddings, embedding_dim)"""
        return len(self.value), len(self.value) if self.value else 0

    def __len__(self) -> int:
        return len(self.value)
