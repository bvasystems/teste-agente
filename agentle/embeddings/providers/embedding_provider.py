import abc
from collections.abc import Mapping
from typing import Any

from rsb.coroutines.run_sync import run_sync

from agentle.embeddings.models.embed_content import EmbedContent


class EmbeddingProvider(abc.ABC):
    def generate_embeddings(
        self,
        contents: str,
        metadata: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> EmbedContent:
        return run_sync(
            self.generate_embeddings_async, contents=contents, metadata=metadata, id=id
        )

    @abc.abstractmethod
    async def generate_embeddings_async(
        self,
        contents: str,
        metadata: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> EmbedContent: ...
