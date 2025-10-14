import abc
import asyncio
from collections.abc import Mapping, Sequence
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

    def generate_batch_embeddings(
        self,
        contents: Sequence[str],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        ids: Sequence[str | None] | None = None,
    ) -> Sequence[EmbedContent]:
        """Generate embeddings for multiple texts in batch.

        Args:
            contents: A sequence of text strings to generate embeddings for
            metadata: Optional sequence of metadata dicts, one per content item
            ids: Optional sequence of IDs, one per content item

        Returns:
            A sequence of EmbedContent objects, one per input content
        """
        return run_sync(
            self.generate_batch_embeddings_async,
            contents=contents,
            metadata=metadata,
            ids=ids,
        )

    async def generate_batch_embeddings_async(
        self,
        contents: Sequence[str],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        ids: Sequence[str | None] | None = None,
    ) -> Sequence[EmbedContent]:
        """Generate embeddings for multiple texts in batch (async).

        Default implementation uses asyncio.gather for parallel processing.
        Subclasses that support native batch APIs should override this method.

        Args:
            contents: A sequence of text strings to generate embeddings for
            metadata: Optional sequence of metadata dicts, one per content item
            ids: Optional sequence of IDs, one per content item

        Returns:
            A sequence of EmbedContent objects, one per input content
        """
        # Prepare metadata and ids lists with proper defaults
        metadata_list = metadata if metadata else [None] * len(contents)
        ids_list = ids if ids else [None] * len(contents)

        # Use asyncio.gather for parallel processing
        tasks = [
            self.generate_embeddings_async(content, metadata=meta, id=id_val)
            for content, meta, id_val in zip(contents, metadata_list, ids_list)
        ]

        return await asyncio.gather(*tasks)
