import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rsb.coroutines.run_sync import run_sync

from agentle.embeddings.models.embed_content import EmbedContent
from agentle.embeddings.models.embedding import Embedding
from agentle.embeddings.providers.embedding_provider import EmbeddingProvider
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.parsing.chunk import Chunk
from agentle.parsing.chunking.chunking_config import ChunkingConfig
from agentle.parsing.chunking.chunking_strategy import ChunkingStrategy
from agentle.parsing.parsed_file import ParsedFile
from agentle.vector_stores.upserted_file import UpsertedFile


@dataclass(frozen=True)
class VectorStore(abc.ABC):
    collection_name: str
    embedding_provider: EmbeddingProvider | None
    generation_provider: GenerationProvider | None

    async def upsert(
        self,
        points: Sequence[Embedding],
        *,
        timeout: float | None = None,
        collection_name: str | None = None,
        points_metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> UpsertedFile:
        return run_sync(
            self.upsert_async,
            points=points,
            collection_name=collection_name,
            points_metadatas=points_metadatas,
        )

    @abc.abstractmethod
    async def upsert_async(
        self,
        points: Sequence[Embedding],
        *,
        timeout: float | None = None,
        collection_name: str | None = None,
        points_metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> UpsertedFile: ...

    def upsert_file(
        self,
        file: ParsedFile,
        *,
        timeout: float | None = None,
        chunking_strategy: ChunkingStrategy,
        chunking_config: ChunkingConfig,
        collection_name: str | None,
    ) -> UpsertedFile:
        return run_sync(
            self.upsert_file_async,
            file=file,
            timeout=timeout,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            collection_name=collection_name,
        )

    async def upsert_file_async(
        self,
        file: ParsedFile,
        *,
        chunking_strategy: ChunkingStrategy,
        chunking_config: ChunkingConfig,
        collection_name: str | None,
    ) -> UpsertedFile:
        if self.embedding_provider is None:
            raise ValueError(
                "In instance of EmbeddingProvider is needed "
                + "to upsert files in the vector store."
            )

        chunks: Sequence[Chunk] = await file.chunkify_async(
            strategy=chunking_strategy, config=chunking_config
        )

        embed_contents: Sequence[EmbedContent] = [
            await self.embedding_provider.generate_embeddings_async(c.text)
            for c in chunks
        ]

        return await self.upsert_async(
            points=[e.embeddings for e in embed_contents],
            collection_name=collection_name,
            points_metadatas=[c.metadata for c in chunks],
        )
