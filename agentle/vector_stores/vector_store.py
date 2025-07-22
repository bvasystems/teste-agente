import abc
from collections.abc import Mapping, Sequence
from typing import Any

from rsb.coroutines.run_sync import run_sync

from agentle.embeddings.models.embed_content import EmbedContent
from agentle.embeddings.models.embedding import Embedding
from agentle.embeddings.providers.embedding_provider import EmbeddingProvider
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.tools.tool import Tool
from agentle.parsing.chunk import Chunk
from agentle.parsing.chunking.chunking_config import ChunkingConfig
from agentle.parsing.chunking.chunking_strategy import ChunkingStrategy
from agentle.parsing.parsed_file import ParsedFile
from agentle.vector_stores.upserted_file import UpsertedFile


class VectorStore(abc.ABC):
    default_collection_name: str
    embedding_provider: EmbeddingProvider
    generation_provider: GenerationProvider | None

    def __init__(
        self,
        *,
        default_collection_name: str = "agentle",
        embedding_provider: EmbeddingProvider,
        generation_provider: GenerationProvider | None,
    ) -> None:
        self.default_collection_name = default_collection_name
        self.embedding_provider = embedding_provider
        self.generation_provider = generation_provider

    async def find_related_content(
        self, query: str | Sequence[Embedding] | Sequence[float], *, k: int = 10
    ) -> Sequence[Chunk]: ...

    @abc.abstractmethod
    async def _find_related_content(
        self, query: Sequence[float], *, k: int = 10
    ) -> Sequence[Chunk]: ...

    async def upsert(
        self,
        points: Embedding | Sequence[float],
        *,
        timeout: float | None = None,
        collection_name: str | None = None,
        points_metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> UpsertedFile:
        return run_sync(
            self.upsert_async,
            points=points,
            timeout=timeout,
            collection_name=collection_name,
            points_metadatas=points_metadatas,
        )

    @abc.abstractmethod
    async def upsert_async(
        self,
        points: Embedding | Sequence[float],
        *,
        timeout: float | None = None,
        collection_name: str | None = None,
        points_metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> UpsertedFile:
        if len(points) == 0:
            return UpsertedFile(chunk_ids=[])

        if isinstance(points, Sequence):
            return await self._upsert_async(
                points=Embedding(value=points),
                timeout=timeout,
                collection_name=collection_name,
                points_metadatas=points_metadatas,
            )

        return await self._upsert_async(
            points=points,
            timeout=timeout,
            collection_name=collection_name,
            points_metadatas=points_metadatas,
        )

    @abc.abstractmethod
    async def _upsert_async(
        self,
        points: Embedding,
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
        chunks: Sequence[Chunk] = await file.chunkify_async(
            strategy=chunking_strategy, config=chunking_config
        )

        embed_contents: Sequence[EmbedContent] = [
            await self.embedding_provider.generate_embeddings_async(
                c.text, metadata=c.metadata
            )
            for c in chunks
        ]

        upserted_chunks = [
            await self.upsert_async(
                points=e.embeddings,
                collection_name=collection_name,
                points_metadatas=[c.metadata for c in chunks],
            )
            for e in embed_contents
        ]

        return sum(upserted_chunks) or UpsertedFile(chunk_ids=[])

    def as_tool(self) -> Tool[Sequence[Chunk]]: ...  # TODO(arthur)
