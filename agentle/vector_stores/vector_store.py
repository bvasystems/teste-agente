import abc
from collections.abc import Sequence

from rsb.coroutines.run_sync import run_sync

from agentle.embeddings.models.embedding import Embedding
from agentle.embeddings.providers.embedding_provider import EmbeddingProvider
from agentle.parsing.chunk import Chunk
from agentle.parsing.parsed_file import ParsedFile
from agentle.vector_stores.ingestion_result import IngestionResult


class VectorStore(abc.ABC):
    collection_name: str
    embedding_provider: EmbeddingProvider | None

    async def upsert(
        self, points: Sequence[Embedding], collection_name: str | None
    ) -> IngestionResult:
        return run_sync(
            self.upsert_async, points=points, collection_name=collection_name
        )

    @abc.abstractmethod
    async def upsert_async(
        self, points: Sequence[Embedding], collection_name: str | None
    ) -> IngestionResult: ...

    def upsert_file(
        self, file: ParsedFile, collection_name: str | None
    ) -> IngestionResult:
        return run_sync(
            self.upsert_file_async, file=file, collection_name=collection_name
        )

    async def upsert_file_async(
        self, file: ParsedFile, collection_name: str | None
    ) -> IngestionResult:
        chunks: Sequence[Chunk] = await file.chunkify_async()
        embed_content = await self.embedding_provider.generate_embeddings_async()
