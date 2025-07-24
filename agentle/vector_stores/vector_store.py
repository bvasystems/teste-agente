import abc
from collections.abc import MutableSequence, Sequence

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
from agentle.vector_stores.collection import Collection
from agentle.vector_stores.create_collection_config import CreateCollectionConfig
from agentle.vector_stores.filters.field_condition import FieldCondition
from agentle.vector_stores.filters.filter import Filter
from agentle.vector_stores.filters.match_value import MatchValue

type ChunkID = str


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

    def find_related_content(
        self,
        query: str | Embedding | Sequence[float] | None = None,
        *,
        filter: Filter | None = None,
        k: int = 10,
        collection_name: str | None = None,
    ) -> Sequence[Chunk]:
        return run_sync(
            self.find_related_content_async,
            query=query,
            filter=filter,
            k=k,
            collection_name=collection_name,
        )

    async def find_related_content_async(
        self,
        query: str | Embedding | Sequence[float] | None = None,
        *,
        filter: Filter | None = None,
        k: int = 10,
        collection_name: str | None = None,
    ) -> Sequence[Chunk]:
        if query and filter is None:
            raise ValueError("Either query or filter must be provided.")

        if query:
            match query:
                case str():
                    embedding = await self.embedding_provider.generate_embeddings_async(
                        contents=query
                    )

                    return await self._find_related_content_async(
                        query=embedding.embeddings.value,
                        k=k,
                        filter=filter,
                        collection_name=collection_name,
                    )
                case Embedding():
                    return await self._find_related_content_async(
                        query=query.value,
                        k=k,
                        filter=filter,
                        collection_name=collection_name,
                    )
                case Sequence():
                    return await self._find_related_content_async(
                        query=query,
                        filter=filter,
                        k=k,
                        collection_name=collection_name,
                    )
        return await self._find_related_content_async(
            query=None, filter=filter, k=k, collection_name=collection_name
        )

    @abc.abstractmethod
    async def _find_related_content_async(
        self,
        query: Sequence[float] | None = None,
        *,
        filter: Filter | None = None,
        k: int = 10,
        collection_name: str | None = None,
    ) -> Sequence[Chunk]: ...

    def delete_vectors(
        self,
        collection_name: str,
        ids: Sequence[str] | None = None,
        filter: Filter | None = None,
    ) -> None:
        return run_sync(
            self.delete_vectors_async,
            collection_name=collection_name,
            ids=ids,
            filter=filter,
        )

    async def delete_vectors_async(
        self,
        collection_name: str,
        ids: Sequence[str] | None = None,
        filter: Filter | None = None,
    ) -> None:
        if filter:
            extra_ids = await self.find_related_content_async(filter=filter)
            _ids = list(list(ids or []) + list([c.id for c in extra_ids]))

            await self._delete_vectors_async(
                collection_name=collection_name, ids=list(set(_ids))
            )

    @abc.abstractmethod
    async def _delete_vectors_async(
        self,
        collection_name: str,
        ids: Sequence[str],
    ) -> None: ...

    def upsert(
        self,
        points: Embedding | Sequence[float],
        *,
        timeout: float | None = None,
        collection_name: str | None = None,
    ) -> None:
        return run_sync(
            self.upsert_async,
            points=points,
            timeout=timeout,
            collection_name=collection_name,
        )

    async def upsert_async(
        self,
        points: Embedding | Sequence[float],
        *,
        collection_name: str | None = None,
    ) -> None:
        if len(points) == 0:
            return None

        if isinstance(points, Sequence):
            return await self._upsert_async(
                points=Embedding(value=points),
                collection_name=collection_name,
            )

        return await self._upsert_async(
            points=points,
            collection_name=collection_name,
        )

    @abc.abstractmethod
    async def _upsert_async(
        self,
        points: Embedding,
        *,
        collection_name: str | None = None,
    ) -> None: ...

    def upsert_file(
        self,
        file: ParsedFile,
        *,
        timeout: float | None = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        chunking_config: ChunkingConfig | None = None,
        collection_name: str | None = None,
        override_if_exists: bool = False,
    ) -> Sequence[ChunkID]:
        return run_sync(
            self.upsert_file_async,
            file=file,
            timeout=timeout,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            collection_name=collection_name,
            override_if_exists=override_if_exists,
        )

    async def upsert_file_async(
        self,
        file: ParsedFile,
        *,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        chunking_config: ChunkingConfig | None = None,
        collection_name: str | None = None,
        override_if_exists: bool = False,
    ) -> Sequence[ChunkID]:
        # Check if file was already ingested in the database.
        possible_file_chunks = self.find_related_content(
            collection_name=collection_name or self.default_collection_name,
            filter=Filter(
                must=FieldCondition(
                    key="source_document_id", match=MatchValue(value=file.unique_id)
                )
            ),
        )

        file_exists = len(possible_file_chunks) > 0

        if file_exists:
            if not override_if_exists:
                raise ValueError("The provided file already exists in the database")
            else:
                await self.delete_vectors_async(
                    collection_name=collection_name or self.default_collection_name,
                    ids=[p.id for p in possible_file_chunks],
                )

        chunks: Sequence[Chunk] = await file.chunkify_async(
            strategy=chunking_strategy, config=chunking_config
        )

        embed_contents: Sequence[EmbedContent] = [
            await self.embedding_provider.generate_embeddings_async(
                c.text, metadata=c.metadata, id=c.id
            )
            for c in chunks
        ]

        ids: MutableSequence[str] = []

        for e in embed_contents:
            await self.upsert_async(
                points=e.embeddings,
                collection_name=collection_name,
            )

            ids.append(e.embeddings.id)

        print(f"Chunk ids: {[c.id for c in chunks]}")

        return ids

    def create_collection(
        self, collection_name: str, *, config: CreateCollectionConfig
    ) -> None:
        return run_sync(
            self.create_collection_async, collection_name=collection_name, config=config
        )

    @abc.abstractmethod
    async def create_collection_async(
        self, collection_name: str, *, config: CreateCollectionConfig
    ) -> None: ...

    def delete_collection(self, collection_name: str) -> None:
        return run_sync(self.delete_collection_async, collection_name=collection_name)

    @abc.abstractmethod
    async def delete_collection_async(self, collection_name: str) -> None: ...

    def list_collections(self) -> Sequence[Collection]:
        return run_sync(self.list_collections_async)

    @abc.abstractmethod
    async def list_collections_async(self) -> Sequence[Collection]: ...

    def as_search_tool(self) -> Tool[Sequence[Chunk]]:
        async def search_async(query: str, *, top_k: int = 3) -> Sequence[Chunk]:
            return await self.find_related_content_async(query=query, k=top_k)

        return Tool.from_callable(search_async)
