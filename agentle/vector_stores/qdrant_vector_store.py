from collections.abc import Awaitable, Callable
from typing import Any, Optional

from agentle.embeddings.providers.embedding_provider import EmbeddingProvider
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.vector_stores.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        *,
        default_collection_name: str = "agentle",
        embedding_provider: EmbeddingProvider,
        generation_provider: GenerationProvider | None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
        grpc_options: Optional[dict[str, Any]] = None,
        auth_token_provider: Optional[
            Callable[[], str] | Callable[[], Awaitable[str]]
        ] = None,
        cloud_inference: bool = False,
        local_inference_batch_size: Optional[int] = None,
        check_compatibility: bool = True,
        **kwargs: Any,
    ) -> None:
        from qdrant_client.async_qdrant_client import AsyncQdrantClient

        super().__init__(
            default_collection_name=default_collection_name,
            embedding_provider=embedding_provider,
            generation_provider=generation_provider,
        )

        self._client = AsyncQdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            force_disable_check_same_thread=force_disable_check_same_thread,
            grpc_options=grpc_options,
            auth_token_provider=auth_token_provider,
            cloud_inference=cloud_inference,
            local_inference_batch_size=local_inference_batch_size,
            check_compatibility=check_compatibility,
            **kwargs,
        )
