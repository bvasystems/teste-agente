from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any, Literal, override

import httpx

from agentle.embeddings.models.embed_content import EmbedContent
from agentle.embeddings.models.embedding import Embedding
from agentle.embeddings.providers.embedding_provider import EmbeddingProvider


class DeepinfraEmbeddingProvider(EmbeddingProvider):
    """Deepinfra embedding provider using their OpenAI-compatible API.

    Args:
        model: Model identifier (e.g., "Qwen/Qwen3-Embedding-8B")
        api_key: Deepinfra API token. If not provided, reads from DEEPINFRA_API_KEY env var
        encoding_format: Format for encoding embeddings (default: "float")
        dimensions: Number of dimensions in the embedding. If not provided, uses model default
        service_tier: Service tier for processing ("default" or "priority")
        base_url: Base URL for the Deepinfra API
        timeout: Timeout for API requests in seconds
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-Embedding-8B",
        api_key: str | None = None,
        encoding_format: Literal["float"] = "float",
        dimensions: int | None = None,
        service_tier: Literal["default", "priority"] = "default",
        base_url: str = "https://api.deepinfra.com/v1/openai",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepinfra API key must be provided either via 'api_key' parameter "
                + "or 'DEEPINFRA_API_KEY' environment variable"
            )
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.service_tier = service_tier
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    def __del__(self) -> None:
        """Cleanup async client on deletion."""
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._client.aclose())
            else:
                asyncio.run(self._client.aclose())
        except Exception:
            # Ignore cleanup errors
            pass

    @override
    async def generate_embeddings_async(
        self,
        contents: str,
        metadata: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> EmbedContent:
        """Generate embeddings for a single text string.

        Args:
            contents: Text string to generate embeddings for
            metadata: Optional metadata to attach to the embedding
            id: Optional ID for the embedding

        Returns:
            EmbedContent object containing the embedding
        """
        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "input": contents,
            "encoding_format": self.encoding_format,
        }

        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        if self.service_tier != "default":
            payload["service_tier"] = self.service_tier

        # Make API request
        response = await self._client.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        response.raise_for_status()

        result = response.json()

        # Extract embeddings from response
        if "data" not in result or not result["data"]:
            raise ValueError(f"No embedding data in response: {result}")

        embedding_data = result["data"][0]
        vectors = embedding_data["embedding"]

        return EmbedContent(
            embeddings=Embedding(
                id=id if id is not None else str(uuid.uuid4()),
                value=vectors,
                original_text=contents,
                metadata=dict(metadata) if metadata is not None else {},
            )
        )

    @override
    async def generate_batch_embeddings_async(
        self,
        contents: Sequence[str],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        ids: Sequence[str | None] | None = None,
    ) -> Sequence[EmbedContent]:
        """Generate embeddings for multiple texts using Deepinfra's batch API.

        This method uses Deepinfra's native batch API by passing an array of
        strings to the input field, which is more efficient than making
        individual API calls.

        Args:
            contents: A sequence of text strings to generate embeddings for
            metadata: Optional sequence of metadata dicts, one per content item
            ids: Optional sequence of IDs, one per content item

        Returns:
            A sequence of EmbedContent objects, one per input content
        """
        # Prepare metadata and ids lists with proper defaults
        metadata_list: Sequence[Mapping[str, Any] | None] = (
            metadata if metadata else [None] * len(contents)
        )
        ids_list: Sequence[str | None] = ids if ids else [None] * len(contents)

        # Build request payload with batch input
        payload: dict[str, Any] = {
            "model": self.model,
            "input": list(contents),
            "encoding_format": self.encoding_format,
        }

        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        if self.service_tier != "default":
            payload["service_tier"] = self.service_tier

        # Make API request
        response = await self._client.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        response.raise_for_status()

        result = response.json()

        # Extract embeddings from response
        if "data" not in result or not result["data"]:
            raise ValueError(f"No embedding data in response: {result}")

        # Process each embedding and pair with metadata/id
        results: MutableSequence[EmbedContent] = []
        for embedding_data in result["data"]:
            index: int = embedding_data["index"]
            vectors: list[float] = embedding_data["embedding"]

            current_id: str | None = ids_list[index]
            current_metadata: Mapping[str, Any] | None = metadata_list[index]
            current_text: str = contents[index]

            results.append(
                EmbedContent(
                    embeddings=Embedding(
                        id=current_id if current_id is not None else str(uuid.uuid4()),
                        value=vectors,
                        original_text=current_text,
                        metadata=dict(current_metadata)
                        if current_metadata is not None
                        else {},
                    )
                )
            )

        return results


if __name__ == "__main__":
    # Example usage
    provider = DeepinfraEmbeddingProvider(
        model="Qwen/Qwen3-Embedding-8B",
        # api_key will be read from DEEPINFRA_API_KEY environment variable
    )

    # Single embedding
    embed_content = provider.generate_embeddings(
        "The food was delicious and the waiter..."
    )
    print(embed_content)
    print(f"Embedding dimensions: {len(embed_content.embeddings.value)}")

    # Batch embeddings
    batch_contents = [
        "The food was delicious and the waiter...",
        "Hello world!",
        "Machine learning is fascinating.",
    ]
    batch_results = provider.generate_batch_embeddings(batch_contents)
    print(f"\nBatch results: {len(batch_results)} embeddings generated")
    for i, result in enumerate(batch_results):
        print(f"  {i}: {len(result.embeddings.value)} dimensions")
