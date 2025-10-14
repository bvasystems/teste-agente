from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, Literal, override
import uuid

from agentle.embeddings.models.embed_content import EmbedContent
from agentle.embeddings.models.embedding import Embedding
from agentle.embeddings.providers.embedding_provider import EmbeddingProvider
from google import genai

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai import types
    from google.genai.client import (
        DebugConfig,
    )


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model: str = "gemini-embedding-001",
        task_type: Literal[
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY",
        ] = "RETRIEVAL_DOCUMENT",
        title: str | None = None,
        output_dimensionality: int | None = None,
        vertexai: bool = False,
        api_key: str | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: str | None = None,
        debug_config: DebugConfig | None = None,
        http_options: types.HttpOptions | None = None,
        extract_metadata: bool = True,
        config: types.EmbedContentConfig | types.EmbedContentConfigDict | None = None,
    ) -> None:
        self.task_type = task_type
        self.title = title
        self.output_dimensionality = output_dimensionality
        self.model = model
        self._client = genai.Client(
            vertexai=vertexai,
            api_key=api_key if vertexai is False else None,
            credentials=credentials,
            project=project,
            location=location,
            debug_config=debug_config,
            http_options=http_options,
        )
        self.config = config

    @override
    async def generate_embeddings_async(
        self,
        contents: str,
        metadata: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> EmbedContent:
        embeddings = await self._client.aio.models.embed_content(
            model=self.model, contents=contents, config=self.config
        )

        content_embeddings = embeddings.embeddings
        if content_embeddings is None:
            raise ValueError("Provided content embeddings is None.")

        vectors: MutableSequence[float] = []

        for content_embedding in content_embeddings:
            if not content_embedding.values:
                raise ValueError(
                    "ERROR: No values found in content_embedding.values. "
                    + f"Content embeddings: {content_embeddings}"
                )

            vectors = content_embedding.values

        return EmbedContent(
            embeddings=Embedding(
                id=id or str(uuid.uuid4()),
                value=vectors,
                original_text=contents,
                metadata=metadata or {},
            )
        )

    @override
    async def generate_batch_embeddings_async(
        self,
        contents: Sequence[str],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        ids: Sequence[str | None] | None = None,
    ) -> Sequence[EmbedContent]:
        """Generate embeddings for multiple texts using Google's native batch API.

        This method uses Google's embed_content API with multiple contents,
        which is more efficient than making individual API calls.

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

        # Use Google's native batch API - pass list of contents
        embeddings_response = await self._client.aio.models.embed_content(
            model=self.model, contents=list(contents), config=self.config
        )

        content_embeddings = embeddings_response.embeddings
        if content_embeddings is None:
            raise ValueError("Provided content embeddings is None.")

        # Process each embedding and pair with metadata/id
        results: MutableSequence[EmbedContent] = []
        for i, content_embedding in enumerate(content_embeddings):
            if not content_embedding.values:
                raise ValueError(
                    f"ERROR: No values found in content_embedding at index {i}. "
                    + f"Content embeddings: {content_embeddings}"
                )

            vectors = content_embedding.values
            results.append(
                EmbedContent(
                    embeddings=Embedding(
                        id=ids_list[i] or str(uuid.uuid4()),
                        value=vectors,
                        original_text=contents[i],
                        metadata=metadata_list[i] or {},
                    )
                )
            )

        return results


if __name__ == "__main__":
    provider = GoogleEmbeddingProvider(
        vertexai=True, project="unicortex", location="global"
    )
    embed_content = provider.generate_embeddings("oi tudo bem?")
    print(embed_content)
    print(len(embed_content.embeddings.value))  # 3072
