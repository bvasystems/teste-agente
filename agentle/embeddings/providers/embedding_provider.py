import abc

from rsb.coroutines.run_sync import run_sync

from agentle.embeddings.models.embed_content import EmbedContent


class EmbeddingProvider(abc.ABC):
    def generate_embeddings(self, contents: str) -> EmbedContent:
        return run_sync(self.generate_embeddings_async, contents=contents)

    @abc.abstractmethod
    async def generate_embeddings_async(self, contents: str) -> EmbedContent: ...
