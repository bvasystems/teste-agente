from rsb.models.base_model import BaseModel

from agentle.embeddings.models.embeddings import Embeddings


class EmbedContent(BaseModel):
    embeddings: Embeddings
