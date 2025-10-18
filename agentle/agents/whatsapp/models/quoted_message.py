from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.audio_message import AudioMessage
from agentle.agents.whatsapp.models.document_message import DocumentMessage
from agentle.agents.whatsapp.models.image_message import ImageMessage
from agentle.agents.whatsapp.models.video_message import VideoMessage


class QuotedMessage(BaseModel):
    """Mensagem citada/respondida no WhatsApp.

    Attributes:
        conversation: Texto da mensagem citada (para mensagens de texto)
        imageMessage: Dados da imagem citada (opcional)
        documentMessage: Dados do documento citado (opcional)
        audioMessage: Dados do áudio citado (opcional)
        videoMessage: Dados do vídeo citado (opcional)
    """

    conversation: str | None = Field(default=None)
    imageMessage: ImageMessage | None = Field(default=None)
    documentMessage: DocumentMessage | None = Field(default=None)
    audioMessage: AudioMessage | None = Field(default=None)
    videoMessage: VideoMessage | None = Field(default=None)
