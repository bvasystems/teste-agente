from typing import NotRequired, TypedDict

from agentle.agents.whatsapp.models.audio_message import AudioMessage
from agentle.agents.whatsapp.models.document_message import DocumentMessage
from agentle.agents.whatsapp.models.image_message import ImageMessage
from agentle.agents.whatsapp.models.video_message import VideoMessage


class QuotedMessage(TypedDict):
    """Mensagem citada/respondida no WhatsApp.

    Attributes:
        conversation: Texto da mensagem citada (para mensagens de texto)
        imageMessage: Dados da imagem citada (opcional)
        documentMessage: Dados do documento citado (opcional)
        audioMessage: Dados do áudio citado (opcional)
        videoMessage: Dados do vídeo citado (opcional)
    """

    conversation: NotRequired[str]
    imageMessage: NotRequired[ImageMessage]
    documentMessage: NotRequired[DocumentMessage]
    audioMessage: NotRequired[AudioMessage]
    videoMessage: NotRequired[VideoMessage]