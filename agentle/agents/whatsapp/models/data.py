from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.context_info import ContextInfo
from agentle.agents.whatsapp.models.key import Key
from agentle.agents.whatsapp.models.message import Message


class Data(BaseModel):
    """Dados principais do webhook WhatsApp.

    Attributes:
        key: Chave identificadora da mensagem
        pushName: Nome de exibição do remetente
        status: Status da mensagem (ex: "SERVER_ACK")
        message: Conteúdo da mensagem (opcional)
        messageType: Tipo da mensagem (ex: "conversation", "imageMessage")
        messageTimestamp: Timestamp Unix da mensagem (opcional)
        instanceId: ID da instância WhatsApp (opcional)
        source: Plataforma de origem (ex: "ios", "android") (opcional)
        contextInfo: Informações de contexto ou resposta (opcional)
    """

    key: Key
    pushName: str
    status: str | None = Field(default=None)
    message: Message | None = Field(default=None)
    messageType: str | None = Field(default=None)
    messageTimestamp: int | None = Field(default=None)
    instanceId: str | None = Field(default=None)
    source: str | None = Field(default=None)
    contextInfo: ContextInfo | None = Field(default=None)
