from rsb.models import BaseModel, Field


class Key(BaseModel):
    """Chave identificadora da mensagem WhatsApp.

    Attributes:
        remoteJid: ID do chat/contato remoto (ex: "553497722562@s.whatsapp.net")
        fromMe: Se a mensagem foi enviada por mim (True) ou recebida (False)
        id: ID único da mensagem no WhatsApp
    """

    remoteJid: str
    remoteJidAlt: str = Field(default="")
    fromMe: bool
    id: str
