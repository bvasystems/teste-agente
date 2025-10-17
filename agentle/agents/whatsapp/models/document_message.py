from typing import Any

from pydantic import field_validator
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class DocumentMessage(BaseModel):
    """Dados de uma mensagem com documento do WhatsApp.

    Attributes:
        url: URL do documento no servidor WhatsApp
        mimetype: Tipo MIME do documento (ex: "application/pdf")
        title: Título/nome exibido do documento
        fileSha256: Hash SHA256 do arquivo para verificação de integridade
        fileLength: Tamanho do arquivo em bytes (como string)
        mediaKey: Chave de criptografia para decodificar a mídia
        fileName: Nome original do arquivo
        fileEncSha256: Hash SHA256 do arquivo criptografado
        directPath: Caminho direto para download da mídia
        mediaKeyTimestamp: Timestamp da chave de mídia
        contactVcard: Se o documento é um cartão de contato vCard
    """

    url: str
    mimetype: str | None = Field(default=None)
    title: str | None = Field(default=None)
    fileSha256: str | dict[str, Any] | None = Field(default=None)
    fileLength: str | dict[str, Any] | None = Field(default=None)
    mediaKey: str | dict[str, Any] | None = Field(default=None)
    fileName: str | None = Field(default=None)
    fileEncSha256: str | dict[str, Any] | None = Field(default=None)
    directPath: str | None = Field(default=None)
    mediaKeyTimestamp: str | dict[str, Any] | None = Field(default=None)
    contactVcard: bool | None = Field(default=None)
    caption: str | None = Field(default=None)

    @field_validator(
        "fileLength",
        "mediaKeyTimestamp",
        mode="before",
    )
    @classmethod
    def convert_long_to_str(cls, v: Any) -> str | None:
        """Converte objetos Long do protobuf para string."""
        if v is None:
            return None
        if isinstance(v, dict) and "low" in v:
            low = v.get("low", 0)
            high = v.get("high", 0)
            value = (high << 32) | low
            return str(value)
        return str(v)

    @field_validator(
        "fileSha256",
        "mediaKey",
        "fileEncSha256",
        mode="before",
    )
    @classmethod
    def convert_buffer_to_str(cls, v: Any) -> str | None:
        """Converte objetos Buffer/Bytes do protobuf para string."""
        if v is None:
            return None
        if isinstance(v, dict) and all(k.isdigit() for k in v.keys()):
            return str(v)
        return str(v) if not isinstance(v, str) else v
