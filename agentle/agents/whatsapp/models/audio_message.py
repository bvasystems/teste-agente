from typing import Any

from pydantic import field_validator
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class AudioMessage(BaseModel):
    """Dados de uma mensagem de áudio do WhatsApp.

    Attributes:
        url: URL do áudio no servidor WhatsApp
        mimetype: Tipo MIME do áudio (ex: "audio/ogg; codecs=opus")
        fileSha256: Hash SHA256 do arquivo para verificação de integridade
        fileLength: Tamanho do arquivo em bytes (como string)
        seconds: Duração do áudio em segundos
        ptt: Se é um áudio push-to-talk (nota de voz)
        mediaKey: Chave de criptografia para decodificar a mídia
        fileEncSha256: Hash SHA256 do arquivo criptografado
        directPath: Caminho direto para download da mídia
        mediaKeyTimestamp: Timestamp da chave de mídia
        streamingSidecar: Dados para streaming do áudio (opcional)
        waveform: Forma de onda do áudio em base64 (opcional)
    """

    url: str
    mimetype: str | None = Field(default=None)
    fileSha256: str | dict[str, Any] | None = Field(default=None)
    fileLength: str | dict[str, Any] | None = Field(default=None)
    seconds: int | None = Field(default=None)
    ptt: bool | None = Field(default=None)
    mediaKey: str | dict[str, Any] | None = Field(default=None)
    fileEncSha256: str | dict[str, Any] | None = Field(default=None)
    directPath: str | None = Field(default=None)
    mediaKeyTimestamp: str | dict[str, Any] | None = Field(default=None)
    streamingSidecar: str | dict[str, Any] | None = Field(default=None)
    waveform: str | dict[str, Any] | None = Field(default=None)

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
        "streamingSidecar",
        "waveform",
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
