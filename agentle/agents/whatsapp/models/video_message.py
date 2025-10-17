from typing import Any

from pydantic import field_validator
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class VideoMessage(BaseModel):
    """Dados de uma mensagem com vídeo do WhatsApp.

    Attributes:
        url: URL do vídeo no servidor WhatsApp
        mimetype: Tipo MIME do vídeo (ex: "video/mp4")
        fileSha256: Hash SHA256 do arquivo para verificação de integridade
        fileLength: Tamanho do arquivo em bytes (como string)
        height: Altura do vídeo em pixels
        width: Largura do vídeo em pixels
        mediaKey: Chave de criptografia para decodificar a mídia
        fileEncSha256: Hash SHA256 do arquivo criptografado
        directPath: Caminho direto para download da mídia
        mediaKeyTimestamp: Timestamp da chave de mídia
        jpegThumbnail: Thumbnail do vídeo em base64 (opcional)
        contextInfo: Informações de contexto da mensagem (opcional)
        firstScanSidecar: Dados do primeiro scan de segurança (opcional)
        firstScanLength: Tamanho do primeiro scan (opcional)
        scansSidecar: Dados dos scans de segurança subsequentes (opcional)
        scanLengths: Lista com tamanhos dos scans realizados (opcional)
        midQualityFileSha256: Hash SHA256 da versão de qualidade média (opcional)
    """

    url: str
    mimetype: str | None = Field(default=None)
    caption: str | None = Field(default=None)
    fileSha256: str | dict[str, Any] | None = Field(default=None)
    fileLength: str | dict[str, Any] | None = Field(default=None)
    height: int | None = Field(default=None)
    width: int | None = Field(default=None)
    mediaKey: str | dict[str, Any] | None = Field(default=None)
    fileEncSha256: str | dict[str, Any] | None = Field(default=None)
    directPath: str | None = Field(default=None)
    mediaKeyTimestamp: str | dict[str, Any] | None = Field(default=None)
    jpegThumbnail: str | dict[str, Any] | None = Field(default=None)
    contextInfo: dict[str, Any] | None = Field(default=None)
    firstScanSidecar: str | None = Field(default=None)
    firstScanLength: int | None = Field(default=None)
    scansSidecar: str | dict[str, Any] | None = Field(default=None)
    scanLengths: list[int] | None = Field(default=None)
    midQualityFileSha256: str | dict[str, Any] | None = Field(default=None)

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
        "jpegThumbnail",
        "scansSidecar",
        "midQualityFileSha256",
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
