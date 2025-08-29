from typing import NotRequired, TypedDict


class AudioMessage(TypedDict):
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
    mimetype: NotRequired[str]
    fileSha256: NotRequired[str]
    fileLength: NotRequired[str]
    seconds: NotRequired[int]
    ptt: NotRequired[bool]
    mediaKey: NotRequired[str]
    fileEncSha256: NotRequired[str]
    directPath: NotRequired[str]
    mediaKeyTimestamp: NotRequired[str]
    streamingSidecar: NotRequired[str]
    waveform: NotRequired[str]
