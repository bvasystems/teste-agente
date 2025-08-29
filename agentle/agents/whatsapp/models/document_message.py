from typing import NotRequired, TypedDict


class DocumentMessage(TypedDict):
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
    mimetype: NotRequired[str]
    title: NotRequired[str]
    fileSha256: NotRequired[str]
    fileLength: NotRequired[str]
    mediaKey: NotRequired[str]
    fileName: NotRequired[str]
    fileEncSha256: NotRequired[str]
    directPath: NotRequired[str]
    mediaKeyTimestamp: NotRequired[str]
    contactVcard: NotRequired[bool]
