from typing import Any, TypedDict


class DeviceListMetadata(TypedDict):
    """Metadados da lista de dispositivos WhatsApp.

    Attributes:
        senderKeyHash: Hash da chave do remetente (pode ser dict ou str)
        senderTimestamp: Timestamp do remetente (pode ser dict ou str)
        recipientKeyHash: Hash da chave do destinatário (pode ser dict ou str)
        recipientTimestamp: Timestamp do destinatário (pode ser dict ou str)
    """

    senderKeyHash: dict[str, Any] | str
    senderTimestamp: dict[str, Any] | str
    recipientKeyHash: dict[str, Any] | str
    recipientTimestamp: dict[str, Any] | str
