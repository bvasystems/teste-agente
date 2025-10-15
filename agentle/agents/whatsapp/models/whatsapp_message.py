from datetime import datetime
from typing import Any, override

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.whatsapp_message_status import WhatsAppMessageStatus
from agentle.agents.whatsapp.models.whatsapp_message_type import WhatsAppMessageType


class WhatsAppMessage(BaseModel):
    """Base WhatsApp message model."""

    id: str
    type: WhatsAppMessageType
    from_number: str
    to_number: str
    push_name: str | None = None
    timestamp: datetime
    status: WhatsAppMessageStatus = WhatsAppMessageStatus.PENDING
    is_group: bool = False
    group_id: str | None = None
    quoted_message_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @override
    def model_post_init(self, context: Any, /) -> None:
        """Post-initialize the model."""
        if "@" in self.from_number:
            self.from_number = self.from_number.split("@")[0]

        if "@" in self.to_number:
            self.to_number = self.to_number.split("@")[0]
