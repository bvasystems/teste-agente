from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.data import Data


class WhatsAppWebhookPayload(BaseModel):
    """Webhook payload from WhatsApp."""

    # Evolution API
    event: str | None = Field(default=None)
    instance: str | None = Field(default=None)
    data: Data | None = Field(default=None)
    destination: str | None = Field(default=None)
    date_time: datetime | None = Field(default=None)
    sender: str | None = Field(default=None)
    server_url: str | None = Field(default=None)
    apikey: str | None = Field(default=None)

    # Meta WhatsApp Business API
    entry: list[dict[str, Any]] | None = Field(default=None)
    changes: list[dict[str, Any]] | None = Field(default=None)
    field: str | None = Field(default=None)
    value: Mapping[str, Any] | None = Field(default=None)
    phone_number_id: str | None = Field(default=None)
    metadata: Mapping[str, Any] | None = Field(default=None)
    status: str | None = Field(default=None)
    status_code: int | None = Field(default=None)

    # Twilio WhatsApp API
    MessageSid: str | None = Field(default=None)
    AccountSid: str | None = Field(default=None)
    MessagingServiceSid: str | None = Field(default=None)
    From: str | None = Field(default=None)  # Format: whatsapp:+15551234567
    To: str | None = Field(default=None)  # Format: whatsapp:+15559876543
    Body: str | None = Field(default=None)
    NumMedia: str | None = Field(default=None)  # Number of media items (as string)
    MediaUrl0: str | None = Field(default=None)
    MediaContentType0: str | None = Field(default=None)
    MediaUrl1: str | None = Field(default=None)
    MediaContentType1: str | None = Field(default=None)
    MediaUrl2: str | None = Field(default=None)
    MediaContentType2: str | None = Field(default=None)
    MediaUrl3: str | None = Field(default=None)
    MediaContentType3: str | None = Field(default=None)
    MediaUrl4: str | None = Field(default=None)
    MediaContentType4: str | None = Field(default=None)
    MediaUrl5: str | None = Field(default=None)
    MediaContentType5: str | None = Field(default=None)
    MediaUrl6: str | None = Field(default=None)
    MediaContentType6: str | None = Field(default=None)
    MediaUrl7: str | None = Field(default=None)
    MediaContentType7: str | None = Field(default=None)
    MediaUrl8: str | None = Field(default=None)
    MediaContentType8: str | None = Field(default=None)
    MediaUrl9: str | None = Field(default=None)
    MediaContentType9: str | None = Field(default=None)
    SmsStatus: str | None = Field(
        default=None
    )  # sent, delivered, read, failed, undelivered
    SmsSid: str | None = Field(default=None)
    SmsMessageSid: str | None = Field(default=None)
    NumSegments: str | None = Field(default=None)
    ReferralNumMedia: str | None = Field(default=None)
    ProfileName: str | None = Field(default=None)
    WaId: str | None = Field(default=None)  # WhatsApp ID without prefix
    ButtonText: str | None = Field(default=None)  # For interactive button messages
    ButtonPayload: str | None = Field(default=None)  # For interactive button messages
    Latitude: str | None = Field(default=None)  # For location messages
    Longitude: str | None = Field(default=None)  # For location messages
    Address: str | None = Field(default=None)  # For location messages
    Label: str | None = Field(default=None)  # For location messages
    ForwardedFrom: str | None = Field(default=None)  # For forwarded messages
    FrequentlyForwarded: str | None = Field(default=None)  # "true" or "false" as string

    def model_post_init(self, context: Any, /) -> None:
        if self.phone_number_id or not self.data:
            return

        key = self.data.key
        if "@lid" in key.remoteJid:
            self.phone_number_id = key.remoteJidAlt.split("@")[0]

        self.phone_number_id = key.remoteJid.split("@")[0]
