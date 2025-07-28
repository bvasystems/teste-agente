from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class WhatsAppBotConfig(BaseModel):
    """Configuration for WhatsApp bot behavior."""

    typing_indicator: bool = Field(
        default=True, description="Show typing indicator while processing"
    )
    typing_duration: int = Field(
        default=3, description="Duration to show typing indicator in seconds"
    )
    auto_read_messages: bool = Field(
        default=True, description="Automatically mark messages as read"
    )
    session_timeout_minutes: int = Field(
        default=30, description="Minutes of inactivity before session reset"
    )
    max_message_length: int = Field(
        default=4096, description="Maximum message length (WhatsApp limit)"
    )
    error_message: str = Field(
        default="Sorry, I encountered an error processing your message. Please try again.",
        description="Default error message",
    )
    welcome_message: str | None = Field(
        default=None, description="Message to send on first interaction"
    )
    
    # Spam protection and message batching settings
    enable_message_batching: bool = Field(
        default=True, description="Enable message batching to prevent spam"
    )
    message_batch_delay_seconds: float = Field(
        default=2.0, description="Delay to wait for additional messages before processing batch"
    )
    max_batch_size: int = Field(
        default=10, description="Maximum number of messages to batch together"
    )
    max_batch_wait_seconds: float = Field(
        default=40.0, description="Maximum time to wait for batching before forcing processing"
    )
    spam_protection_enabled: bool = Field(
        default=True, description="Enable spam protection mechanisms"
    )
    min_message_interval_seconds: float = Field(
        default=0.5, description="Minimum interval between processing messages from same user"
    )
    max_messages_per_minute: int = Field(
        default=20, description="Maximum messages per minute per user before rate limiting"
    )
    rate_limit_cooldown_seconds: int = Field(
        default=60, description="Cooldown period after rate limit is triggered"
    )