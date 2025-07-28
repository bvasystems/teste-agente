from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact


class WhatsAppSession(BaseModel):
    """WhatsApp conversation session with message batching and spam protection."""

    session_id: str
    phone_number: str
    contact: WhatsAppContact
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    is_active: bool = True
    context_data: MutableMapping[str, Any] = Field(default_factory=dict)
    agent_context_id: str | None = None

    # Message batching and spam protection fields
    is_processing: bool = Field(
        default=False, description="Whether the bot is currently processing messages"
    )
    pending_messages: MutableSequence[dict[str, Any]] = Field(
        default_factory=list, description="Queue of messages waiting to be batched"
    )
    last_batch_started_at: datetime | None = Field(
        default=None, description="When the current message batch processing started"
    )
    batch_processing_timeout_at: datetime | None = Field(
        default=None, description="When to force process the current batch"
    )

    # Spam protection tracking
    last_message_at: datetime | None = Field(
        default=None, description="Timestamp of last message received"
    )
    messages_in_current_minute: int = Field(
        default=0, description="Count of messages in current minute window"
    )
    current_minute_start: datetime | None = Field(
        default=None, description="Start of current minute window for rate limiting"
    )
    is_rate_limited: bool = Field(
        default=False, description="Whether user is currently rate limited"
    )
    rate_limit_until: datetime | None = Field(
        default=None, description="When rate limiting expires"
    )

    def add_pending_message(self, message_data: dict[str, Any]) -> None:
        """Add a message to the pending queue."""
        self.pending_messages.append(message_data)
        self.last_activity = datetime.now()

    def clear_pending_messages(self) -> MutableSequence[dict[str, Any]]:
        """Clear and return all pending messages."""
        messages = list(self.pending_messages)
        self.pending_messages.clear()
        return messages

    def update_rate_limiting(
        self, max_messages_per_minute: int, cooldown_seconds: int
    ) -> bool:
        """
        Update rate limiting state and return True if message should be processed.

        Args:
            max_messages_per_minute: Maximum allowed messages per minute
            cooldown_seconds: Cooldown period in seconds

        Returns:
            True if message can be processed, False if rate limited
        """
        now = datetime.now()

        # Check if rate limit has expired
        if (
            self.is_rate_limited
            and self.rate_limit_until
            and now >= self.rate_limit_until
        ):
            self.is_rate_limited = False
            self.rate_limit_until = None
            self.messages_in_current_minute = 0
            self.current_minute_start = None

        # If currently rate limited, deny
        if self.is_rate_limited:
            return False

        # Reset minute counter if needed
        if (
            self.current_minute_start is None
            or (now - self.current_minute_start).total_seconds() >= 60
        ):
            self.current_minute_start = now
            self.messages_in_current_minute = 0

        # Increment message count
        self.messages_in_current_minute += 1

        # Check if rate limit should be triggered
        if self.messages_in_current_minute > max_messages_per_minute:
            self.is_rate_limited = True
            self.rate_limit_until = now.replace(
                second=now.second + cooldown_seconds, microsecond=0
            )
            return False

        self.last_message_at = now
        return True

    def should_process_batch(
        self, batch_delay_seconds: float, max_wait_seconds: float
    ) -> bool:
        """
        Determine if the current message batch should be processed.

        Args:
            batch_delay_seconds: Normal delay before processing batch
            max_wait_seconds: Maximum time to wait before forcing processing

        Returns:
            True if batch should be processed now
        """
        if not self.pending_messages:
            return False

        now = datetime.now()

        # Force processing if max wait time exceeded
        if self.batch_processing_timeout_at and now >= self.batch_processing_timeout_at:
            return True

        # Process if delay period has passed since last message
        if self.last_activity:
            time_since_last = (now - self.last_activity).total_seconds()
            return time_since_last >= batch_delay_seconds

        return False

    def start_batch_processing(self, max_wait_seconds: float) -> None:
        """Start batch processing with timeout."""
        now = datetime.now()
        self.is_processing = True
        self.last_batch_started_at = now
        self.batch_processing_timeout_at = now.replace(
            second=now.second + int(max_wait_seconds),
            microsecond=int((max_wait_seconds % 1) * 1_000_000),
        )

    def finish_batch_processing(self) -> None:
        """Finish batch processing and reset state."""
        self.is_processing = False
        self.last_batch_started_at = None
        self.batch_processing_timeout_at = None
