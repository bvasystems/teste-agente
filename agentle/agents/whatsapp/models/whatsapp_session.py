from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from typing import Any
import logging

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact

logger = logging.getLogger(__name__)


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
        logger.debug(
            f"[SESSION] Adding pending message for {self.phone_number}. "
            + f"Queue size before: {len(self.pending_messages)}"
        )

        self.pending_messages.append(message_data)
        self.last_activity = datetime.now()

        logger.info(
            f"[SESSION] Added pending message to {self.phone_number}. "
            + f"Queue size now: {len(self.pending_messages)}, "
            + f"Message ID: {message_data.get('id', 'unknown')}"
        )

    def clear_pending_messages(self) -> MutableSequence[dict[str, Any]]:
        """Clear and return all pending messages."""
        messages = list(self.pending_messages)
        logger.info(
            f"[SESSION] Clearing {len(messages)} pending messages for {self.phone_number}"
        )

        self.pending_messages.clear()

        logger.debug(
            f"[SESSION] Cleared pending messages for {self.phone_number}. "
            + f"Remaining in queue: {len(self.pending_messages)}"
        )

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

        logger.debug(
            f"[RATE_LIMIT] Checking rate limits for {self.phone_number}: "
            + f"is_rate_limited={self.is_rate_limited}, "
            + f"messages_in_current_minute={self.messages_in_current_minute}"
        )

        # Check if rate limit has expired
        if (
            self.is_rate_limited
            and self.rate_limit_until
            and now >= self.rate_limit_until
        ):
            logger.info(
                f"[RATE_LIMIT] Rate limit expired for {self.phone_number}, resetting"
            )
            self.is_rate_limited = False
            self.rate_limit_until = None
            self.messages_in_current_minute = 0
            self.current_minute_start = None

        # If currently rate limited, deny
        if self.is_rate_limited:
            logger.warning(
                f"[RATE_LIMIT] User {self.phone_number} is rate limited until {self.rate_limit_until}"
            )
            return False

        # Reset minute counter if needed
        if (
            self.current_minute_start is None
            or (now - self.current_minute_start).total_seconds() >= 60
        ):
            logger.debug(
                f"[RATE_LIMIT] Resetting minute counter for {self.phone_number}"
            )
            self.current_minute_start = now
            self.messages_in_current_minute = 0

        # Increment message count
        self.messages_in_current_minute += 1
        logger.debug(
            f"[RATE_LIMIT] Message count for {self.phone_number}: "
            + f"{self.messages_in_current_minute}/{max_messages_per_minute}"
        )

        # Check if rate limit should be triggered
        if self.messages_in_current_minute > max_messages_per_minute:
            logger.warning(
                f"[RATE_LIMIT] Rate limit triggered for {self.phone_number}. "
                + f"Messages: {self.messages_in_current_minute}/{max_messages_per_minute}"
            )
            self.is_rate_limited = True
            self.rate_limit_until = now.replace(
                second=now.second + cooldown_seconds, microsecond=0
            )
            return False

        self.last_message_at = now
        logger.debug(f"[RATE_LIMIT] Message allowed for {self.phone_number}")
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
            logger.debug(
                f"[BATCH_DECISION] No pending messages for {self.phone_number}"
            )
            return False

        now = datetime.now()

        logger.debug(
            f"[BATCH_DECISION] Checking batch processing conditions for {self.phone_number}: "
            + f"pending_messages={len(self.pending_messages)}, "
            + f"batch_timeout_at={self.batch_processing_timeout_at}, "
            + f"last_activity={self.last_activity}, "
            + f"batch_delay_seconds={batch_delay_seconds}, "
            + f"max_wait_seconds={max_wait_seconds}"
        )

        # Force processing if max wait time exceeded
        if self.batch_processing_timeout_at and now >= self.batch_processing_timeout_at:
            logger.info(
                f"[BATCH_DECISION] Max wait time exceeded for {self.phone_number}, forcing processing"
            )
            return True

        # Process if delay period has passed since last message
        if self.last_activity:
            time_since_last = (now - self.last_activity).total_seconds()
            should_process = time_since_last >= batch_delay_seconds

            logger.debug(
                f"[BATCH_DECISION] Time since last activity for {self.phone_number}: "
                + f"{time_since_last:.2f}s (threshold: {batch_delay_seconds}s) -> {should_process}"
            )

            return should_process

        logger.debug(f"[BATCH_DECISION] No last_activity time for {self.phone_number}")
        return False

    def start_batch_processing(self, max_wait_seconds: float) -> None:
        """Start batch processing with timeout."""
        from datetime import timedelta

        now = datetime.now()

        logger.info(
            f"[BATCH_START] Starting batch processing for {self.phone_number} "
            + f"with max_wait_seconds={max_wait_seconds}"
        )

        # CRITICAL FIX: Set processing state first
        was_processing = self.is_processing
        self.is_processing = True
        self.last_batch_started_at = now

        # CRITICAL FIX: Use timedelta for proper datetime arithmetic
        # The previous code had a bug where it could create invalid datetime objects
        self.batch_processing_timeout_at = now + timedelta(seconds=max_wait_seconds)

        logger.info(
            f"[BATCH_START] CRITICAL STATE CHANGE for {self.phone_number}: "
            + f"was_processing={was_processing} -> is_processing={self.is_processing}, "
            + f"started_at={self.last_batch_started_at}, "
            + f"timeout_at={self.batch_processing_timeout_at}, "
            + f"pending_messages={len(self.pending_messages)}"
        )

    def finish_batch_processing(self) -> None:
        """Finish batch processing and reset state."""
        logger.info(
            f"[BATCH_FINISH] Finishing batch processing for {self.phone_number}. "
            + f"Was processing: {self.is_processing}"
        )

        self.is_processing = False
        self.last_batch_started_at = None
        self.batch_processing_timeout_at = None

        logger.debug(
            f"[BATCH_FINISH] Batch processing finished for {self.phone_number}: "
            + f"is_processing={self.is_processing}, "
            + f"pending_messages={len(self.pending_messages)}"
        )
