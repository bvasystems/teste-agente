from collections.abc import MutableMapping, MutableSequence
from datetime import datetime, timedelta
from typing import Any, override
import logging
import uuid

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.models.whatsapp_contact import WhatsAppContact

logger = logging.getLogger(__name__)


# In whatsapp_session.py - Remove agent_context_id field
class WhatsAppSession(BaseModel):
    """WhatsApp conversation session with improved message batching and spam protection."""

    session_id: str
    phone_number: str
    contact: WhatsAppContact
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    is_active: bool = True
    context_data: MutableMapping[str, Any] = Field(default_factory=dict)

    # Message batching and spam protection fields
    is_processing: bool = Field(
        default=False, description="Whether the bot is currently processing messages"
    )
    pending_messages: MutableSequence[dict[str, Any]] = Field(
        default_factory=list, description="Queue of messages waiting to be batched"
    )
    batch_started_at: datetime | None = Field(
        default=None, description="When the current message batch processing started"
    )
    batch_timeout_at: datetime | None = Field(
        default=None, description="When to force process the current batch"
    )
    last_message_added_at: datetime | None = Field(
        default=None, description="When the last message was added to the current batch"
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

    # Enhanced state management
    processing_token: str | None = Field(
        default=None, description="Unique token for the current processing session"
    )
    last_state_change: datetime = Field(
        default_factory=datetime.now, description="Last time session state changed"
    )

    @override
    def model_post_init(self, context: Any, /) -> None:
        """Post-initialize the model."""
        if "@" in self.phone_number:
            self.phone_number = self.phone_number.split("@")[0]

    def add_pending_message(self, message_data: dict[str, Any]) -> None:
        """Add a message to the pending queue with improved logging and timer reset."""
        logger.debug(
            f"[SESSION] Adding pending message for {self.phone_number}. "
            + f"Queue size before: {len(self.pending_messages)}"
        )

        current_time = datetime.now()
        self.pending_messages.append(message_data)
        
        # CRITICAL FIX: Always update when last message was added
        self.last_message_added_at = current_time
        
        # CRITICAL FIX: Reset batch timer when new messages arrive to existing batch
        # This ensures messages are properly accumulated instead of processed separately
        if self.is_processing and self.batch_started_at:
            old_batch_started = self.batch_started_at
            self.batch_started_at = current_time
            logger.info(
                f"[SESSION] ⏰ TIMER RESET for {self.phone_number}: "
                + f"batch_started_at updated from {old_batch_started} to {self.batch_started_at} "
                + f"due to new message arrival (queue size: {len(self.pending_messages)})"
            )
        
        # Update last activity only if we're not in the middle of batch processing
        # This prevents race conditions with the batch timer
        if not self.is_processing:
            self.last_activity = current_time
            self.last_state_change = current_time

        logger.info(
            f"[SESSION] Added pending message to {self.phone_number}. "
            + f"Queue size now: {len(self.pending_messages)}, "
            + f"Message ID: {message_data.get('id', 'unknown')}"
        )

    def clear_pending_messages(self) -> MutableSequence[dict[str, Any]]:
        """Clear and return all pending messages with improved state management."""
        messages = list(self.pending_messages)
        logger.info(
            f"[SESSION] Clearing {len(messages)} pending messages for {self.phone_number}"
        )

        self.pending_messages.clear()
        self.last_state_change = datetime.now()

        logger.debug(
            f"[SESSION] Cleared pending messages for {self.phone_number}. "
            + f"Remaining in queue: {len(self.pending_messages)}"
        )

        return messages

    def update_rate_limiting(
        self, max_messages_per_minute: int, cooldown_seconds: int
    ) -> bool:
        """
        Update rate limiting state with improved reliability.

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
            self.last_state_change = now

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
        self.last_message_at = now
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
            self.rate_limit_until = now + timedelta(seconds=cooldown_seconds)
            self.last_state_change = now
            return False

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
            + f"batch_timeout_at={self.batch_timeout_at}, "
            + f"batch_started_at={self.batch_started_at}, "
            + f"batch_delay_seconds={batch_delay_seconds}, "
            + f"max_wait_seconds={max_wait_seconds}"
        )

        # Force processing if max wait time exceeded
        if self.batch_timeout_at and now >= self.batch_timeout_at:
            logger.info(
                f"[BATCH_DECISION] Max wait time exceeded for {self.phone_number}, forcing processing"
            )
            return True

        # IMPROVED LOGIC: Consider time since last message was added
        # This ensures we wait for the full delay after the last message
        if self.batch_started_at:
            reference_time = self.batch_started_at
            if self.last_message_added_at and self.last_message_added_at > self.batch_started_at:
                reference_time = self.last_message_added_at

            time_since_reference = (now - reference_time).total_seconds()
            should_process = time_since_reference >= batch_delay_seconds

            logger.debug(
                f"[BATCH_DECISION] Time since reference time for {self.phone_number}: "
                + f"{time_since_reference:.2f}s (threshold: {batch_delay_seconds}s) -> {should_process}"
            )

            return should_process

        # This should not happen if start_batch_processing was called correctly
        logger.warning(
            f"[BATCH_DECISION] No batch start time available for {self.phone_number}"
        )
        return False

    def start_batch_processing(self, max_wait_seconds: float) -> str:
        """
        Start batch processing with improved state management.

        Args:
            max_wait_seconds: Maximum seconds to wait before forcing processing

        Returns:
            Processing token for this batch
        """
        now = datetime.now()
        processing_token = str(uuid.uuid4())

        logger.info(
            f"[BATCH_START] Starting batch processing for {self.phone_number} "
            + f"with max_wait_seconds={max_wait_seconds}, token={processing_token}"
        )

        # Set processing state
        was_processing = self.is_processing
        self.is_processing = True
        self.batch_started_at = now
        self.processing_token = processing_token
        self.last_state_change = now

        # Set timeout
        self.batch_timeout_at = now + timedelta(seconds=max_wait_seconds)

        logger.info(
            f"[BATCH_START] CRITICAL STATE CHANGE for {self.phone_number}: "
            + f"was_processing={was_processing} -> is_processing={self.is_processing}, "
            + f"started_at={self.batch_started_at}, "
            + f"timeout_at={self.batch_timeout_at}, "
            + f"pending_messages={len(self.pending_messages)}, "
            + f"token={processing_token}"
        )

        return processing_token

    def finish_batch_processing(self, processing_token: str | None = None) -> bool:
        """
        Finish batch processing with token validation.

        Args:
            processing_token: Token from start_batch_processing

        Returns:
            True if processing was finished, False if token mismatch
        """
        logger.info(
            f"[BATCH_FINISH] Finishing batch processing for {self.phone_number}. "
            + f"Was processing: {self.is_processing}, token: {processing_token}"
        )

        # Validate token if provided
        if processing_token is not None and self.processing_token != processing_token:
            logger.warning(
                f"[BATCH_FINISH] Token mismatch for {self.phone_number}: "
                + f"expected={self.processing_token}, got={processing_token}"
            )
            return False

        current_time = datetime.now()
        self.is_processing = False
        self.batch_started_at = None
        self.batch_timeout_at = None
        self.last_message_added_at = None  # Reset last message time
        self.processing_token = None
        self.last_activity = current_time
        self.last_state_change = current_time

        logger.debug(
            f"[BATCH_FINISH] Batch processing finished for {self.phone_number}: "
            + f"is_processing={self.is_processing}, "
            + f"pending_messages={len(self.pending_messages)}"
        )

        return True

    def is_batch_expired(self, max_wait_seconds: float) -> bool:
        """
        Check if the current batch has expired and should be force-processed.

        Args:
            max_wait_seconds: Maximum seconds to wait

        Returns:
            True if batch has expired
        """
        if not self.is_processing or not self.batch_started_at:
            return False

        now = datetime.now()
        time_since_start = (now - self.batch_started_at).total_seconds()

        return time_since_start >= max_wait_seconds

    def get_batch_age_seconds(self) -> float:
        """Get the age of the current batch in seconds."""
        if not self.batch_started_at:
            return 0.0

        return (datetime.now() - self.batch_started_at).total_seconds()

    def reset_session(self) -> None:
        """Reset session to initial state."""
        logger.info(f"[SESSION_RESET] Resetting session for {self.phone_number}")

        self.is_processing = False
        self.pending_messages.clear()
        self.batch_started_at = None
        self.batch_timeout_at = None
        self.processing_token = None

        # Reset rate limiting
        self.is_rate_limited = False
        self.rate_limit_until = None
        self.messages_in_current_minute = 0
        self.current_minute_start = None

        # Update timestamps
        self.last_activity = datetime.now()
        self.last_state_change = datetime.now()

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of the session state for debugging."""
        return {
            "session_id": self.session_id,
            "phone_number": self.phone_number,
            "is_processing": self.is_processing,
            "pending_messages_count": len(self.pending_messages),
            "batch_started_at": self.batch_started_at.isoformat()
            if self.batch_started_at
            else None,
            "batch_timeout_at": self.batch_timeout_at.isoformat()
            if self.batch_timeout_at
            else None,
            "processing_token": self.processing_token,
            "is_rate_limited": self.is_rate_limited,
            "rate_limit_until": self.rate_limit_until.isoformat()
            if self.rate_limit_until
            else None,
            "messages_in_current_minute": self.messages_in_current_minute,
            "last_activity": self.last_activity.isoformat(),
            "last_state_change": self.last_state_change.isoformat(),
            "message_count": self.message_count,
        }
