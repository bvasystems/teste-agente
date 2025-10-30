from __future__ import annotations

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.whatsapp.v2.message_limit import MessageLimit
from agentle.tts.speech_config import SpeechConfig


class BotConfig(BaseModel):
    """Configuration for WhatsApp bot behavior with simplified constructors and better organization.

    This configuration class provides comprehensive control over WhatsApp bot behavior including:
    - Core bot behavior (typing indicators, message reading, quoting)
    - Message batching for handling rapid message sequences
    - Spam protection and rate limiting
    - Human-like delays to simulate realistic human behavior patterns
    - Text-to-speech integration
    - Error handling and retry logic
    - Debug and monitoring settings

    Human-Like Delays Feature:
        The human-like delays feature simulates realistic human behavior patterns by introducing
        configurable delays at three critical points in message processing:

        1. Read Delay: Time between receiving a message and marking it as read
           - Simulates the time a human takes to read and comprehend a message
           - Calculated based on message length using realistic reading speeds

        2. Typing Delay: Time between generating a response and sending it
           - Simulates the time a human takes to compose and type a response
           - Calculated based on response length using realistic typing speeds

        3. Send Delay: Brief final delay before message transmission
           - Simulates the final review time before a human sends a message
           - Random delay within configured bounds

        These delays help prevent platform detection and account restrictions while
        maintaining natural interaction timing. All delays support jitter (random variation)
        to prevent detectable patterns.

    Configuration Presets:
        Use the class methods to create pre-configured instances optimized for specific use cases:
        - development(): Fast iteration with delays disabled
        - production(): Balanced configuration with delays enabled
        - high_volume(): Optimized for throughput with balanced delays
        - customer_service(): Professional timing with thoughtful delays
        - minimal(): Bare minimum configuration with delays disabled

    Examples:
        >>> # Create a production configuration with default delay settings
        >>> config = BotConfig.production()

        >>> # Create a custom configuration with specific delay bounds
        >>> config = BotConfig(
        ...     enable_human_delays=True,
        ...     min_read_delay_seconds=3.0,
        ...     max_read_delay_seconds=20.0,
        ...     min_typing_delay_seconds=5.0,
        ...     max_typing_delay_seconds=60.0
        ... )

        >>> # Override delay settings on an existing configuration
        >>> prod_config = BotConfig.production()
        >>> custom_config = prod_config.with_overrides(
        ...     min_read_delay_seconds=5.0,
        ...     max_typing_delay_seconds=90.0
        ... )
    """

    quote_messages: bool = Field(
        default=False, description="Whether to quote user messages in replies"
    )
    session_timeout_minutes: int = Field(
        default=30, description="Minutes of inactivity before session reset"
    )
    max_message_length: MessageLimit = Field(
        default=MessageLimit.NEWLY_CREATED,
        description="Maximum message length (WhatsApp limit)",
    )
    max_split_messages: int = Field(
        default=5,
        description="Maximum number of split messages to send (remaining will be grouped)",
    )
    error_message: str = Field(
        default="Sorry, I encountered an error processing your message. Please try again.",
        description="Default error message",
    )
    welcome_message: str | None = Field(
        default=None, description="Message to send on first interaction"
    )

    # === Message Batching (Simplified) ===
    enable_message_batching: bool = Field(
        default=True, description="Enable message batching to prevent spam"
    )
    batch_delay_seconds: float = Field(
        default=15.0,
        description="Time to wait for additional messages before processing batch",
    )
    max_batch_size: int = Field(
        default=10, description="Maximum number of messages to batch together"
    )

    # === Spam Protection ===
    spam_protection_enabled: bool = Field(
        default=True, description="Enable spam protection mechanisms"
    )
    min_message_interval_seconds: float = Field(
        default=1,
        description="Minimum interval between processing messages from same user",
    )
    max_messages_per_minute: int = Field(
        default=20,
        description="Maximum messages per minute per user before rate limiting",
    )
    rate_limit_cooldown_seconds: int = Field(
        default=60, description="Cooldown period after rate limit is triggered"
    )

    # === Text-to-Speech (TTS) ===
    speech_play_chance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability (0.0-1.0) of sending audio response instead of text",
    )
    speech_config: SpeechConfig | None = Field(
        default=None,
        description="Optional SpeechConfig for TTS provider customization",
    )

    # === Error Handling ===
    retry_failed_messages: bool = Field(
        default=True, description="Retry processing failed messages"
    )
    max_retry_attempts: int = Field(
        default=3, description="Maximum number of retry attempts for failed messages"
    )
    retry_delay_seconds: float = Field(
        default=1.0, description="Delay between retry attempts"
    )

    # === Human-Like Delays ===
    enable_human_delays: bool = Field(
        default=False,
        description="Enable human-like delays for message processing to simulate realistic human behavior patterns",
    )
    min_read_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Minimum delay before marking message as read (seconds). Simulates time to read incoming messages.",
    )
    max_read_delay_seconds: float = Field(
        default=15.0,
        ge=0.0,
        description="Maximum delay before marking message as read (seconds). Prevents excessively long read delays.",
    )
    min_typing_delay_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum delay before sending response (seconds). Simulates time to compose a response.",
    )
    max_typing_delay_seconds: float = Field(
        default=45.0,
        ge=0.0,
        description="Maximum delay before sending response (seconds). Prevents excessively long typing delays.",
    )
    min_send_delay_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum delay before message transmission (seconds). Simulates final message review time.",
    )
    max_send_delay_seconds: float = Field(
        default=4.0,
        ge=0.0,
        description="Maximum delay before message transmission (seconds). Prevents excessively long send delays.",
    )
    enable_delay_jitter: bool = Field(
        default=True,
        description="Enable random variation (±20%) in delay calculations to prevent detectable patterns and simulate natural human behavior variability",
    )
    batch_read_compression_factor: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Compression factor (0.1-1.0) applied to batch read delays. Lower values simulate faster batch reading (e.g., 0.7 = 30% faster than reading individually)",
    )
