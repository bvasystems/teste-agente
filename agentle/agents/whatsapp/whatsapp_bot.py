from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import (
    Awaitable,
    Callable,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, cast
from dataclasses import dataclass

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.agents.agent import Agent
from agentle.agents.agent_input import AgentInput
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.agents.whatsapp.models.data import Data
from agentle.agents.whatsapp.models.message import Message
from agentle.agents.whatsapp.models.whatsapp_audio_message import WhatsAppAudioMessage
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_document_message import (
    WhatsAppDocumentMessage,
)
from agentle.agents.whatsapp.models.whatsapp_image_message import WhatsAppImageMessage
from agentle.agents.whatsapp.models.whatsapp_media_message import WhatsAppMediaMessage
from agentle.agents.whatsapp.models.whatsapp_message import WhatsAppMessage
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.models.whatsapp_text_message import WhatsAppTextMessage
from agentle.agents.whatsapp.models.whatsapp_video_message import WhatsAppVideoMessage
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import (
    WhatsAppWebhookPayload,
)
from agentle.agents.whatsapp.providers.base.whatsapp_provider import WhatsAppProvider
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import (
    EvolutionAPIProvider,
)
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.tools.tool import Tool
from agentle.generations.tools.tool_execution_result import ToolExecutionResult


if TYPE_CHECKING:
    from blacksheep import Application
    from blacksheep.server.openapi.v3 import OpenAPIHandler
    from blacksheep.server.routing import MountRegistry, Router
    from rodi import ContainerProtocol

try:
    import blacksheep
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CallbackWithContext:
    """Container for callback function with optional context."""

    callback: (
        Callable[
            [str, GeneratedAssistantMessage[Any] | None, dict[str, Any] | None], None
        ]
        | Callable[
            [str, GeneratedAssistantMessage[Any] | None, dict[str, Any] | None],
            Awaitable[None],
        ]
    )
    context: dict[str, Any] | None = None


class WhatsAppBot(BaseModel):
    """
    WhatsApp bot that wraps an Agentle agent with enhanced message batching and spam protection.

    Now uses the Agent's conversation store directly instead of managing contexts separately.
    """

    agent: Agent[Any]
    provider: WhatsAppProvider
    config: WhatsAppBotConfig = Field(default_factory=WhatsAppBotConfig)

    # REMOVED: context_manager field - no longer needed

    _running: bool = PrivateAttr(default=False)
    _webhook_handlers: MutableSequence[Callable[..., Any]] = PrivateAttr(
        default_factory=list
    )
    _batch_processors: MutableMapping[str, asyncio.Task[Any]] = PrivateAttr(
        default_factory=dict
    )
    _processing_locks: MutableMapping[str, asyncio.Lock] = PrivateAttr(
        default_factory=dict
    )
    _cleanup_task: Optional[asyncio.Task[Any]] = PrivateAttr(default=None)
    _response_callbacks: MutableSequence[CallbackWithContext] = PrivateAttr(
        default_factory=list
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __post_init__(self):
        """Validate that agent has conversation store configured."""
        if self.agent.conversation_store is None:
            raise ValueError(
                "Agent must have a conversation_store configured for WhatsApp integration. "
                + "Please set agent.conversation_store before creating WhatsAppBot."
            )

    def start(self) -> None:
        """Start the WhatsApp bot."""
        run_sync(self.start_async)

    def stop(self) -> None:
        """Stop the WhatsApp bot."""
        run_sync(self.stop_async)

    def change_instance(self, instance_name: str) -> None:
        """Change the instance of the WhatsApp bot."""
        provider = self.provider
        if isinstance(provider, EvolutionAPIProvider):
            provider.change_instance(instance_name)

    async def start_async(self) -> None:
        """Start the WhatsApp bot with proper initialization."""
        await self.provider.initialize()
        self._running = True

        # Start cleanup task for abandoned batch processors
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("WhatsApp bot started with message batching enabled")

    async def stop_async(self) -> None:
        """Stop the WhatsApp bot with proper cleanup."""
        self._running = False

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Cancel all batch processors
        for phone_number, task in self._batch_processors.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.debug(f"Cancelled batch processor for {phone_number}")

        self._batch_processors.clear()
        self._processing_locks.clear()

        await self.provider.shutdown()
        # REMOVED: context_manager.close() - no longer needed
        logger.info("WhatsApp bot stopped")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up abandoned batch processors."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_abandoned_processors()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_abandoned_processors(self) -> None:
        """Clean up batch processors that have been running too long."""
        abandoned_processors: MutableSequence[str] = []

        for phone_number, task in self._batch_processors.items():
            if task.done():
                abandoned_processors.append(phone_number)
                continue

            # Check if session is still processing
            session = await self.provider.get_session(phone_number)
            if session and session.is_batch_expired(
                self.config.max_batch_timeout_seconds * 2
            ):
                logger.warning(f"Found abandoned batch processor for {phone_number}")
                abandoned_processors.append(phone_number)
                task.cancel()

                # Reset session state
                session.reset_session()
                await self.provider.update_session(session)

        # Clean up abandoned processors
        for phone_number in abandoned_processors:
            if phone_number in self._batch_processors:
                del self._batch_processors[phone_number]
            if phone_number in self._processing_locks:
                del self._processing_locks[phone_number]

        if abandoned_processors:
            logger.info(
                f"Cleaned up {len(abandoned_processors)} abandoned batch processors"
            )

    async def handle_message(
        self, message: WhatsAppMessage
    ) -> GeneratedAssistantMessage[Any] | None:
        """
        Handle incoming WhatsApp message with enhanced error handling and batching.
        """
        logger.info("[MESSAGE_HANDLER] ═══════════ MESSAGE HANDLER ENTRY ═══════════")
        logger.info(
            f"[MESSAGE_HANDLER] Received message from {message.from_number}: ID={message.id}, Type={type(message).__name__}"
        )
        logger.info(
            f"[MESSAGE_HANDLER] Current response callbacks count: {len(self._response_callbacks)}"
        )

        # Log callback details
        for i, cb in enumerate(self._response_callbacks):
            logger.info(
                f"[MESSAGE_HANDLER] Callback {i + 1}: {cb.callback.__name__ if hasattr(cb.callback, '__name__') else 'unnamed'}, Context: {cb.context is not None}"
            )

        try:
            # Mark as read if configured
            if self.config.auto_read_messages:
                logger.debug(f"[MESSAGE_HANDLER] Marking message {message.id} as read")
                await self.provider.mark_message_as_read(message.id)

            # Get or create session
            logger.debug(f"[MESSAGE_HANDLER] Getting session for {message.from_number}")
            session = await self.provider.get_session(message.from_number)
            if not session:
                logger.error(
                    f"[MESSAGE_HANDLER] ❌ Failed to get session for {message.from_number}"
                )
                return

            logger.info(
                f"[SESSION_STATE] Session for {message.from_number}: is_processing={session.is_processing}, pending_messages={len(session.pending_messages)}, message_count={session.message_count}"
            )

            # Check rate limiting if spam protection is enabled
            if self.config.spam_protection_enabled:
                logger.debug(
                    f"[SPAM_PROTECTION] Checking rate limits for {message.from_number}"
                )
                can_process = session.update_rate_limiting(
                    self.config.max_messages_per_minute,
                    self.config.rate_limit_cooldown_seconds,
                )

                if not can_process:
                    logger.warning(
                        f"[SPAM_PROTECTION] Rate limited user {message.from_number}"
                    )
                    if session.is_rate_limited:
                        await self._send_rate_limit_message(message.from_number)
                    return None

            # Check welcome message for first interaction
            if (
                await cast(
                    ConversationStore, self.agent.conversation_store
                ).get_conversation_history_length(message.from_number)
                == 0
                and self.config.welcome_message
            ):
                logger.info(
                    f"[WELCOME] Sending welcome message to {message.from_number}"
                )
                await self.provider.send_text_message(
                    message.from_number, self.config.welcome_message
                )
                session.message_count += 1
                await self.provider.update_session(session)

                # Get updated session after welcome message
                updated_session = await self.provider.get_session(message.from_number)
                if updated_session:
                    session = updated_session
                else:
                    logger.warning(
                        f"[WELCOME] Could not retrieve updated session for {message.from_number}"
                    )

            # Handle message based on batching configuration
            response = None
            if self.config.enable_message_batching:
                logger.info(
                    f"[BATCHING] 📦 Processing message with batching for {message.from_number}"
                )
                logger.info(
                    f"[BATCHING] Batching config: delay={self.config.batch_delay_seconds}s, max_size={self.config.max_batch_size}, max_timeout={self.config.max_batch_timeout_seconds}s"
                )
                response = await self._handle_message_with_batching(message, session)
            else:
                logger.info(
                    f"[IMMEDIATE] ⚡ Processing message immediately for {message.from_number}"
                )
                response = await self._process_single_message(message, session)

            logger.info(
                f"[MESSAGE_HANDLER] ✅ Message handling completed. Response generated: {response is not None}"
            )
            return response

        except Exception as e:
            logger.error(
                f"[MESSAGE_HANDLER_ERROR] ❌ Error handling message from {message.from_number}: {e}",
                exc_info=True,
            )
            if self._is_user_facing_error(e):
                await self._send_error_message(message.from_number, message.id)
            return None
        finally:
            logger.info(
                "[MESSAGE_HANDLER] ═══════════ MESSAGE HANDLER EXIT ═══════════"
            )

    async def _handle_message_with_batching(
        self, message: WhatsAppMessage, session: WhatsAppSession
    ) -> GeneratedAssistantMessage[Any] | None:
        """Handle message with improved batching logic and atomic state management."""
        phone_number = message.from_number

        logger.info("[BATCHING] ═══════════ BATCH HANDLING START ═══════════")
        logger.info(f"[BATCHING] Phone: {phone_number}")
        logger.info(
            f"[BATCHING] Current session state: processing={session.is_processing}, pending={len(session.pending_messages)}"
        )
        logger.info(
            f"[BATCHING] Current response callbacks count: {len(self._response_callbacks)}"
        )

        try:
            # Get or create processing lock for this user
            if phone_number not in self._processing_locks:
                logger.info(
                    f"[BATCHING] Creating new processing lock for {phone_number}"
                )
                self._processing_locks[phone_number] = asyncio.Lock()

            async with self._processing_locks[phone_number]:
                logger.info(f"[BATCHING] Acquired processing lock for {phone_number}")

                # Re-fetch session to ensure we have latest state
                current_session = await self.provider.get_session(phone_number)
                if not current_session:
                    logger.error(f"[BATCHING] ❌ Lost session for {phone_number}")
                    return None

                # Convert message to storable format
                message_data = await self._message_to_dict(message)
                logger.info(
                    f"[BATCHING] Converted message to dict: {message_data.get('id')}"
                )

                # Atomic session update with validation
                success = await self._atomic_session_update(
                    phone_number, current_session, message_data
                )

                if not success:
                    logger.error(
                        f"[BATCHING] ❌ Failed to update session for {phone_number}"
                    )
                    logger.info("[BATCHING] 🔄 Falling back to immediate processing")
                    return await self._process_single_message(message, current_session)

                # Re-fetch session after update to get latest processing state
                updated_session = await self.provider.get_session(phone_number)
                if not updated_session:
                    logger.error(
                        f"[BATCHING] ❌ Lost session after update for {phone_number}"
                    )
                    return None

                logger.info(
                    f"[BATCHING] Updated session state: processing={updated_session.is_processing}, token={updated_session.processing_token}, pending={len(updated_session.pending_messages)}"
                )

                # Only start processor if we successfully initiated processing
                if (
                    updated_session.is_processing
                    and updated_session.processing_token
                    and phone_number not in self._batch_processors
                ):
                    logger.info(
                        f"[BATCHING] 🚀 Starting new batch processor for {phone_number}"
                    )
                    logger.info(
                        f"[BATCHING] Processing token: {updated_session.processing_token}"
                    )
                    self._batch_processors[phone_number] = asyncio.create_task(
                        self._batch_processor(
                            phone_number, updated_session.processing_token
                        )
                    )
                    logger.info(
                        f"[BATCHING] ✅ Batch processor task created for {phone_number}"
                    )
                else:
                    logger.info(
                        f"[BATCHING] Message added to existing batch for {phone_number}"
                    )
                    logger.info(
                        f"[BATCHING] Existing processor active: {phone_number in self._batch_processors}"
                    )

                # Return None for batched messages since they're processed asynchronously
                logger.info("[BATCHING] Returning None (batched processing)")
                return None

        except Exception as e:
            logger.error(
                f"[BATCHING_ERROR] ❌ Error in message batching for {phone_number}: {e}",
                exc_info=True,
            )
            # Always fall back to immediate processing on error
            try:
                logger.info("[BATCHING] 🔄 Attempting fallback to immediate processing")
                return await self._process_single_message(message, session)
            except Exception as fallback_error:
                logger.error(
                    f"[FALLBACK_ERROR] ❌ Fallback processing failed: {fallback_error}",
                    exc_info=True,
                )
                await self._send_error_message(message.from_number, message.id)
                return None
        finally:
            logger.info("[BATCHING] ═══════════ BATCH HANDLING END ═══════════")

    async def _atomic_session_update(
        self, phone_number: str, session: WhatsAppSession, message_data: dict[str, Any]
    ) -> bool:
        """Atomically update session with proper state transitions."""
        try:
            # Add message to pending queue
            session.add_pending_message(message_data)

            # If not currently processing, transition to processing state
            if not session.is_processing:
                processing_token = session.start_batch_processing(
                    self.config.max_batch_timeout_seconds
                )

                # Validate the state transition worked
                if not session.is_processing or not session.processing_token:
                    logger.error(
                        f"[ATOMIC_UPDATE] Failed to start processing for {phone_number}"
                    )
                    return False

                logger.info(
                    f"[ATOMIC_UPDATE] Started processing for {phone_number} with token {processing_token}"
                )

            # Persist the updated session
            await self.provider.update_session(session)

            # Verify the session was persisted correctly by re-reading
            verification_session = await self.provider.get_session(phone_number)
            if not verification_session:
                logger.error(
                    f"[ATOMIC_UPDATE] Session disappeared after update for {phone_number}"
                )
                return False

            # Verify critical state is preserved
            if verification_session.is_processing != session.is_processing:
                logger.error(
                    f"[ATOMIC_UPDATE] Processing state not persisted for {phone_number}"
                )
                return False

            if len(verification_session.pending_messages) != len(
                session.pending_messages
            ):
                logger.error(
                    f"[ATOMIC_UPDATE] Pending messages not persisted for {phone_number}"
                )
                return False

            return True

        except Exception as e:
            logger.error(
                f"[ATOMIC_UPDATE] Failed atomic session update for {phone_number}: {e}"
            )
            return False

    async def _batch_processor(self, phone_number: str, processing_token: str) -> None:
        """
        Background task to process batched messages for a user with improved reliability.
        """
        logger.info("[BATCH_PROCESSOR] ═══════════ BATCH PROCESSOR START ═══════════")
        logger.info(
            f"[BATCH_PROCESSOR] Phone: {phone_number}, Token: {processing_token}"
        )
        logger.info(
            f"[BATCH_PROCESSOR] Current response callbacks count: {len(self._response_callbacks)}"
        )

        iteration_count = 0
        max_iterations = 1000  # Safety limit to prevent infinite loops
        batch_processed = False

        try:
            while (
                self._running
                and not batch_processed
                and iteration_count < max_iterations
            ):
                iteration_count += 1

                # Log early iterations for debugging
                if iteration_count <= 10:
                    logger.info(
                        f"[BATCH_PROCESSOR] 🔄 ENTERING iteration {iteration_count} for {phone_number}"
                    )

                try:
                    # Get current session
                    session = await self.provider.get_session(phone_number)
                    if not session:
                        logger.error(
                            f"[BATCH_PROCESSOR] ❌ No session found for {phone_number}, exiting at iteration {iteration_count}"
                        )
                        break

                    # Validate processing token
                    if session.processing_token != processing_token:
                        logger.warning(
                            f"[BATCH_PROCESSOR] ⚠️ Token mismatch for {phone_number}, exiting. Expected: {processing_token}, Got: {session.processing_token}"
                        )
                        break

                    if not session.is_processing:
                        logger.info(
                            f"[BATCH_PROCESSOR] ℹ️ Session no longer processing for {phone_number}, exiting at iteration {iteration_count}"
                        )
                        break

                    # Check for pending messages
                    if not session.pending_messages:
                        logger.warning(
                            f"[BATCH_PROCESSOR] ⚠️ No pending messages for {phone_number}, exiting at iteration {iteration_count}"
                        )
                        break

                    # Log session state for debugging
                    if iteration_count <= 20:
                        logger.info(
                            f"[BATCH_PROCESSOR] Session state for {phone_number}: pending_messages={len(session.pending_messages)}, batch_timeout_at={session.batch_timeout_at}, batch_started_at={session.batch_started_at}, iteration={iteration_count}"
                        )

                    # Check if batch should be processed
                    should_process = session.should_process_batch(
                        self.config.batch_delay_seconds,
                        self.config.max_batch_timeout_seconds,
                    )

                    # Check if max batch size reached
                    if len(session.pending_messages) >= self.config.max_batch_size:
                        logger.info(
                            f"[BATCH_PROCESSOR] 📏 Max batch size ({self.config.max_batch_size}) reached for {phone_number}, processing immediately"
                        )
                        should_process = True

                    # Check if batch has expired
                    if session.is_batch_expired(self.config.max_batch_timeout_seconds):
                        logger.info(
                            f"[BATCH_PROCESSOR] ⏰ Batch expired for {phone_number}, processing immediately"
                        )
                        should_process = True

                    # Log the decision for debugging
                    if iteration_count <= 20 or should_process:
                        logger.info(
                            f"[BATCH_PROCESSOR] Should process batch for {phone_number}: {should_process} (iteration {iteration_count}, messages={len(session.pending_messages)})"
                        )

                    if should_process:
                        logger.info(
                            f"[BATCH_PROCESSOR] 🚀 Batch ready for processing for {phone_number} (condition met after {iteration_count} iterations)"
                        )
                        logger.info(
                            f"[BATCH_PROCESSOR] About to call _process_message_batch with {len(self._response_callbacks)} callbacks registered"
                        )

                        await self._process_message_batch(
                            phone_number, session, processing_token
                        )

                        logger.info(
                            f"[BATCH_PROCESSOR] ✅ _process_message_batch completed for {phone_number}"
                        )
                        batch_processed = True
                        break

                except Exception as e:
                    logger.error(
                        f"[BATCH_PROCESSOR_ERROR] ❌ Error in batch processing loop for {phone_number}: {e}",
                        exc_info=True,
                    )
                    # Try to clean up the session state
                    try:
                        session = await self.provider.get_session(phone_number)
                        if session:
                            logger.debug(
                                f"[BATCH_PROCESSOR] 🧹 Cleaning up session state for {phone_number}"
                            )
                            session.finish_batch_processing(processing_token)
                            await self.provider.update_session(session)
                    except Exception as cleanup_error:
                        logger.error(
                            f"[BATCH_PROCESSOR] ❌ Failed to cleanup session for {phone_number}: {cleanup_error}"
                        )
                    break

                # Add delay between iterations
                await asyncio.sleep(0.1)  # Small polling interval

            # Log why we exited the loop
            logger.info(
                f"[BATCH_PROCESSOR] Loop exit for {phone_number}: self._running={self._running}, batch_processed={batch_processed}, iterations={iteration_count}, max_iterations={max_iterations}"
            )

        except asyncio.CancelledError:
            logger.info(
                f"[BATCH_PROCESSOR] ⚠️ Batch processor for {phone_number} was cancelled"
            )
            raise
        except Exception as e:
            logger.error(
                f"[BATCH_PROCESSOR_CRITICAL] ❌ Critical error in batch processor for {phone_number}: {e}",
                exc_info=True,
            )
        finally:
            # Clean up
            logger.info(
                f"[BATCH_PROCESSOR] 🧹 Cleaning up batch processor for {phone_number}"
            )
            if phone_number in self._batch_processors:
                del self._batch_processors[phone_number]
                logger.debug(
                    f"[BATCH_PROCESSOR] ✅ Removed batch processor task for {phone_number}"
                )

            # Ensure session is not left in processing state
            try:
                cleanup_session = await self.provider.get_session(phone_number)
                if cleanup_session and cleanup_session.is_processing:
                    logger.warning(
                        f"[BATCH_PROCESSOR] 🧹 Cleaning up processing state for {phone_number}"
                    )
                    cleanup_session.finish_batch_processing(processing_token)
                    await self.provider.update_session(cleanup_session)
            except Exception as cleanup_error:
                logger.error(
                    f"[BATCH_PROCESSOR] ❌ Final cleanup error for {phone_number}: {cleanup_error}"
                )

            logger.info("[BATCH_PROCESSOR] ═══════════ BATCH PROCESSOR END ═══════════")

    async def _process_message_batch(
        self, phone_number: str, session: WhatsAppSession, processing_token: str
    ) -> GeneratedAssistantMessage[Any] | None:
        """Process a batch of messages for a user with token validation."""
        logger.info("[BATCH_PROCESSING] ═══════════ BATCH PROCESSING START ═══════════")
        logger.info(
            f"[BATCH_PROCESSING] Phone: {phone_number}, Token: {processing_token}"
        )
        logger.info(
            f"[BATCH_PROCESSING] Pending messages count: {len(session.pending_messages)}"
        )
        logger.info(
            f"[BATCH_PROCESSING] Current response callbacks count: {len(self._response_callbacks)}"
        )

        if not session.pending_messages:
            logger.warning(
                f"[BATCH_PROCESSING] ⚠️ No pending messages for {phone_number}, finishing batch processing"
            )
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)
            return None

        try:
            # Show typing indicator
            if self.config.typing_indicator:
                logger.debug(
                    f"[BATCH_PROCESSING] Sending typing indicator to {phone_number}"
                )
                await self.provider.send_typing_indicator(
                    phone_number, self.config.typing_duration
                )

            # Get all pending messages
            pending_messages = session.clear_pending_messages()
            logger.info(
                f"[BATCH_PROCESSING] 📦 Processing batch of {len(pending_messages)} messages for {phone_number}"
            )

            # Convert message batch to agent input
            logger.debug(
                f"[BATCH_PROCESSING] Converting message batch to agent input for {phone_number}"
            )
            agent_input = await self._convert_message_batch_to_input(
                pending_messages, session
            )

            # Process with agent
            logger.info(f"[BATCH_PROCESSING] 🤖 Running agent for {phone_number}")
            response = await self._process_with_agent(agent_input, session)
            logger.info(
                f"[BATCH_PROCESSING] ✅ Agent processing complete for {phone_number}"
            )
            logger.info(
                f"[BATCH_PROCESSING] Response generated: {response is not None}"  # type: ignore
            )

            if response:
                logger.info(
                    f"[BATCH_PROCESSING] Response text length: {len(response.text)}"
                )

            # Send response (use the first message ID for reply if quoting is enabled)
            first_message_id = (
                pending_messages[0].get("id")
                if pending_messages and self.config.quote_messages
                else None
            )
            logger.info(
                f"[BATCH_PROCESSING] 📤 Sending response to {phone_number} (quote_messages={self.config.quote_messages}, reply to: {first_message_id})"
            )
            await self._send_response(phone_number, response, first_message_id)

            # Update session
            session.message_count += len(pending_messages)
            session.last_activity = datetime.now()

            # Finish batch processing with token validation
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)

            logger.info(
                f"[BATCH_PROCESSING] ✅ Successfully processed batch for {phone_number}. Total messages processed: {session.message_count}"
            )

            # Call response callbacks - THIS IS CRITICAL
            logger.info(
                f"[BATCH_PROCESSING] 📞 About to call response callbacks for {phone_number}"
            )
            await self._call_response_callbacks(phone_number, response)
            logger.info(
                f"[BATCH_PROCESSING] ✅ Response callbacks completed for {phone_number}"
            )

            return response

        except Exception as e:
            logger.error(
                f"[BATCH_PROCESSING_ERROR] ❌ Error processing message batch for {phone_number}: {e}",
                exc_info=True,
            )
            await self._send_error_message(phone_number)
            # Ensure session state is cleaned up even on error
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)

            # Call response callbacks with None response on error - THIS IS CRITICAL
            logger.info(
                "[BATCH_PROCESSING] 📞 Calling response callbacks with None response due to error"
            )
            await self._call_response_callbacks(phone_number, None)
            logger.info("[BATCH_PROCESSING] ✅ Error response callbacks completed")
            raise
        finally:
            logger.info(
                "[BATCH_PROCESSING] ═══════════ BATCH PROCESSING END ═══════════"
            )

    async def _process_single_message(
        self, message: WhatsAppMessage, session: WhatsAppSession
    ) -> GeneratedAssistantMessage[Any]:
        """Process a single message immediately with quote message support."""
        logger.info(
            "[SINGLE_MESSAGE] ═══════════ SINGLE MESSAGE PROCESSING START ═══════════"
        )
        logger.info(f"[SINGLE_MESSAGE] Phone: {message.from_number}")
        logger.info(
            f"[SINGLE_MESSAGE] Current response callbacks count: {len(self._response_callbacks)}"
        )

        try:
            # Show typing indicator
            if self.config.typing_indicator:
                logger.debug(
                    f"[SINGLE_MESSAGE] Sending typing indicator to {message.from_number}"
                )
                await self.provider.send_typing_indicator(
                    message.from_number, self.config.typing_duration
                )

            # Convert WhatsApp message to agent input
            logger.debug(
                f"[SINGLE_MESSAGE] Converting message to agent input for {message.from_number}"
            )
            agent_input = await self._convert_message_to_input(message, session)

            # Process with agent
            logger.info(f"[SINGLE_MESSAGE] 🤖 Running agent for {message.from_number}")
            response = await self._process_with_agent(agent_input, session)
            logger.info(
                f"[SINGLE_MESSAGE] ✅ Agent processing complete for {message.from_number}"
            )
            logger.info(f"[SINGLE_MESSAGE] Response generated: {response is not None}")  # type: ignore

            if response:
                logger.info(
                    f"[SINGLE_MESSAGE] Response text length: {len(response.text)}"
                )

            # Send response (quote message if enabled)
            quote_message_id = message.id if self.config.quote_messages else None
            logger.info(
                f"[SINGLE_MESSAGE] 📤 Sending response to {message.from_number} (quote_messages={self.config.quote_messages}, quote_id={quote_message_id})"
            )
            await self._send_response(message.from_number, response, quote_message_id)

            # Update session
            session.message_count += 1
            session.last_activity = datetime.now()
            await self.provider.update_session(session)

            logger.info(
                f"[SINGLE_MESSAGE] ✅ Successfully processed single message for {message.from_number}. Total messages processed: {session.message_count}"
            )

            # Call response callbacks - THIS IS CRITICAL
            logger.info(
                f"[SINGLE_MESSAGE] 📞 About to call response callbacks for {message.from_number}"
            )
            await self._call_response_callbacks(message.from_number, response)
            logger.info(
                f"[SINGLE_MESSAGE] ✅ Response callbacks completed for {message.from_number}"
            )

            return response

        except Exception as e:
            logger.error(
                f"[SINGLE_MESSAGE_ERROR] ❌ Error processing single message: {e}",
                exc_info=True,
            )

            # Call response callbacks with None response on error - THIS IS CRITICAL
            logger.info(
                "[SINGLE_MESSAGE] 📞 Calling response callbacks with None response due to error"
            )
            await self._call_response_callbacks(message.from_number, None)
            logger.info("[SINGLE_MESSAGE] ✅ Error response callbacks completed")
            raise
        finally:
            logger.info(
                "[SINGLE_MESSAGE] ═══════════ SINGLE MESSAGE PROCESSING END ═══════════"
            )

    async def _message_to_dict(self, message: WhatsAppMessage) -> dict[str, Any]:
        """Convert WhatsApp message to dictionary for storage."""
        message_data: dict[str, Any] = {
            "id": message.id,
            "type": message.__class__.__name__,
            "from_number": message.from_number,
            "to_number": message.to_number,
            "timestamp": message.timestamp.isoformat(),
            "push_name": message.push_name,
        }

        # Add type-specific data
        if isinstance(message, WhatsAppTextMessage):
            message_data["text"] = message.text
        elif isinstance(message, WhatsAppMediaMessage):
            message_data.update(
                {
                    "media_url": message.media_url,
                    "media_mime_type": message.media_mime_type,
                    "caption": message.caption,
                    "filename": getattr(message, "filename", None),
                }
            )

        logger.debug(f"[MESSAGE_TO_DICT] Converted message {message.id} to dict")
        return message_data

    async def _convert_message_batch_to_input(
        self, message_batch: Sequence[dict[str, Any]], session: WhatsAppSession
    ) -> Any:
        """Convert a batch of messages to agent input using phone number as chat_id."""
        logger.info(
            f"[BATCH_CONVERSION] Converting batch of {len(message_batch)} messages to agent input"
        )

        parts: MutableSequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ] = []

        # Add batch header if multiple messages
        if len(message_batch) > 1:
            parts.append(
                TextPart(
                    text=f"[Batch of {len(message_batch)} messages received together]"
                )
            )

        # Process each message in the batch
        for i, msg_data in enumerate(message_batch):
            logger.debug(
                f"[BATCH_CONVERSION] Processing message {i + 1}/{len(message_batch)}: {msg_data.get('id')}"
            )

            if i > 0:  # Add separator between messages
                parts.append(TextPart(text="\n\n"))

            # Handle text messages
            if msg_data["type"] == "WhatsAppTextMessage":
                text = msg_data.get("text", "")
                if text:
                    parts.append(TextPart(text=text))
                    logger.debug(f"[BATCH_CONVERSION] Added text part: {text[:50]}...")

            # Handle media messages
            elif msg_data["type"] in [
                "WhatsAppImageMessage",
                "WhatsAppDocumentMessage",
                "WhatsAppAudioMessage",
                "WhatsAppVideoMessage",
            ]:
                try:
                    logger.debug(
                        f"[BATCH_CONVERSION] Downloading media for message {msg_data['id']}"
                    )
                    media_data = await self.provider.download_media(msg_data["id"])
                    parts.append(
                        FilePart(data=media_data.data, mime_type=media_data.mime_type)
                    )
                    logger.debug(
                        f"[BATCH_CONVERSION] Successfully downloaded media for {msg_data['id']}"
                    )

                    # Add caption if present
                    caption = msg_data.get("caption")
                    if caption:
                        parts.append(TextPart(text=f"Caption: {caption}"))
                        logger.debug(f"[BATCH_CONVERSION] Added caption: {caption}")

                except Exception as e:
                    logger.error(
                        f"[BATCH_CONVERSION] Failed to download media from batch: {e}"
                    )
                    parts.append(TextPart(text="[Media file - failed to download]"))

        # If no parts were added, add a placeholder
        if not parts:
            logger.warning(
                "[BATCH_CONVERSION] No parts were created, adding placeholder"
            )
            parts.append(TextPart(text="[Empty message batch]"))

        # Create user message with first message's push name
        first_message = message_batch[0] if message_batch else {}
        push_name = first_message.get("push_name", "User")
        user_message = UserMessage.create_named(parts=parts, name=push_name)
        logger.debug(f"[BATCH_CONVERSION] Created user message with name: {push_name}")

        # Simply return the user message - Agent will handle conversation history via chat_id
        return user_message

    async def handle_webhook(
        self,
        payload: WhatsAppWebhookPayload,
        callback: Callable[
            [str, GeneratedAssistantMessage[Any] | None, dict[str, Any] | None], None
        ]
        | Callable[
            [str, GeneratedAssistantMessage[Any] | None, dict[str, Any] | None],
            Awaitable[None],
        ]
        | None = None,
        callback_context: dict[str, Any] | None = None,
    ) -> GeneratedAssistantMessage[Any] | None:
        """
        Handle incoming webhook from WhatsApp.
        """
        logger.info("[WEBHOOK] ═══════════ WEBHOOK HANDLER ENTRY ═══════════")
        logger.info(f"[WEBHOOK] Received webhook event: {payload.event}")
        logger.info(f"[WEBHOOK] Callback provided: {callback is not None}")
        logger.info(
            f"[WEBHOOK] Callback context provided: {callback_context is not None}"
        )
        logger.info(
            f"[WEBHOOK] Current response callbacks count: {len(self._response_callbacks)}"
        )

        # FOR BATCHING: Register callback permanently, don't remove it in finally block
        # The batch processor will call it when the batch is processed
        if callback:
            logger.info("[WEBHOOK] Processing callback registration for batching...")
            callback_with_context = CallbackWithContext(
                callback=callback, context=callback_context
            )

            # Check if this exact callback is already registered to avoid duplicates
            callback_exists = any(
                existing.callback == callback and existing.context == callback_context
                for existing in self._response_callbacks
            )

            logger.info(f"[WEBHOOK] Callback already exists: {callback_exists}")

            if not callback_exists:
                self._response_callbacks.append(callback_with_context)
                logger.info(
                    f"[WEBHOOK] ✅ Added callback for batching. Total callbacks: {len(self._response_callbacks)}"
                )
                logger.info(
                    "[WEBHOOK] ⚠️  IMPORTANT: Callback will be called when batch is processed, not removed immediately"
                )
            else:
                logger.warning("[WEBHOOK] ⚠️ Duplicate callback not added")

        try:
            logger.info("[WEBHOOK] Starting webhook validation...")
            await self.provider.validate_webhook(payload)
            logger.info("[WEBHOOK] ✅ Webhook validation passed")

            response = None

            # Handle Evolution API events
            if payload.event == "messages.upsert":
                logger.info("[WEBHOOK] 🔄 Handling messages.upsert event")
                response = await self._handle_message_upsert(payload)
                logger.info(
                    f"[WEBHOOK] Message upsert response: {response is not None}"
                )
            elif payload.event == "messages.update":
                logger.info("[WEBHOOK] 🔄 Handling messages.update event")
                await self._handle_message_update(payload)
            elif payload.event == "connection.update":
                logger.info("[WEBHOOK] 🔄 Handling connection.update event")
                await self._handle_connection_update(payload)
            # Handle Meta API events
            elif payload.entry:
                logger.info("[WEBHOOK] 🔄 Handling Meta API webhook")
                response = await self._handle_meta_webhook(payload)
                logger.info(f"[WEBHOOK] Meta webhook response: {response is not None}")

            # Call custom handlers
            logger.info(
                f"[WEBHOOK] Calling {len(self._webhook_handlers)} custom webhook handlers"
            )
            for i, handler in enumerate(self._webhook_handlers):
                logger.debug(
                    f"[WEBHOOK] Calling custom webhook handler {i + 1}/{len(self._webhook_handlers)}"
                )
                await handler(payload)

            logger.info(
                f"[WEBHOOK] ✅ Webhook processing completed. Response generated: {response is not None}"
            )
            return response

        except Exception as e:
            logger.error(
                f"[WEBHOOK_ERROR] ❌ Error handling webhook: {e}", exc_info=True
            )
            return None
        finally:
            # FOR BATCHING: DON'T remove the callback here since batch processing is asynchronous
            # The callback will be called later when the batch is processed
            logger.info(
                "[WEBHOOK] ℹ️  Callback will remain registered for batch processing"
            )
            logger.info(
                f"[WEBHOOK] Current callbacks after webhook: {len(self._response_callbacks)}"
            )
            logger.info("[WEBHOOK] ═══════════ WEBHOOK HANDLER EXIT ═══════════")

    def to_blacksheep_app(
        self,
        *,
        router: "Router | None" = None,
        services: "ContainerProtocol | None" = None,
        show_error_details: bool = False,
        mount: "MountRegistry | None" = None,
        docs: "OpenAPIHandler | None" = None,
        webhook_path: str = "/webhook/whatsapp",
    ) -> "Application":
        """
        Convert the WhatsApp bot to a BlackSheep ASGI application.

        Args:
            router: Optional router to use
            services: Optional services container
            show_error_details: Whether to show error details in responses
            mount: Optional mount registry
            docs: Optional OpenAPI handler
            webhook_path: Path for the webhook endpoint

        Returns:
            BlackSheep application with webhook endpoint
        """
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application(
            router=router,
            services=services,
            show_error_details=show_error_details,
            mount=mount,
        )

        if docs is None:
            docs = OpenAPIHandler(
                ui_path="/openapi",
                info=Info(title="Agentle WhatsApp Bot API", version="1.0.0"),
            )
            docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))

        docs.bind_app(app)

        @blacksheep.post(webhook_path)
        async def _(
            webhook_payload: blacksheep.FromJSON[WhatsAppWebhookPayload],
        ) -> blacksheep.Response:
            """
            Handle incoming WhatsApp webhooks.

            Args:
                webhook_payload: The webhook payload from WhatsApp

            Returns:
                Success response
            """
            try:
                # Process the webhook payload
                payload_data: WhatsAppWebhookPayload = webhook_payload.value
                logger.info(
                    f"[WEBHOOK_ENDPOINT] Received webhook payload: {payload_data.event}"
                )
                await self.handle_webhook(payload_data)

                # Return success response
                return blacksheep.json(
                    {"status": "success", "message": "Webhook processed"}
                )

            except Exception as e:
                logger.error(
                    f"[WEBHOOK_ENDPOINT_ERROR] Webhook processing error: {e}",
                    exc_info=True,
                )
                return blacksheep.json(
                    {"status": "error", "message": "Failed to process webhook"},
                    status=500,
                )

        @app.on_start
        async def _() -> None:
            await self.start_async()

        return app

    def add_webhook_handler(self, handler: Callable[..., Any]) -> None:
        """Add custom webhook handler."""
        self._webhook_handlers.append(handler)

    def add_response_callback(
        self,
        callback: Callable[
            [str, GeneratedAssistantMessage[Any] | None, dict[str, Any] | None], None
        ]
        | Callable[
            [str, GeneratedAssistantMessage[Any] | None, dict[str, Any] | None],
            Awaitable[None],
        ],
        context: dict[str, Any] | None = None,
        allow_duplicates: bool = False,
    ) -> None:
        """Add callback to be called when a response is generated."""
        logger.info("[ADD_CALLBACK] ═══════════ ADDING RESPONSE CALLBACK ═══════════")
        logger.info(
            f"[ADD_CALLBACK] Callback function: {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}"
        )
        logger.info(f"[ADD_CALLBACK] Context provided: {context is not None}")
        logger.info(f"[ADD_CALLBACK] Allow duplicates: {allow_duplicates}")
        logger.info(
            f"[ADD_CALLBACK] Current callbacks count: {len(self._response_callbacks)}"
        )

        callback_with_context = CallbackWithContext(callback=callback, context=context)

        if not allow_duplicates:
            # Check if this exact callback+context combination already exists
            callback_exists = any(
                existing.callback == callback and existing.context == context
                for existing in self._response_callbacks
            )
            if callback_exists:
                logger.warning(
                    f"[ADD_CALLBACK] ⚠️ Duplicate callback registration prevented for {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}"
                )
                return

        self._response_callbacks.append(callback_with_context)
        logger.info(
            f"[ADD_CALLBACK] ✅ Callback added successfully. Total callbacks: {len(self._response_callbacks)}"
        )
        logger.info("[ADD_CALLBACK] ═══════════ CALLBACK ADDED ═══════════")

    def remove_response_callback(
        self,
        callback: Callable[
            [str, GeneratedAssistantMessage[Any] | None, dict[str, Any] | None], None
        ],
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Remove a specific callback from the registered callbacks."""
        logger.info(
            "[REMOVE_CALLBACK] ═══════════ REMOVING RESPONSE CALLBACK ═══════════"
        )
        logger.info(
            f"[REMOVE_CALLBACK] Callback function: {callback.__name__ if hasattr(callback, '__name__') else 'unnamed'}"
        )
        logger.info(f"[REMOVE_CALLBACK] Context provided: {context is not None}")
        logger.info(
            f"[REMOVE_CALLBACK] Current callbacks count: {len(self._response_callbacks)}"
        )

        for i, existing in enumerate(self._response_callbacks):
            if existing.callback == callback and existing.context == context:
                self._response_callbacks.pop(i)
                logger.info(
                    f"[REMOVE_CALLBACK] ✅ Callback removed successfully. Remaining callbacks: {len(self._response_callbacks)}"
                )
                logger.info(
                    "[REMOVE_CALLBACK] ═══════════ CALLBACK REMOVED ═══════════"
                )
                return True

        logger.warning("[REMOVE_CALLBACK] ⚠️ Callback not found for removal")
        logger.info("[REMOVE_CALLBACK] ═══════════ CALLBACK NOT FOUND ═══════════")
        return False

    def clear_response_callbacks(self) -> int:
        """Remove all registered response callbacks."""
        logger.info(
            "[CLEAR_CALLBACKS] ═══════════ CLEARING ALL RESPONSE CALLBACKS ═══════════"
        )
        count = len(self._response_callbacks)
        logger.info(f"[CLEAR_CALLBACKS] Clearing {count} callbacks")
        self._response_callbacks.clear()
        logger.info("[CLEAR_CALLBACKS] ✅ All callbacks cleared")
        logger.info("[CLEAR_CALLBACKS] ═══════════ CALLBACKS CLEARED ═══════════")
        return count

    async def _call_response_callbacks(
        self, phone_number: str, response: GeneratedAssistantMessage[Any] | None
    ) -> None:
        """Call all registered response callbacks."""
        logger.info("[CALLBACKS] ═══════════ CALLING RESPONSE CALLBACKS ═══════════")
        logger.info(f"[CALLBACKS] Phone number: {phone_number}")
        logger.info(f"[CALLBACKS] Response provided: {response is not None}")
        logger.info(
            f"[CALLBACKS] Total callbacks to call: {len(self._response_callbacks)}"
        )

        if response:
            logger.info(f"[CALLBACKS] Response text length: {len(response.text)}")
            logger.debug(f"[CALLBACKS] Response text preview: {response.text[:100]}...")

        if not self._response_callbacks:
            logger.warning("[CALLBACKS] ⚠️ No callbacks registered to call!")
            return

        for i, callback_with_context in enumerate(self._response_callbacks):
            logger.info(
                f"[CALLBACKS] 🔄 Calling callback {i + 1}/{len(self._response_callbacks)}"
            )
            logger.info(
                f"[CALLBACKS] Callback function: {callback_with_context.callback.__name__ if hasattr(callback_with_context.callback, '__name__') else 'unnamed'}"
            )
            logger.info(
                f"[CALLBACKS] Callback context keys: {list(callback_with_context.context.keys()) if callback_with_context.context else 'None'}"
            )

            try:
                if inspect.iscoroutinefunction(callback_with_context.callback):
                    logger.info(f"[CALLBACKS] Executing async callback {i + 1}")
                    await callback_with_context.callback(
                        phone_number, response, callback_with_context.context
                    )
                    logger.info(
                        f"[CALLBACKS] ✅ Async callback {i + 1} completed successfully"
                    )
                else:
                    logger.info(f"[CALLBACKS] Executing sync callback {i + 1}")
                    callback_with_context.callback(
                        phone_number, response, callback_with_context.context
                    )
                    logger.info(
                        f"[CALLBACKS] ✅ Sync callback {i + 1} completed successfully"
                    )
            except Exception as e:
                logger.error(
                    f"[CALLBACKS] ❌ Error calling callback {i + 1} for {phone_number}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"[CALLBACKS] ✅ All {len(self._response_callbacks)} callbacks processed"
        )
        logger.info("[CALLBACKS] ═══════════ CALLBACKS COMPLETE ═══════════")

    async def _convert_message_to_input(
        self, message: WhatsAppMessage, session: WhatsAppSession
    ) -> Any:
        """Convert WhatsApp message to agent input using phone number as chat_id."""
        logger.info(
            f"[SINGLE_CONVERSION] Converting single message to agent input for {message.from_number}"
        )

        parts: MutableSequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ] = []

        # Handle text messages
        if isinstance(message, WhatsAppTextMessage):
            parts.append(TextPart(text=message.text))
            logger.debug(f"[SINGLE_CONVERSION] Added text part: {message.text[:50]}...")

        # Handle media messages
        elif isinstance(message, WhatsAppMediaMessage):
            try:
                logger.debug(
                    f"[SINGLE_CONVERSION] Downloading media for message {message.id}"
                )
                media_data = await self.provider.download_media(message.id)
                parts.append(
                    FilePart(data=media_data.data, mime_type=media_data.mime_type)
                )
                logger.debug(
                    f"[SINGLE_CONVERSION] Successfully downloaded media for {message.id}"
                )

                # Add caption if present
                if message.caption:
                    parts.append(TextPart(text=f"Caption: {message.caption}"))
                    logger.debug(
                        f"[SINGLE_CONVERSION] Added caption: {message.caption}"
                    )

            except Exception as e:
                logger.error(f"[SINGLE_CONVERSION] Failed to download media: {e}")
                parts.append(TextPart(text="[Media file - failed to download]"))

        # Create user message
        user_message = UserMessage.create_named(parts=parts, name=message.push_name)
        logger.debug(
            f"[SINGLE_CONVERSION] Created user message with name: {message.push_name}"
        )

        # Simply return the user message - Agent will handle conversation history via chat_id
        return user_message

    async def _process_with_agent(
        self, agent_input: AgentInput, session: WhatsAppSession
    ) -> GeneratedAssistantMessage[Any]:
        """Process input with agent using phone number as chat_id for conversation persistence."""
        logger.info("[AGENT_PROCESSING] Starting agent processing")

        try:
            async with self.agent.start_mcp_servers_async():
                logger.debug("[AGENT_PROCESSING] Started MCP servers")

                # Run agent with phone number as chat_id for conversation persistence
                result = await self.agent.run_async(
                    agent_input,
                    chat_id=session.phone_number,  # Use phone number as conversation ID
                )
                logger.info("[AGENT_PROCESSING] Agent run completed successfully")

            if result.generation:
                generated_message = result.generation.message
                logger.info(
                    f"[AGENT_PROCESSING] Generated response (length: {len(generated_message.text)})"
                )
                return generated_message

            logger.warning("[AGENT_PROCESSING] No generation found in result")
            # Return an empty GeneratedAssistantMessage when no generation is found
            from agentle.generations.models.message_parts.text import TextPart

            return GeneratedAssistantMessage[Any](
                parts=[TextPart(text="I processed your message but have no response.")],
                parsed=None,
            )

        except Exception as e:
            logger.error(
                f"[AGENT_PROCESSING_ERROR] Agent processing error: {e}", exc_info=True
            )
            raise

    async def _send_response(
        self,
        to: str,
        response: GeneratedAssistantMessage[Any] | str,
        reply_to: str | None = None,
    ) -> None:
        """Send response message(s) to user with quote support and line break splitting."""
        # Extract text from GeneratedAssistantMessage if needed
        response_text = (
            response.text
            if isinstance(response, GeneratedAssistantMessage)
            else response
        )

        logger.info(
            f"[SEND_RESPONSE] Sending response to {to} (length: {len(response_text)}, reply_to: {reply_to})"
        )

        # Split messages by line breaks and length
        messages = self._split_message_by_line_breaks(response_text)
        logger.debug(f"[SEND_RESPONSE] Split response into {len(messages)} parts")

        for i, msg in enumerate(messages):
            logger.debug(
                f"[SEND_RESPONSE] Sending message part {i + 1}/{len(messages)} to {to}"
            )

            # Show typing indicator before each message if configured
            if self.config.typing_indicator:
                logger.debug(
                    f"[SEND_RESPONSE] Sending typing indicator to {to} for message {i + 1}"
                )
                await self.provider.send_typing_indicator(
                    to, self.config.typing_duration
                )

            # Only quote the first message if quote_messages is enabled
            quoted_id = reply_to if i == 0 else None

            try:
                sent_message = await self.provider.send_text_message(
                    to=to, text=msg, quoted_message_id=quoted_id
                )
                logger.debug(
                    f"[SEND_RESPONSE] Successfully sent message part {i + 1} to {to}: {sent_message.id}"
                )
            except Exception as e:
                logger.error(
                    f"[SEND_RESPONSE_ERROR] Failed to send message part {i + 1} to {to}: {e}"
                )
                raise

            # Delay between messages (respecting typing duration + small buffer)
            if i < len(messages) - 1:
                # Use typing duration if typing indicator is enabled, otherwise use a small delay
                delay = (
                    self.config.typing_duration + 0.5
                    if self.config.typing_indicator
                    else 1.0
                )
                logger.debug(
                    f"[SEND_RESPONSE] Waiting {delay}s before sending next message part"
                )
                await asyncio.sleep(delay)

        logger.info(
            f"[SEND_RESPONSE] Successfully sent all {len(messages)} message parts to {to}"
        )

    def _split_message_by_line_breaks(self, text: str) -> Sequence[str]:
        """Split message by line breaks first, then by length if needed."""
        if not text.strip():
            return [""]

        # First split by double line breaks (paragraphs)
        paragraphs = text.split("\n\n")
        messages: MutableSequence[str] = []

        for paragraph in paragraphs:
            # Then split each paragraph by single line breaks
            lines = paragraph.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this line fits within message length limits
                if len(line) <= self.config.max_message_length:
                    messages.append(line)
                else:
                    # Split long lines by length
                    split_lines = self._split_long_line(line)
                    messages.extend(split_lines)

        # Filter out empty messages
        final_messages = [msg for msg in messages if msg.strip()]

        # If no messages were created, return a placeholder
        if not final_messages:
            final_messages = [""]

        logger.debug(
            f"[SPLIT_MESSAGE] Split '{text[:100]}...' into {len(final_messages)} messages"
        )
        return final_messages

    def _split_long_line(self, line: str) -> Sequence[str]:
        """Split a single long line into chunks that fit within message length limits."""
        if len(line) <= self.config.max_message_length:
            return [line]

        chunks: MutableSequence[str] = []

        # Try to split by sentences first (by periods, exclamation marks, question marks)
        sentence_endings = [". ", "! ", "? "]
        sentences: MutableSequence[str] = []
        current_sentence = ""

        i = 0
        while i < len(line):
            current_sentence += line[i]

            # Check if we hit a sentence ending
            for ending in sentence_endings:
                if line[i : i + len(ending)] == ending:
                    sentences.append(current_sentence)
                    current_sentence = ""
                    i += len(ending) - 1
                    break

            i += 1

        # Add remaining text as last sentence
        if current_sentence:
            sentences.append(current_sentence)

        # If we couldn't split by sentences effectively, fall back to word splitting
        if len(sentences) <= 1:
            sentences = line.split(" ")

        # Group sentences/words into chunks that fit
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            test_chunk = current_chunk + (" " if current_chunk else "") + sentence

            if len(test_chunk) <= self.config.max_message_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # Single sentence/word is too long, hard split it
                    for i in range(0, len(sentence), self.config.max_message_length):
                        chunk = sentence[i : i + self.config.max_message_length]
                        chunks.append(chunk)
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def _send_error_message(self, to: str, reply_to: str | None = None) -> None:
        """Send error message to user."""
        logger.warning(f"[SEND_ERROR] Sending error message to {to}")
        try:
            # Only quote if quote_messages is enabled
            quoted_id = reply_to if self.config.quote_messages else None
            await self.provider.send_text_message(
                to=to, text=self.config.error_message, quoted_message_id=quoted_id
            )
            logger.debug(f"[SEND_ERROR] Successfully sent error message to {to}")
        except Exception as e:
            logger.error(
                f"[SEND_ERROR_ERROR] Failed to send error message to {to}: {e}"
            )

    def _is_user_facing_error(self, error: Exception) -> bool:
        """Determine if an error should be communicated to the user."""
        # Don't show technical errors to users
        technical_errors = [
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            ImportError,
            ConnectionError,
        ]

        # Show only user-relevant errors like rate limiting
        user_relevant_errors = [
            "rate limit",
            "quota exceeded",
            "service unavailable",
        ]

        error_str = str(error).lower()

        # Don't show technical errors
        if any(isinstance(error, err_type) for err_type in technical_errors):
            return False

        # Show user-relevant errors
        if any(keyword in error_str for keyword in user_relevant_errors):
            return True

        # Default to not showing the error to users
        return False

    async def _send_rate_limit_message(self, to: str) -> None:
        """Send rate limit notification to user."""
        message = "You're sending messages too quickly. Please wait a moment before sending more messages."
        logger.info(f"[RATE_LIMIT] Sending rate limit message to {to}")
        try:
            await self.provider.send_text_message(to=to, text=message)
            logger.debug(f"[RATE_LIMIT] Successfully sent rate limit message to {to}")
        except Exception as e:
            logger.error(
                f"[RATE_LIMIT_ERROR] Failed to send rate limit message to {to}: {e}"
            )

    def _split_message(self, text: str) -> Sequence[str]:
        """Split long message into chunks."""
        if len(text) <= self.config.max_message_length:
            return [text]

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        messages: MutableSequence[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.config.max_message_length:
                if current:
                    current += "\n\n"
                current += para
            else:
                if current:
                    messages.append(current)
                current = para

        if current:
            messages.append(current)

        # Further split if any message is still too long
        final_messages = []
        for msg in messages:
            if len(msg) <= self.config.max_message_length:
                final_messages.append(msg)
            else:
                # Hard split
                for i in range(0, len(msg), self.config.max_message_length):
                    final_messages.append(msg[i : i + self.config.max_message_length])

        return final_messages

    async def _handle_message_upsert(
        self, payload: WhatsAppWebhookPayload
    ) -> GeneratedAssistantMessage[Any] | None:
        """Handle new message event."""
        logger.info("[MESSAGE_UPSERT] ═══════════ MESSAGE UPSERT START ═══════════")
        logger.debug("[MESSAGE_UPSERT] Processing message upsert event")
        logger.info(
            f"[MESSAGE_UPSERT] Current response callbacks count: {len(self._response_callbacks)}"
        )

        # Ensure bot is running before processing messages
        if not self._running:
            logger.warning(
                "[MESSAGE_UPSERT] ⚠️ Bot is not running, skipping message processing"
            )
            return None

        # Check if this is Evolution API format
        if payload.event == "messages.upsert" and payload.data:
            logger.info("[MESSAGE_UPSERT] Processing Evolution API format")
            # Evolution API format - single message in data field
            data = payload.data

            # Skip outgoing messages
            if data["key"].get("fromMe", False):
                logger.debug("[MESSAGE_UPSERT] Skipping outgoing message")
                return None

            logger.info("[MESSAGE_UPSERT] Parsing message from Evolution API data")
            # Parse message directly from data (which contains the message info)
            message = self._parse_evolution_message_from_data(data)

            if message:
                logger.info(
                    f"[MESSAGE_UPSERT] ✅ Parsed message: {message.id} from {message.from_number}"
                )
                logger.info(
                    f"[MESSAGE_UPSERT] About to call handle_message with {len(self._response_callbacks)} callbacks"
                )

                result = await self.handle_message(message)

                logger.info(
                    f"[MESSAGE_UPSERT] ✅ handle_message completed. Result: {result is not None}"
                )
                return result
            else:
                logger.warning(
                    "[MESSAGE_UPSERT] ❌ Failed to parse message from Evolution API data"
                )
                return None

        # Check if this is Meta API format
        elif payload.entry:
            # Meta API format - handle through provider
            logger.debug("[MESSAGE_UPSERT] Processing Meta API message upsert")
            await self.provider.validate_webhook(payload)
            return None
        else:
            logger.warning(
                "[MESSAGE_UPSERT] ⚠️ Unknown webhook format in message upsert"
            )
            return None

    async def _handle_message_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle message update event (status changes)."""
        if payload.event == "messages.update" and payload.data:
            logger.debug(f"[MESSAGE_UPDATE] Message update: {payload.data}")
        elif payload.entry:
            logger.debug(f"[MESSAGE_UPDATE] Message update: {payload.entry}")
        else:
            logger.debug(f"[MESSAGE_UPDATE] Message update: {payload}")

    async def _handle_connection_update(self, payload: WhatsAppWebhookPayload) -> None:
        """Handle connection status update."""
        if payload.event == "connection.update" and payload.data:
            logger.info(
                f"[CONNECTION_UPDATE] WhatsApp connection update: {payload.data}"
            )
        elif payload.entry:
            logger.info(
                f"[CONNECTION_UPDATE] WhatsApp connection update: {payload.entry}"
            )
        else:
            logger.info(f"[CONNECTION_UPDATE] WhatsApp connection update: {payload}")

    def _parse_evolution_message_from_data(self, data: Data) -> WhatsAppMessage | None:
        """Parse Evolution API message from webhook data field."""
        logger.debug("[PARSE_EVOLUTION] Parsing Evolution message from data")

        try:
            # Extract key information
            key = data["key"]
            message_id = key.get("id")
            from_number = key.get("remoteJid")

            if not message_id or not from_number:
                logger.warning("[PARSE_EVOLUTION] Missing message ID or from_number")
                return None

            logger.debug(
                f"[PARSE_EVOLUTION] Message ID: {message_id}, From: {from_number}"
            )

            # Get message type from the data
            message_type = data.get("messageType", "")
            logger.info(f"[PARSE_EVOLUTION] Message type: {message_type}")

            # Handle different message types
            if message_type == "editedMessage":
                logger.info(
                    "[PARSE_EVOLUTION] Handling editedMessage - treating as text message"
                )
                # For edited messages, we might not have the content, but we should still process it
                return WhatsAppTextMessage(
                    id=message_id,
                    push_name=data.get("pushName", "Unknown"),
                    from_number=from_number,
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=datetime.fromtimestamp(
                        data.get("messageTimestamp", 0)
                        / 1000  # Convert from milliseconds
                    ),
                    text="[Message was edited]",  # Placeholder text for edited messages
                )

            # Check if there's a message field
            if data.get("message"):
                msg_content = cast(Message, data.get("message"))

                # Handle text messages
                if msg_content.get("conversation"):
                    text = msg_content.get("conversation")
                    logger.debug(
                        f"[PARSE_EVOLUTION] Found conversation text: {text[:50] if text else 'None'}..."
                    )

                    return WhatsAppTextMessage(
                        id=message_id,
                        push_name=data.get("pushName", "Unknown"),
                        from_number=from_number,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            data.get("messageTimestamp", 0)
                            / 1000  # Convert from milliseconds
                        ),
                        text=text or ".",
                    )

                # Handle extended text messages
                elif msg_content.get("extendedTextMessage"):
                    extended_text_message = msg_content.get("extendedTextMessage")
                    text = (
                        extended_text_message.get("text", "")
                        if extended_text_message
                        else ""
                    )
                    logger.debug(
                        f"[PARSE_EVOLUTION] Found extended text: {text[:50] if text else 'None'}..."
                    )

                    return WhatsAppTextMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.get("pushName", "Unknown"),
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            data.get("messageTimestamp", 0) / 1000
                        ),
                        text=text,
                    )

                # Handle image messages
                elif msg_content.get("imageMessage"):
                    logger.debug("[PARSE_EVOLUTION] Found image message")
                    image_msg = msg_content.get("imageMessage")
                    return WhatsAppImageMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.get("pushName", "Unknown"),
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            data.get("messageTimestamp", 0) / 1000
                        ),
                        media_url=image_msg.get("url", "") if image_msg else "",
                        media_mime_type=image_msg.get("mimetype", "image/jpeg")
                        if image_msg
                        else "image/jpeg",
                        caption=image_msg.get("caption") if image_msg else "",
                    )

                # Handle document messages
                elif msg_content.get("documentMessage"):
                    logger.debug("[PARSE_EVOLUTION] Found document message")
                    doc_msg = msg_content.get("documentMessage")
                    return WhatsAppDocumentMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.get("pushName", "Unknown"),
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            data.get("messageTimestamp", 0) / 1000
                        ),
                        media_url=doc_msg.get("url", "") if doc_msg else "",
                        media_mime_type=doc_msg.get(
                            "mimetype", "application/octet-stream"
                        )
                        if doc_msg
                        else "application/octet-stream",
                        filename=doc_msg.get("fileName") if doc_msg else "",
                        caption=doc_msg.get("caption") if doc_msg else "",
                    )

                # Handle audio messages
                elif msg_content.get("audioMessage"):
                    logger.debug("[PARSE_EVOLUTION] Found audio message")
                    audio_msg = msg_content.get("audioMessage")
                    return WhatsAppAudioMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.get("pushName", "Unknown"),
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            data.get("messageTimestamp", 0) / 1000
                        ),
                        media_url=audio_msg.get("url", "") if audio_msg else "",
                        media_mime_type=audio_msg.get("mimetype", "audio/ogg")
                        if audio_msg
                        else "audio/ogg",
                    )
                elif msg_content.get("videoMessage"):
                    logger.debug("[PARSE_EVOLUTION] Found video message")
                    video_msg = msg_content.get("videoMessage")
                    return WhatsAppVideoMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.get("pushName", "Unknown"),
                        caption=video_msg.get("caption") if video_msg else None,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            data.get("messageTimestamp", 0) / 1000
                        ),
                        media_url=video_msg.get("url", "") if video_msg else "",
                        media_mime_type=video_msg.get("mimetype", "")
                        if video_msg
                        else "",
                    )
                else:
                    logger.warning(
                        f"[PARSE_EVOLUTION] Unknown message type in content: {list(msg_content.keys())}"
                    )

            # If we get here and message is empty but we have messageType info, create a placeholder
            elif message_type and message_type != "":
                logger.info(
                    f"[PARSE_EVOLUTION] Empty message content but messageType is '{message_type}' - creating placeholder"
                )
                return WhatsAppTextMessage(
                    id=message_id,
                    push_name=data.get("pushName", "Unknown"),
                    from_number=from_number,
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=datetime.fromtimestamp(
                        data.get("messageTimestamp", 0) / 1000
                    ),
                    text=f"[{message_type}]",  # Placeholder for message types we can't parse
                )

            logger.warning("[PARSE_EVOLUTION] No recognizable message content found")
            return None

        except Exception as e:
            logger.error(
                f"[PARSE_EVOLUTION_ERROR] Error parsing Evolution message from data: {e}",
                exc_info=True,
            )

        return None

    async def _handle_meta_webhook(
        self, payload: WhatsAppWebhookPayload
    ) -> GeneratedAssistantMessage[Any] | None:
        """Handle Meta WhatsApp Business API webhooks."""
        logger.debug("[META_WEBHOOK] Processing Meta webhook")

        try:
            if not payload.entry:
                logger.warning("[META_WEBHOOK] No entry data in Meta webhook")
                return None

            response = None

            for entry_item in payload.entry:
                changes = entry_item.get("changes", [])
                for change in changes:
                    field = change.get("field")
                    value = change.get("value", {})

                    if field == "messages":
                        logger.debug("[META_WEBHOOK] Processing messages field")
                        # Process incoming messages
                        messages = value.get("messages", [])
                        for msg_data in messages:
                            # Skip outgoing messages
                            if (
                                msg_data.get("from")
                                == self.provider.get_instance_identifier()
                            ):
                                logger.debug("[META_WEBHOOK] Skipping outgoing message")
                                continue

                            message = await self._parse_meta_message(msg_data)
                            if message:
                                logger.info(
                                    f"[META_WEBHOOK] Parsed message: {message.id} from {message.from_number}"
                                )
                                # Return the response from the last processed message
                                response = await self.handle_message(message)

            return response

        except Exception as e:
            logger.error(
                f"[META_WEBHOOK_ERROR] Error handling Meta webhook: {e}", exc_info=True
            )
            return None

    async def _parse_meta_message(
        self, msg_data: dict[str, Any]
    ) -> WhatsAppMessage | None:
        """Parse Meta API message format."""
        logger.debug("[PARSE_META] Parsing Meta API message")

        try:
            message_id = msg_data.get("id")
            from_number = msg_data.get("from")
            timestamp_str = msg_data.get("timestamp")

            if not message_id or not from_number:
                logger.warning("[PARSE_META] Missing message ID or from_number")
                return None

            logger.debug(f"[PARSE_META] Message ID: {message_id}, From: {from_number}")

            # Convert timestamp
            timestamp = (
                datetime.fromtimestamp(int(timestamp_str))
                if timestamp_str
                else datetime.now()
            )

            # Handle different message types
            msg_type = msg_data.get("type")
            logger.debug(f"[PARSE_META] Message type: {msg_type}")

            if msg_type == "text":
                text_data = msg_data.get("text", {})
                text = text_data.get("body", "")

                return WhatsAppTextMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    text=text,
                )

            elif msg_type == "image":
                image_data = msg_data.get("image", {})

                return WhatsAppImageMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=image_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=image_data.get("mime_type", "image/jpeg"),
                    caption=image_data.get("caption"),
                )

            elif msg_type == "document":
                doc_data = msg_data.get("document", {})

                return WhatsAppDocumentMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=doc_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=doc_data.get(
                        "mime_type", "application/octet-stream"
                    ),
                    filename=doc_data.get("filename"),
                    caption=doc_data.get("caption"),
                )

            elif msg_type == "audio":
                audio_data = msg_data.get("audio", {})

                return WhatsAppAudioMessage(
                    id=message_id,
                    from_number=from_number,
                    push_name=msg_data.get("pushName", "user"),
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=timestamp,
                    media_url=audio_data.get("id", ""),  # Meta uses ID for media
                    media_mime_type=audio_data.get("mime_type", "audio/ogg"),
                )

        except Exception as e:
            logger.error(
                f"[PARSE_META_ERROR] Error parsing Meta message: {e}", exc_info=True
            )

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the bot's current state."""
        return {
            "running": self._running,
            "active_batch_processors": len(self._batch_processors),
            "processing_locks": len(self._processing_locks),
            "agent_has_conversation_store": self.agent.conversation_store is not None,
            "config": {
                "message_batching_enabled": self.config.enable_message_batching,
                "spam_protection_enabled": self.config.spam_protection_enabled,
                "quote_messages": self.config.quote_messages,
                "batch_delay_seconds": self.config.batch_delay_seconds,
                "max_batch_size": self.config.max_batch_size,
                "max_messages_per_minute": self.config.max_messages_per_minute,
                "debug_mode": self.config.debug_mode,
            },
        }
