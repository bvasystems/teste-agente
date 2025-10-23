from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import (
    Awaitable,
    Callable,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, cast
from dataclasses import dataclass, field

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.agents.agent import Agent
from agentle.agents.agent_input import AgentInput
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.agents.whatsapp.models.data import Data
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

# Type aliases for cleaner type hints
PhoneNumber = str  # WhatsApp phone number (e.g., "5511999999999")
ChatId = str  # Chat/conversation identifier

CallbackFunction = (
    Callable[
        [
            PhoneNumber,
            ChatId | None,
            GeneratedAssistantMessage[Any] | None,
            dict[str, Any] | None,
        ],
        None,
    ]
    | Callable[
        [
            PhoneNumber,
            ChatId | None,
            GeneratedAssistantMessage[Any] | None,
            dict[str, Any] | None,
        ],
        Awaitable[None],
    ]
)

CallbackInput = CallbackFunction | list[CallbackFunction] | None

logger = logging.getLogger(__name__)


@dataclass
class CallbackWithContext:
    """Container for callback function with optional context."""

    # Callbacks must accept (phone_number, chat_id, response, context) now
    callback: (
        Callable[
            [
                PhoneNumber,
                ChatId | None,
                GeneratedAssistantMessage[Any] | None,
                dict[str, Any],
            ],
            Awaitable[None],
        ]
        | Callable[
            [
                PhoneNumber,
                ChatId | None,
                GeneratedAssistantMessage[Any] | None,
                dict[str, Any],
            ],
            None,
        ]
    )

    context: dict[str, Any] = field(default_factory=dict)


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

    async def handle_message(
        self, message: WhatsAppMessage, chat_id: ChatId | None = None
    ) -> GeneratedAssistantMessage[Any] | None:
        """
        Handle incoming WhatsApp message with enhanced error handling and batching.
        """
        logger.info("[MESSAGE_HANDLER] ═══════════ MESSAGE HANDLER ENTRY ═══════════")
        logger.info(
            f"[MESSAGE_HANDLER] Received message from {message.from_number}: ID={message.id}, Type={type(message).__name__}"
        )
        # ADICIONE ESTE LOG:
        logger.info(f"[MESSAGE_HANDLER] Chat ID recebido: {chat_id}")
        logger.info(
            f"[MESSAGE_HANDLER] Current response callbacks count: {len(self._response_callbacks)}"
        )

        # Log callback details
        for i, cb in enumerate(self._response_callbacks):
            logger.info(
                f"[MESSAGE_HANDLER] Callback {i + 1}: {cb.callback.__name__ if hasattr(cb.callback, '__name__') else 'unnamed'}, Context: {cb.context}"
            )

        try:
            # Get or create session FIRST to check rate limiting
            logger.debug(f"[MESSAGE_HANDLER] Getting session for {message.from_number}")
            session = await self.provider.get_session(message.from_number)
            if not session:
                logger.error(
                    f"[MESSAGE_HANDLER] ❌ Failed to get session for {message.from_number}"
                )
                return

            # CRITICAL FIX: Check rate limiting BEFORE any message processing
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
                        f"[SPAM_PROTECTION] ❌ Rate limited user {message.from_number} - BLOCKING message processing"
                    )
                    if session.is_rate_limited:
                        await self._send_rate_limit_message(message.from_number)
                    # CRITICAL: Update session to persist rate limiting state and return immediately
                    await self.provider.update_session(session)
                    return None

            # Mark as read if configured (only after rate limiting check passes)
            if self.config.auto_read_messages:
                logger.debug(f"[MESSAGE_HANDLER] Marking message {message.id} as read")
                await self.provider.mark_message_as_read(message.id)

            # Store custom chat_id in session
            if chat_id:
                session.context_data["custom_chat_id"] = chat_id
                logger.info(
                    f"[MESSAGE_HANDLER] Stored custom chat_id in session: {chat_id}"
                )

            # CRITICAL FIX: Store remoteJid for @lid numbers
            if message.remote_jid:
                session.context_data["remote_jid"] = message.remote_jid
                logger.info(
                    f"[MESSAGE_HANDLER] 🔑 Stored remote_jid in session: {message.remote_jid} for phone: {message.from_number}"
                )

            logger.info(
                f"[SESSION_STATE] Session for {message.from_number}: is_processing={session.is_processing}, pending_messages={len(session.pending_messages)}, message_count={session.message_count}"
            )

            effective_chat_id = chat_id or message.from_number
            logger.info(
                f"[MESSAGE_HANDLER] Effective chat_id para conversação: {effective_chat_id}"
            )

            # Check welcome message for first interaction
            if (
                await cast(
                    ConversationStore, self.agent.conversation_store
                ).get_conversation_history_length(effective_chat_id)
                == 0
                and self.config.welcome_message
            ):
                logger.info(
                    f"[WELCOME] Sending welcome message to {message.from_number}"
                )
                formatted_welcome = self._format_whatsapp_markdown(
                    self.config.welcome_message
                )
                await self.provider.send_text_message(
                    message.from_number, formatted_welcome
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
                response = await self._handle_message_with_batching(
                    message, session, chat_id=effective_chat_id
                )
            else:
                logger.info(
                    f"[IMMEDIATE] ⚡ Processing message immediately for {message.from_number}"
                )
                response = await self._process_single_message(
                    message, session, chat_id=effective_chat_id
                )

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

    async def handle_webhook(
        self,
        payload: WhatsAppWebhookPayload,
        callback: CallbackInput = None,
        callback_context: dict[str, Any] | None = None,
        chat_id: ChatId | None = None,
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

        # FOR BATCHING: Register callback(s) permanently, don't remove them in finally block
        # The batch processor will call them when the batch is processed
        if callback:
            logger.info("[WEBHOOK] Processing callback registration for batching...")

            # Handle both single callback and list of callbacks
            callbacks_to_register: list[CallbackFunction]
            if isinstance(callback, list):
                callbacks_to_register = callback
            else:
                callbacks_to_register = [callback]

            logger.info(
                f"[WEBHOOK] Registering {len(callbacks_to_register)} callback(s)"
            )

            for i, cb in enumerate(callbacks_to_register):
                callback_with_context = CallbackWithContext(
                    callback=cb, context=callback_context or {}
                )

                # Check if this exact callback is already registered to avoid duplicates
                callback_exists = any(
                    existing.callback == cb and existing.context == callback_context
                    for existing in self._response_callbacks
                )

                logger.info(
                    f"[WEBHOOK] Callback {i + 1} already exists: {callback_exists}"
                )

                if not callback_exists:
                    self._response_callbacks.append(callback_with_context)
                    logger.info(
                        f"[WEBHOOK] ✅ Added callback {i + 1} for batching. Total callbacks: {len(self._response_callbacks)}"
                    )
                else:
                    logger.warning(f"[WEBHOOK] ⚠️ Duplicate callback {i + 1} not added")

            logger.info(
                "[WEBHOOK] ⚠️  IMPORTANT: Callback(s) will be called when batch is processed, not removed immediately"
            )

        try:
            logger.info("[WEBHOOK] Starting webhook validation...")
            await self.provider.validate_webhook(payload)
            logger.info("[WEBHOOK] ✅ Webhook validation passed")

            response = None

            # Handle Evolution API events
            if payload.event == "messages.upsert":
                logger.info("[WEBHOOK] 🔄 Handling messages.upsert event")
                response = await self._handle_message_upsert(payload, chat_id=chat_id)
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
        callback: (
            Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                Awaitable[None],
            ]
            | Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                None,
            ]
        ),
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

        callback_with_context = CallbackWithContext(
            callback=callback, context=context or {}
        )

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
        callback: (
            Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                Awaitable[None],
            ]
            | Callable[
                [
                    PhoneNumber,
                    ChatId | None,
                    GeneratedAssistantMessage[Any] | None,
                    dict[str, Any],
                ],
                None,
            ]
        ),
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

    async def _cleanup_abandoned_processors(self) -> None:
        """Clean up batch processors that have been running too long, but protect active message sending."""
        abandoned_processors: MutableSequence[PhoneNumber] = []

        for phone_number, task in self._batch_processors.items():
            if task.done():
                abandoned_processors.append(phone_number)
                continue

            # Check if session is still processing
            session = await self.provider.get_session(phone_number)
            if not session:
                # No session found, abandon this processor
                abandoned_processors.append(phone_number)
                task.cancel()
                continue

            # CRITICAL: Don't abandon if currently sending messages
            is_sending_messages = session.context_data.get("is_sending_messages", False)

            if is_sending_messages:
                logger.info(
                    f"[CLEANUP] Protecting batch processor for {phone_number} - currently sending messages"
                )
                continue

            # Check if batch has been running too long
            if session.is_batch_expired(
                self.config.max_batch_timeout_seconds * 3
            ):  # Give more time
                logger.warning(
                    f"[CLEANUP] Found abandoned batch processor for {phone_number} (not sending messages)"
                )
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

    async def _handle_message_with_batching(
        self,
        message: WhatsAppMessage,
        session: WhatsAppSession,
        chat_id: ChatId | None = None,
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

        if chat_id:
            session.context_data["custom_chat_id"] = chat_id
            logger.info(f"[BATCHING] ✅ Stored custom_chat_id in session: {chat_id}")
        else:
            logger.warning("[BATCHING] ⚠️ No chat_id provided to store in session")

        try:
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

                # CRITICAL FIX: Preserve custom_chat_id from original session
                original_chat_id = session.context_data.get("custom_chat_id")
                if original_chat_id and not current_session.context_data.get(
                    "custom_chat_id"
                ):
                    logger.warning(
                        f"[BATCHING] ⚠️ custom_chat_id lost during session re-fetch, restoring: {original_chat_id}"
                    )
                    current_session.context_data["custom_chat_id"] = original_chat_id
                    logger.info(
                        f"[BATCHING] ✅ Restored custom_chat_id: {original_chat_id}"
                    )
                elif original_chat_id:
                    logger.info(
                        f"[BATCHING] ✅ custom_chat_id preserved during re-fetch: {original_chat_id}"
                    )
                else:
                    logger.info("[BATCHING] No custom_chat_id to preserve")

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

                # CRITICAL FIX: Ensure custom_chat_id is preserved after update
                expected_chat_id = current_session.context_data.get("custom_chat_id")
                actual_chat_id = updated_session.context_data.get("custom_chat_id")
                if expected_chat_id and not actual_chat_id:
                    logger.error(
                        f"[BATCHING] ❌ custom_chat_id lost after session update! Expected: {expected_chat_id}, Got: {actual_chat_id}"
                    )
                    updated_session.context_data["custom_chat_id"] = expected_chat_id
                    await self.provider.update_session(updated_session)
                    logger.info(
                        f"[BATCHING] ✅ Restored custom_chat_id after update: {expected_chat_id}"
                    )
                elif expected_chat_id:
                    logger.info(
                        f"[BATCHING] ✅ custom_chat_id preserved after update: {expected_chat_id}"
                    )

                logger.info(
                    f"[BATCHING] Updated session state: processing={updated_session.is_processing}, token={updated_session.processing_token}, pending={len(updated_session.pending_messages)}"
                )

                # CRITICAL FIX: Enhanced race condition protection for batch processor creation
                should_create_processor = (
                    updated_session.is_processing
                    and updated_session.processing_token
                    and phone_number not in self._batch_processors
                )

                # Double-check to prevent race conditions
                if should_create_processor:
                    # Check again inside the lock to prevent duplicate processors
                    if phone_number in self._batch_processors:
                        existing_task = self._batch_processors[phone_number]
                        if not existing_task.done():
                            logger.info(
                                f"[BATCHING] ⚠️ Processor already exists for {phone_number}, skipping creation"
                            )
                            should_create_processor = False
                        else:
                            logger.info(
                                f"[BATCHING] 🧹 Cleaning up completed processor for {phone_number}"
                            )
                            del self._batch_processors[phone_number]

                if should_create_processor:
                    logger.info(
                        f"[BATCHING] 🚀 Starting new batch processor for {phone_number}"
                    )
                    logger.info(
                        f"[BATCHING] Processing token: {updated_session.processing_token}"
                    )
                    # Ensure processing_token is not None before passing to _batch_processor
                    if updated_session.processing_token:
                        self._batch_processors[phone_number] = asyncio.create_task(
                            self._batch_processor(
                                phone_number, updated_session.processing_token
                            )
                        )
                        logger.info(
                            f"[BATCHING] ✅ Batch processor task created for {phone_number}"
                        )
                    else:
                        logger.error(
                            f"[BATCHING] ❌ Cannot create processor: processing_token is None for {phone_number}"
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
        self,
        phone_number: PhoneNumber,
        session: WhatsAppSession,
        message_data: dict[str, Any],
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

            # Log context_data before persisting
            logger.info(
                f"[ATOMIC_UPDATE] Context data before update: {session.context_data}"
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

            # Log context_data after persisting
            logger.info(
                f"[ATOMIC_UPDATE] Context data after update: {verification_session.context_data}"
            )

            # Verify context_data is preserved
            if verification_session.context_data.get(
                "custom_chat_id"
            ) != session.context_data.get("custom_chat_id"):
                logger.error(
                    f"[ATOMIC_UPDATE] ❌ custom_chat_id not preserved! Before: {session.context_data.get('custom_chat_id')}, After: {verification_session.context_data.get('custom_chat_id')}"
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

    async def _batch_processor(
        self, phone_number: PhoneNumber, processing_token: str
    ) -> None:
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
        self, phone_number: PhoneNumber, session: WhatsAppSession, processing_token: str
    ) -> GeneratedAssistantMessage[Any] | None:
        """Process a batch of messages for a user with enhanced timeout protection."""
        logger.info("[BATCH_PROCESSING] ═══════════ BATCH PROCESSING START ═══════════")
        logger.info(
            f"[BATCH_PROCESSING] Phone: {phone_number}, Token: {processing_token}"
        )

        chat_id = session.context_data.get("custom_chat_id")
        logger.info(f"[BATCH_PROCESSING] Retrieved chat_id from session: {chat_id}")
        logger.info(
            f"[BATCH_PROCESSING] Session context_data keys: {list(session.context_data.keys())}"
        )

        # DEBUG: Log all context data for troubleshooting
        if chat_id is None:
            logger.warning(
                f"[BATCH_PROCESSING] ⚠️ custom_chat_id is None! Full context_data: {session.context_data}"
            )
        else:
            logger.info(f"[BATCH_PROCESSING] ✅ Using custom chat_id: {chat_id}")

        if not session.pending_messages:
            logger.warning(
                f"[BATCH_PROCESSING] ⚠️ No pending messages for {phone_number}, finishing batch processing"
            )
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)
            return None

        try:
            # IMPORTANT: Mark session as "sending messages" to prevent cleanup during sending
            session.context_data["is_sending_messages"] = True
            session.context_data["sending_started_at"] = datetime.now().isoformat()
            await self.provider.update_session(session)

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

            # Check if batch conversion returned None (empty batch)
            if not agent_input:
                logger.warning(
                    f"[BATCH_PROCESSING] Batch conversion returned None for {phone_number} - skipping empty batch"
                )
                # Clear sending state and finish batch processing
                session.context_data["is_sending_messages"] = False
                session.context_data["sending_completed_at"] = (
                    datetime.now().isoformat()
                )
                session.finish_batch_processing(processing_token)
                await self.provider.update_session(session)
                return None

            # Process with agent
            logger.info(f"[BATCH_PROCESSING] 🤖 Running agent for {phone_number}")
            response, input_tokens, output_tokens = await self._process_with_agent(
                agent_input, session, chat_id=chat_id
            )
            logger.info(
                f"[BATCH_PROCESSING] ✅ Agent processing complete for {phone_number}"
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

            # CRITICAL: Send response with enhanced error handling
            await self._send_response(phone_number, response, first_message_id)

            # Update session - clear sending state
            session.message_count += len(pending_messages)
            session.last_activity = datetime.now()
            session.context_data["is_sending_messages"] = False
            session.context_data["sending_completed_at"] = datetime.now().isoformat()

            # Finish batch processing with token validation
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)

            logger.info(
                f"[BATCH_PROCESSING] ✅ Successfully processed batch for {phone_number}. Total messages processed: {session.message_count}"
            )

            # Call response callbacks
            await self._call_response_callbacks(
                phone_number,
                response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                chat_id=chat_id,
            )
            return response

        except Exception as e:
            logger.error(
                f"[BATCH_PROCESSING_ERROR] ❌ Error processing message batch for {phone_number}: {e}",
                exc_info=True,
            )

            # Clear sending state on error
            try:
                session.context_data["is_sending_messages"] = False
                session.context_data["sending_error_at"] = datetime.now().isoformat()
                session.context_data["sending_error"] = str(e)
            except Exception:
                pass

            await self._send_error_message(phone_number)

            # Ensure session state is cleaned up even on error
            session.finish_batch_processing(processing_token)
            await self.provider.update_session(session)

            # Call response callbacks with None response on error
            await self._call_response_callbacks(
                phone_number,
                None,
                input_tokens=0,
                output_tokens=0,
                chat_id=chat_id,
            )
            raise

    async def _process_single_message(
        self,
        message: WhatsAppMessage,
        session: WhatsAppSession,
        chat_id: ChatId | None = None,
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
            response, input_tokens, output_tokens = await self._process_with_agent(
                agent_input, session, chat_id=chat_id
            )
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
            await self._call_response_callbacks(
                message.from_number,
                response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                chat_id=chat_id,
            )
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
            await self._call_response_callbacks(
                phone_number=message.from_number,
                response=None,
                input_tokens=0,
                output_tokens=0,
                chat_id=chat_id,
            )
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

        # If no parts were added, skip processing instead of creating placeholder
        if not parts:
            logger.warning(
                "[BATCH_CONVERSION] No parts were created - skipping batch processing to avoid empty message"
            )
            # Return None to indicate this batch should be skipped
            # This prevents the agent from receiving empty content
            return None

        # Create user message with first message's push name
        first_message = message_batch[0] if message_batch else {}
        push_name = first_message.get("push_name", "User")
        user_message = UserMessage.create_named(parts=parts, name=push_name)
        logger.debug(f"[BATCH_CONVERSION] Created user message with name: {push_name}")

        # Simply return the user message - Agent will handle conversation history via chat_id
        return user_message

    async def _call_response_callbacks(
        self,
        phone_number: PhoneNumber,
        response: GeneratedAssistantMessage[Any] | None,
        input_tokens: float,
        output_tokens: float,
        *,
        chat_id: ChatId | None = None,
    ) -> None:
        """Call all registered response callbacks with (phone_number, chat_id, response, context)."""
        logger.info("[CALLBACKS] ═══════════ CALLING RESPONSE CALLBACKS ═══════════")
        logger.info(f"[CALLBACKS] Phone number: {phone_number}")
        logger.info(f"[CALLBACKS] Response provided: {response is not None}")
        logger.info(f"[CALLBACKS] chat_id: {chat_id}")
        logger.info(
            f"[CALLBACKS] Total callbacks to call: {len(self._response_callbacks)}"
        )

        if response:
            logger.info(f"[CALLBACKS] Response text length: {len(response.text)}")
            logger.debug(f"[CALLBACKS] Response text preview: {response.text[:100]}...")

        if not self._response_callbacks:
            logger.warning("[CALLBACKS] ⚠️ No callbacks registered to call!")
            return

        for i, cb in enumerate(self._response_callbacks):
            logger.info(
                f"[CALLBACKS] 🔄 Calling callback {i + 1}/{len(self._response_callbacks)}"
            )
            logger.info(
                f"[CALLBACKS] Callback function: {getattr(cb.callback, '__name__', 'unnamed')}"
            )
            logger.info(
                f"[CALLBACKS] Callback context keys: {list(cb.context.keys()) if cb.context else 'None'}"
            )

            cb.context.update(
                {"input_tokens": input_tokens, "output_tokens": output_tokens}
            )

            try:
                if inspect.iscoroutinefunction(cb.callback):
                    await cb.callback(phone_number, chat_id, response, cb.context)
                else:
                    cb.callback(phone_number, chat_id, response, cb.context)
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
        self,
        agent_input: AgentInput,
        session: WhatsAppSession,
        chat_id: ChatId | None = None,
    ) -> tuple[GeneratedAssistantMessage[Any], int, int]:
        """Process input with agent using custom chat_id for conversation persistence."""
        logger.info("[AGENT_PROCESSING] Starting agent processing")

        # MUDANÇA CRÍTICA: Recuperar chat_id personalizado da sessão se não fornecido
        effective_chat_id = chat_id
        if not effective_chat_id:
            effective_chat_id = session.context_data.get("custom_chat_id")
        if not effective_chat_id:
            effective_chat_id = session.phone_number

        logger.info(f"[AGENT_PROCESSING] Using effective chat_id: {effective_chat_id}")
        logger.info(
            f"[AGENT_PROCESSING] Chat ID type: {'CUSTOM' if chat_id else 'FALLBACK'}"
        )

        try:
            async with self.agent.start_mcp_servers_async():
                logger.debug("[AGENT_PROCESSING] Started MCP servers")

                # Run agent with effective chat_id for conversation persistence
                result = await self.agent.run_async(
                    agent_input,
                    chat_id=effective_chat_id,
                )
                input_tokens = result.input_tokens
                output_tokens = result.output_tokens

                logger.debug(f"Input tokens: {input_tokens}")
                logger.debug(f"Output tokens: {output_tokens}")

                logger.info("[AGENT_PROCESSING] Agent run completed successfully")

            if result.generation:
                generated_message = result.generation.message
                logger.info(
                    f"[AGENT_PROCESSING] Generated response (length: {len(generated_message.text)})"
                )

                # FIXED: Always clean thinking tags from all parts
                cleaned_parts: list[TextPart | ToolExecutionSuggestion] = []

                for part in generated_message.parts:
                    if part.type == "text":
                        from agentle.generations.models.message_parts.text import (
                            TextPart,
                        )

                        part_text = str(part.text) if part.text else ""
                        cleaned_part_text = self._remove_thinking_tags(part_text)
                        cleaned_parts.append(TextPart(text=cleaned_part_text))
                    else:
                        # FIXED: Keep non-text parts
                        cleaned_parts.append(part)

                # Always return cleaned message
                return (
                    GeneratedAssistantMessage[Any](
                        parts=cleaned_parts,
                        parsed=generated_message.parsed,
                    ),
                    input_tokens,
                    output_tokens,
                )

            logger.warning("[AGENT_PROCESSING] No generation found in result")
            from agentle.generations.models.message_parts.text import TextPart

            return (
                GeneratedAssistantMessage[Any](
                    parts=[
                        TextPart(
                            text="Desculpe, não consegui processar sua mensagem no momento. Tente novamente."
                        )
                    ],
                    parsed=None,
                ),
                input_tokens,
                output_tokens,
            )

        except Exception as e:
            logger.error(
                f"[AGENT_PROCESSING_ERROR] Agent processing error: {e}", exc_info=True
            )
            raise

    def _remove_thinking_tags(self, text: str) -> str:
        """Remove thinking tags and their content from the response text.

        This method handles:
        - Multiple occurrences of thinking tags
        - Tags spanning multiple lines
        - Malformed or incomplete tags
        - Case-insensitive matching
        - Responses with no thinking tags

        Args:
            text: The original response text that may contain thinking tags

        Returns:
            The cleaned text with thinking tags and their content removed
        """
        if not text:
            return text

        original_text = text

        # Pattern 1: Complete thinking tags (case-insensitive, multiline)
        # Use re.DOTALL flag to make . match newlines as well
        text = re.sub(
            r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Pattern 2: Handle malformed tags or incomplete tags
        # Remove opening thinking tags without closing tags (to the end of text)
        text = re.sub(r"<thinking>.*?$", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Pattern 3: Remove any remaining orphaned closing tags
        text = re.sub(r"</thinking>", "", text, flags=re.IGNORECASE)

        # Pattern 4: Handle variations with attributes or whitespace
        text = re.sub(
            r"<thinking[^>]*>.*?</thinking[^>]*>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Clean up any extra whitespace that might be left after removing thinking tags
        # Replace multiple consecutive newlines with double newlines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Remove excessive spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Clean up leading/trailing whitespace
        text = text.strip()

        # Log if thinking tags were found and removed
        if original_text != text:
            thinking_tags_removed = len(
                re.findall(
                    r"<thinking[^>]*>.*?</thinking[^>]*>",
                    original_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
            )
            logger.warning(
                f"[THINKING_CLEANUP] Removed {thinking_tags_removed} thinking tag(s). "
                + f"Original length: {len(original_text)}, Cleaned length: {len(text)}"
            )

            # Additional debug info for persistent issues
            if self.config.debug_mode:
                logger.debug(
                    f"[THINKING_CLEANUP] Original text preview: {original_text[:200]}..."
                )
                logger.debug(
                    f"[THINKING_CLEANUP] Cleaned text preview: {text[:200]}..."
                )

        return text

    def _format_whatsapp_markdown(self, text: str) -> str:
        """Convert standard markdown to WhatsApp-compatible formatting.

        WhatsApp supports:
        - *bold* for bold text
        - _italic_ for italic text
        - ~strikethrough~ for strikethrough text
        - ```code``` for monospace text
        - No support for headers, tables, or complex markdown structures

        This method converts:
        - Headers (# ## ###) to bold text with separators
        - Tables to formatted text
        - Markdown lists to plain text lists
        - Links to "text (url)" format
        - Images to descriptive text
        - Blockquotes to indented text
        """
        if not text:
            return text

        # Split text into lines for processing
        lines = text.split("\n")
        processed_lines: list[str] = []
        in_code_block = False
        in_table = False
        table_lines: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Track code blocks (preserve them as-is)
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                processed_lines.append(line)
                i += 1
                continue

            # If we're in a code block, don't process the line
            if in_code_block:
                processed_lines.append(line)
                i += 1
                continue

            # Detect table start (header line followed by separator line)
            if i + 1 < len(lines) and self._is_table_separator(lines[i + 1]):
                in_table = True
                table_lines = [line]
                i += 1
                continue

            # Continue collecting table rows
            if in_table:
                if self._is_table_row(line):
                    table_lines.append(line)
                    i += 1
                    continue
                else:
                    # End of table, process it
                    processed_lines.extend(self._format_table(table_lines))
                    in_table = False
                    table_lines = []
                    # Don't increment i, process current line

            # Process headers
            if line.strip().startswith("#"):
                processed_lines.append(self._format_header(line))
                i += 1
                continue

            # Process blockquotes
            if line.strip().startswith(">"):
                processed_lines.append(self._format_blockquote(line))
                i += 1
                continue

            # Process horizontal rules
            if re.match(r"^[\s]*(-{3,}|\*{3,}|_{3,})[\s]*$", line):
                processed_lines.append("─" * 30)
                i += 1
                continue

            # Process regular line
            processed_lines.append(line)
            i += 1

        # If we ended while in a table, process it
        if table_lines:
            processed_lines.extend(self._format_table(table_lines))

        # Rejoin lines
        text = "\n".join(processed_lines)

        # Handle inline markdown elements

        # Handle images ![alt](url) -> [Image: alt]
        text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"[Imagem: \1]", text)

        # Handle links [text](url) -> text (url)
        text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 (\2)", text)

        # Handle **bold** -> *bold* (WhatsApp format)
        # Use negative lookbehind and lookahead to avoid matching within code blocks
        text = re.sub(r"(?<!`)\*\*([^*`]+)\*\*(?!`)", r"*\1*", text)

        # Handle __italic__ -> _italic_
        text = re.sub(r"(?<!`)__([^_`]+)__(?!`)", r"_\1_", text)

        # Handle ~~strikethrough~~ -> ~strikethrough~
        text = re.sub(r"(?<!`)~~([^~`]+)~~(?!`)", r"~\1~", text)

        # Handle single backtick `code` -> ```code``` (WhatsApp monospace)
        # But don't convert if already part of triple backticks
        text = re.sub(r"(?<!`)(`(?!``)([^`]+)`)(?!`)", r"```\2```", text)

        return text

    def _is_table_separator(self, line: str) -> bool:
        """Check if a line is a markdown table separator (e.g., |---|---|)."""
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            return False
        # Remove outer pipes and split
        content = stripped[1:-1]
        cells = content.split("|")
        # Check if all cells are just dashes, colons, and spaces
        for cell in cells:
            cell = cell.strip()
            if cell and not re.match(r"^:?-+:?$", cell):
                return False
        return True

    def _is_table_row(self, line: str) -> bool:
        """Check if a line is a table row."""
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    def _format_table(self, table_lines: list[str]) -> list[str]:
        """Convert a markdown table to WhatsApp-friendly vertical list format.

        Uses a consistent vertical format for all tables to ensure readability
        on all screen sizes and prevent any horizontal overflow issues.
        """
        if not table_lines:
            return []

        # Parse table rows
        rows: list[list[str]] = []
        for line in table_lines:
            if self._is_table_separator(line):
                continue  # Skip separator line
            # Remove outer pipes and split
            stripped = line.strip()
            if stripped.startswith("|"):
                stripped = stripped[1:]
            if stripped.endswith("|"):
                stripped = stripped[:-1]
            cells = [cell.strip() for cell in stripped.split("|")]
            rows.append(cells)

        if not rows:
            return []

        # Use vertical list format for all tables (mobile-friendly)
        result: list[str] = []
        result.append("")  # Empty line before table

        headers = rows[0] if rows else []

        for row_idx, row in enumerate(rows[1:], start=1):
            result.append(f"*Item {row_idx}:*")
            for header, value in zip(headers, row):
                result.append(f"  • {header}: {value}")
            if row_idx < len(rows) - 1:  # Add spacing between items
                result.append("")

        result.append("")  # Empty line after table
        return result

    def _format_header(self, line: str) -> str:
        """Convert markdown header to bold text with decorations."""
        match = re.match(r"^(#+)\s+(.+)$", line.strip())
        if not match:
            return line

        level = len(match.group(1))
        text = match.group(2).strip()

        if level == 1:
            # H1: Bold text with double line separator
            return f"\n*{text.upper()}*\n{'═' * min(len(text), 30)}"
        elif level == 2:
            # H2: Bold text with single line separator
            return f"\n*{text}*\n{'─' * min(len(text), 30)}"
        else:
            # H3+: Just bold text
            return f"\n*{text}*"

    def _format_blockquote(self, line: str) -> str:
        """Format blockquote by removing > and adding indentation."""
        # Remove leading > and optional space
        text = re.sub(r"^>\s?", "", line.strip())
        return f"  ┃ {text}"

    async def _send_response(
        self,
        to: PhoneNumber,
        response: GeneratedAssistantMessage[Any] | str,
        reply_to: str | None = None,
    ) -> None:
        """Send response message(s) to user with enhanced error handling and retry logic."""
        # Extract text from GeneratedAssistantMessage if needed
        response_text = (
            response.text
            if isinstance(response, GeneratedAssistantMessage)
            else response
        )

        # Apply WhatsApp-specific markdown formatting
        response_text = self._format_whatsapp_markdown(response_text)

        logger.info(
            f"[SEND_RESPONSE] Sending response to {to} (length: {len(response_text)}, reply_to: {reply_to})"
        )

        # Split messages by line breaks and length
        messages = self._split_message_by_line_breaks(response_text)
        logger.info(f"[SEND_RESPONSE] Split response into {len(messages)} parts")

        # Track sending state to handle partial failures
        successfully_sent_count = 0
        failed_parts: list[dict[str, Any]] = []

        for i, msg in enumerate(messages):
            logger.debug(
                f"[SEND_RESPONSE] Sending message part {i + 1}/{len(messages)} to {to}"
            )

            # Show typing indicator before each message if configured
            if self.config.typing_indicator:
                try:
                    logger.debug(
                        f"[SEND_RESPONSE] Sending typing indicator to {to} for message {i + 1}"
                    )
                    await self.provider.send_typing_indicator(
                        to, self.config.typing_duration
                    )
                except Exception as e:
                    # Don't let typing indicator failures break message sending
                    logger.warning(
                        f"[SEND_RESPONSE] Failed to send typing indicator: {e}"
                    )

            # Only quote the first message if quote_messages is enabled
            quoted_id = reply_to if i == 0 else None

            # Retry logic for individual message parts
            max_retries = 3
            retry_delay = 1.0
            sent_successfully = False

            for attempt in range(max_retries + 1):
                try:
                    sent_message = await self.provider.send_text_message(
                        to=to, text=msg, quoted_message_id=quoted_id
                    )
                    logger.debug(
                        f"[SEND_RESPONSE] Successfully sent message part {i + 1} to {to}: {sent_message.id}"
                    )
                    sent_successfully = True
                    successfully_sent_count += 1
                    break

                except Exception as e:
                    if attempt < max_retries:
                        # Calculate exponential backoff delay
                        delay = retry_delay * (2**attempt)
                        logger.warning(
                            f"[SEND_RESPONSE] Failed to send message part {i + 1} to {to} (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Final failure - log but continue with next parts
                        logger.error(
                            f"[SEND_RESPONSE_ERROR] Failed to send message part {i + 1} to {to} after {max_retries + 1} attempts: {e}"
                        )
                        failed_parts.append(
                            {
                                "part_number": i + 1,
                                "text": msg[:100] + "..." if len(msg) > 100 else msg,
                                "error": str(e),
                            }
                        )

            # If this part failed, continue with next parts instead of stopping
            if not sent_successfully:
                logger.warning(
                    f"[SEND_RESPONSE] Message part {i + 1} failed, continuing with remaining parts"
                )

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

        # Log final sending results
        if failed_parts:
            logger.error(
                f"[SEND_RESPONSE] Completed sending with {successfully_sent_count}/{len(messages)} parts successful, {len(failed_parts)} failed"
            )
            logger.error(f"[SEND_RESPONSE] Failed parts details: {failed_parts}")

            # Optionally send error notification for partial failures
            if successfully_sent_count == 0:
                # Total failure - send error message
                await self._send_error_message(to, reply_to)
            elif len(failed_parts) > 0:
                # Partial failure - optionally notify user
                try:
                    error_msg = f"⚠️ Algumas partes da mensagem podem não ter sido enviadas devido a problemas técnicos. {len(failed_parts)} de {len(messages)} partes falharam."
                    formatted_error_msg = self._format_whatsapp_markdown(error_msg)
                    await self.provider.send_text_message(
                        to=to, text=formatted_error_msg
                    )
                except Exception as e:
                    logger.warning(
                        f"[SEND_RESPONSE] Failed to send partial failure notification: {e}"
                    )
        else:
            logger.info(
                f"[SEND_RESPONSE] Successfully sent all {len(messages)} message parts to {to}"
            )

    def _split_message_by_line_breaks(self, text: str) -> Sequence[str]:
        """Split message by line breaks first, then by length if needed with enhanced validation."""
        if not text or not text.strip():
            return ["[Mensagem vazia]"]  # Portuguese: "Empty message"

        try:
            # First split by double line breaks (paragraphs)
            paragraphs = text.split("\n\n")
            messages: MutableSequence[str] = []

            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue

                # Check if paragraph is a list (has list markers)
                lines = paragraph.split("\n")
                is_list_paragraph = self._is_list_content(lines)

                if is_list_paragraph:
                    # Group list items together instead of splitting each line
                    grouped_list = self._group_list_items(lines)
                    messages.extend(grouped_list)
                else:
                    # For non-list paragraphs, split by lines as before
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

            # Filter out empty messages and validate
            final_messages = []
            for msg in messages:
                if msg and msg.strip():
                    # Ensure message doesn't exceed WhatsApp's absolute limit
                    if len(msg) > 65536:  # WhatsApp's hard limit
                        # Split even further if needed
                        for i in range(0, len(msg), 65536):
                            chunk = msg[i : i + 65536]
                            if chunk.strip():
                                final_messages.append(chunk.strip())
                    else:
                        final_messages.append(msg.strip())

            # If no valid messages were created, return a placeholder
            if not final_messages:
                final_messages = [
                    "[Não foi possível processar a mensagem]"
                ]  # Portuguese: "Could not process message"

            # Apply max split messages limit
            if len(final_messages) > self.config.max_split_messages:
                logger.info(
                    f"[SPLIT_MESSAGE] Limiting messages from {len(final_messages)} to {self.config.max_split_messages}"
                )
                final_messages = self._apply_message_limit(
                    final_messages, self.config.max_split_messages
                )

            logger.debug(
                f"[SPLIT_MESSAGE] Split message of {len(text)} chars into {len(final_messages)} parts"
            )

            # Log if we have many parts (potential performance issue)
            if len(final_messages) > 10:
                logger.warning(
                    f"[SPLIT_MESSAGE] Large message split into {len(final_messages)} parts - this may take time to send"
                )

            return final_messages

        except Exception as e:
            logger.error(f"[SPLIT_MESSAGE_ERROR] Error splitting message: {e}")
            # Fallback: return original message truncated if needed
            if len(text) <= self.config.max_message_length:
                return [text]
            else:
                return [text[: self.config.max_message_length]]

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

    def _is_list_content(self, lines: Sequence[str]) -> bool:
        """Check if lines contain list markers (numbered or bullet points)."""
        if not lines:
            return False

        list_markers = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for numbered lists: "1.", "2)", "1 -", etc.
            if re.match(r"^\d+[\.\)]\s", stripped) or re.match(
                r"^\d+\s[-–—]\s", stripped
            ):
                list_markers += 1
            # Check for bullet points: "•", "*", "-", "→", etc.
            elif re.match(r"^[•\*\-→▪►]\s", stripped):
                list_markers += 1

        # If more than 50% of non-empty lines are list items, consider it a list
        non_empty_lines = sum(1 for line in lines if line.strip())
        return non_empty_lines > 0 and (list_markers / non_empty_lines) >= 0.5

    def _group_list_items(self, lines: Sequence[str]) -> Sequence[str]:
        """Group list items together to avoid splitting each item into a separate message."""
        if not lines:
            return []

        messages: MutableSequence[str] = []
        current_group: MutableSequence[str] = []
        current_length = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Calculate potential length if we add this line
            potential_length = current_length + len(line)
            if current_group:
                potential_length += 1  # For the newline

            # If adding this line would exceed the limit, save current group and start new one
            if potential_length > self.config.max_message_length and current_group:
                messages.append("\n".join(current_group))
                current_group = [line]
                current_length = len(line)
            else:
                current_group.append(line)
                current_length = potential_length

        # Add remaining group
        if current_group:
            messages.append("\n".join(current_group))

        return messages

    def _apply_message_limit(
        self, messages: Sequence[str], max_messages: int
    ) -> Sequence[str]:
        """
        Apply limit to number of split messages.
        If exceeded, group remaining messages together.
        """
        if len(messages) <= max_messages:
            return messages

        # Keep first (max_messages - 1) messages as-is
        limited_messages = list(messages[: max_messages - 1])

        # Group all remaining messages into one
        remaining = messages[max_messages - 1 :]

        # Try to join remaining messages with double line breaks
        grouped_remaining = "\n\n".join(remaining)

        # If the grouped message is too long, split it more intelligently
        if len(grouped_remaining) > self.config.max_message_length:
            # Split by chunks that fit
            chunks: MutableSequence[str] = []
            current_chunk = ""

            for msg in remaining:
                test_chunk = current_chunk + ("\n\n" if current_chunk else "") + msg

                if len(test_chunk) <= self.config.max_message_length:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = msg

                    # If single message is too long, hard split it
                    if len(current_chunk) > self.config.max_message_length:
                        for i in range(
                            0, len(current_chunk), self.config.max_message_length
                        ):
                            chunk = current_chunk[
                                i : i + self.config.max_message_length
                            ]
                            chunks.append(chunk)
                        current_chunk = ""

            if current_chunk:
                chunks.append(current_chunk)

            limited_messages.extend(chunks)
        else:
            limited_messages.append(grouped_remaining)

        return limited_messages

    async def _send_error_message(
        self, to: PhoneNumber, reply_to: str | None = None
    ) -> None:
        """Send error message to user."""
        logger.warning(f"[SEND_ERROR] Sending error message to {to}")
        try:
            # Only quote if quote_messages is enabled
            quoted_id = reply_to if self.config.quote_messages else None
            formatted_error = self._format_whatsapp_markdown(self.config.error_message)
            await self.provider.send_text_message(
                to=to, text=formatted_error, quoted_message_id=quoted_id
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

    async def _send_rate_limit_message(self, to: PhoneNumber) -> None:
        """Send rate limit notification to user."""
        message = "You're sending messages too quickly. Please wait a moment before sending more messages."
        logger.info(f"[RATE_LIMIT] Sending rate limit message to {to}")
        try:
            formatted_message = self._format_whatsapp_markdown(message)
            await self.provider.send_text_message(to=to, text=formatted_message)
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
        self, payload: WhatsAppWebhookPayload, chat_id: ChatId | None = None
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
            if data.key.fromMe:
                logger.debug("[MESSAGE_UPSERT] Skipping outgoing message")
                return None

            logger.info("[MESSAGE_UPSERT] Parsing message from Evolution API data")
            # Parse message directly from data (which contains the message info)

            message = self._parse_evolution_message_from_data(
                data,
                from_number=payload.data.key.remoteJidAlt
                if "@lid" in payload.data.key.remoteJid
                else payload.data.key.remoteJid,
            )

            if message:
                logger.info(
                    f"[MESSAGE_UPSERT] ✅ Parsed message: {message.id} from {message.from_number}"
                )

                # CRITICAL FIX: Store the actual remoteJid for @lid numbers
                # This is needed to send messages back to the correct WhatsApp JID
                if "@lid" in payload.data.key.remoteJid:
                    logger.info(
                        f"[MESSAGE_UPSERT] 🔑 Detected @lid number. Storing remoteJid: {payload.data.key.remoteJid} for phone: {message.from_number}"
                    )
                    message.remote_jid = payload.data.key.remoteJid

                logger.info(
                    f"[MESSAGE_UPSERT] About to call handle_message with {len(self._response_callbacks)} callbacks"
                )

                result = await self.handle_message(message, chat_id=chat_id)

                logger.info(
                    f"[MESSAGE_UPSERT] ✅ handle_message completed. Result: {result is not None}"
                )
                return result
            else:
                logger.warning(
                    "[MESSAGE_UPSERT] ❌ Failed to parse message or message was skipped (empty/placeholder content)"
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

    def _parse_evolution_message_from_data(
        self, data: Data, from_number: str
    ) -> WhatsAppMessage | None:
        """Parse Evolution API message from webhook data field."""
        logger.debug("[PARSE_EVOLUTION] Parsing Evolution message from data")

        try:
            # Extract key information
            key = data.key
            message_id = key.id

            if not message_id or not from_number:
                logger.warning("[PARSE_EVOLUTION] Missing message ID or from_number")
                return None

            logger.debug(
                f"[PARSE_EVOLUTION] Message ID: {message_id}, From: {from_number}"
            )

            # Get message type from the data
            message_type = data.messageType or ""
            logger.info(f"[PARSE_EVOLUTION] Message type: {message_type}")

            # Handle different message types
            if message_type == "editedMessage":
                logger.info(
                    "[PARSE_EVOLUTION] Handling editedMessage - treating as text message"
                )
                # For edited messages, we might not have the content, but we should still process it
                return WhatsAppTextMessage(
                    id=message_id,
                    push_name=data.pushName or "Unknown",
                    from_number=from_number,
                    to_number=self.provider.get_instance_identifier(),
                    timestamp=datetime.fromtimestamp(
                        (data.messageTimestamp or 0) / 1000  # Convert from milliseconds
                    ),
                    text="[Message was edited]",  # Placeholder text for edited messages
                )

            # Check if there's a message field
            if data.message:
                msg_content = data.message

                # Handle text messages
                if msg_content.conversation:
                    text = msg_content.conversation
                    logger.debug(
                        f"[PARSE_EVOLUTION] Found conversation text: {text[:50] if text else 'None'}..."
                    )

                    return WhatsAppTextMessage(
                        id=message_id,
                        push_name=data.pushName or "Unknown",
                        from_number=from_number,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0)
                            / 1000  # Convert from milliseconds
                        ),
                        text=text or ".",
                    )

                # Handle extended text messages (extendedTextMessage is not in Message BaseModel, needs dict access)
                # TODO: Add extendedTextMessage to Message BaseModel if needed
                elif hasattr(msg_content, "__dict__") and msg_content.__dict__.get(
                    "extendedTextMessage"
                ):
                    extended_text_message = msg_content.__dict__.get(
                        "extendedTextMessage"
                    )
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
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        text=text,
                    )

                # Handle image messages
                elif msg_content.imageMessage:
                    logger.debug("[PARSE_EVOLUTION] Found image message")
                    image_msg = msg_content.imageMessage
                    return WhatsAppImageMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=image_msg.url if image_msg else "",
                        media_mime_type=image_msg.mimetype
                        if image_msg and image_msg.mimetype
                        else "image/jpeg",
                        caption=image_msg.caption
                        if image_msg and image_msg.caption
                        else "",
                    )

                # Handle document messages
                elif msg_content.documentMessage:
                    logger.debug("[PARSE_EVOLUTION] Found document message")
                    doc_msg = msg_content.documentMessage
                    return WhatsAppDocumentMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=doc_msg.url if doc_msg else "",
                        media_mime_type=doc_msg.mimetype
                        if doc_msg and doc_msg.mimetype
                        else "application/octet-stream",
                        filename=doc_msg.fileName
                        if doc_msg and doc_msg.fileName
                        else "",
                        caption=doc_msg.caption if doc_msg and doc_msg.caption else "",
                    )

                # Handle audio messages
                elif msg_content.audioMessage:
                    logger.debug("[PARSE_EVOLUTION] Found audio message")
                    audio_msg = msg_content.audioMessage
                    return WhatsAppAudioMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=audio_msg.url if audio_msg else "",
                        media_mime_type=audio_msg.mimetype
                        if audio_msg and audio_msg.mimetype
                        else "audio/ogg",
                    )
                elif msg_content.videoMessage:
                    logger.debug("[PARSE_EVOLUTION] Found video message")
                    video_msg = msg_content.videoMessage
                    return WhatsAppVideoMessage(
                        id=message_id,
                        from_number=from_number,
                        push_name=data.pushName or "Unknown",
                        caption=video_msg.caption
                        if video_msg and video_msg.caption
                        else None,
                        to_number=self.provider.get_instance_identifier(),
                        timestamp=datetime.fromtimestamp(
                            (data.messageTimestamp or 0) / 1000
                        ),
                        media_url=video_msg.url if video_msg else "",
                        media_mime_type=video_msg.mimetype
                        if video_msg and video_msg.mimetype
                        else "",
                    )
                else:
                    logger.warning(
                        f"[PARSE_EVOLUTION] Unknown message type in content: {msg_content.__class__.__name__}"
                    )

            # If we get here and message is empty but we have messageType info, skip processing
            elif message_type and message_type != "":
                logger.info(
                    f"[PARSE_EVOLUTION] Empty message content with messageType '{message_type}' - skipping to avoid empty message processing"
                )
                # Return None to skip processing instead of creating placeholder messages
                # This prevents the agent from receiving empty/placeholder content
                return None

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

    async def send_message(
        self,
        to: PhoneNumber,
        message: str,
        reply_to: str | None = None,
    ) -> bool:
        """
        Send a message independently to a WhatsApp number.

        Args:
            to: The phone number to send the message to (e.g., "5511999999999")
            message: The message text to send
            reply_to: Optional message ID to reply to

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        logger.info(f"[SEND_MESSAGE] Sending independent message to {to}")

        if not self._running:
            logger.error("[SEND_MESSAGE] Bot is not running")
            return False

        if not message or not message.strip():
            logger.error("[SEND_MESSAGE] Message is empty")
            return False

        try:
            if "@" not in to:
                to += "@s.whatsapp.net"

            await self._send_response(to, message, reply_to)
            logger.info(f"[SEND_MESSAGE] ✅ Message sent successfully to {to}")
            return True
        except Exception as e:
            logger.error(
                f"[SEND_MESSAGE] ❌ Failed to send message to {to}: {e}", exc_info=True
            )
            return False

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
