"""
Callback-based conversation store for the Agentle framework.

This module provides a CallbackConversationStore that allows users to define
custom callback functions for handling conversation persistence operations.
This provides maximum flexibility for integrating with any storage system
or implementing custom logic without subclassing the abstract base class.

Example:
    ```python
    import asyncio
    from agentle.agents.conversations.callback_conversation_store import CallbackConversationStore
    from agentle.generations.models.messages.user_message import UserMessage
    from agentle.generations.models.message_parts.text import TextPart

    # Custom storage (could be Redis, custom API, etc.)
    custom_storage = {}

    async def get_messages(chat_id: str):
        return custom_storage.get(chat_id, [])

    async def add_message(chat_id: str, message):
        if chat_id not in custom_storage:
            custom_storage[chat_id] = []
        custom_storage[chat_id].append(message)

    async def clear_messages(chat_id: str):
        custom_storage.pop(chat_id, None)

    # Create the callback store
    store = CallbackConversationStore(
        get_callback=get_messages,
        add_callback=add_message,
        clear_callback=clear_messages,
        message_limit=100,
        override_old_messages=True
    )

    # Use it like any other conversation store
    message = UserMessage(parts=[TextPart(text="Hello!")])
    await store.add_message_async("chat-123", message)
    history = await store.get_conversation_history_async("chat-123")
    ```
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, override

from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage


class CallbackConversationStore(ConversationStore):
    """
    A conversation store that uses user-provided callback functions for persistence.

    This implementation allows users to define custom callback functions for getting,
    adding, and clearing conversation messages. This provides maximum flexibility
    for integrating with any storage system (Redis, custom APIs, cloud storage, etc.)
    or implementing custom business logic without needing to subclass ConversationStore.

    The store handles message limit enforcement automatically, calling the user's
    callbacks as needed to maintain the specified limits.

    Attributes:
        get_callback: Async function to retrieve conversation history for a chat ID
        add_callback: Async function to add a message to storage
        clear_callback: Async function to clear all messages for a chat ID
        message_limit: Maximum number of messages to store per conversation
        override_old_messages: Whether to remove old messages when limit is reached
    """

    def __init__(
        self,
        get_callback: Callable[
            [str],
            Awaitable[Sequence[DeveloperMessage | UserMessage | AssistantMessage]],
        ],
        add_callback: Callable[
            [
                str,
                DeveloperMessage
                | UserMessage
                | AssistantMessage
                | GeneratedAssistantMessage[Any],
            ],
            Awaitable[None],
        ],
        clear_callback: Callable[[str], Awaitable[None]],
        message_limit: int | None = None,
        override_old_messages: bool | None = None,
    ) -> None:
        """
        Initialize the callback-based conversation store.

        Args:
            get_callback: Async function that takes a chat_id and returns the conversation history.
                         Should return a sequence of messages for the given chat ID.
            add_callback: Async function that takes a chat_id and message and stores the message.
                         Note: The store will handle message limits, so this should just store the message.
            clear_callback: Async function that takes a chat_id and clears all messages for that chat.
            message_limit: Maximum number of messages to store per conversation
            override_old_messages: Whether to remove old messages when limit is reached

        Example:
            ```python
            # Using with a custom dictionary store
            storage = {}

            async def get_messages(chat_id: str):
                return storage.get(chat_id, [])

            async def add_message(chat_id: str, message):
                if chat_id not in storage:
                    storage[chat_id] = []
                storage[chat_id].append(message)

            async def clear_messages(chat_id: str):
                storage.pop(chat_id, None)

            store = CallbackConversationStore(
                get_callback=get_messages,
                add_callback=add_message,
                clear_callback=clear_messages
            )
            ```
        """
        super().__init__(message_limit, override_old_messages)
        self._get_callback = get_callback
        self._add_callback = add_callback
        self._clear_callback = clear_callback

    @override
    async def get_conversation_history_async(
        self, chat_id: str
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]:
        """
        Retrieve conversation history using the user-provided get callback.

        Args:
            chat_id: The unique identifier for the conversation

        Returns:
            Sequence of messages representing the conversation history

        Raises:
            Any exceptions raised by the user's get_callback function
        """
        return await self._get_callback(chat_id)

    @override
    async def add_message_async[T = Any](
        self,
        chat_id: str,
        message: DeveloperMessage
        | UserMessage
        | AssistantMessage
        | GeneratedAssistantMessage[T],
    ) -> None:
        """
        Add a message to the conversation using the user-provided add callback.

        This method handles message limit enforcement by retrieving the current
        conversation history and managing message limits before calling the
        user's add callback.

        Args:
            chat_id: The unique identifier for the conversation
            message: The message to add to the conversation

        Raises:
            Any exceptions raised by the user's callback functions
        """
        # Handle message limit logic
        if self.message_limit is not None:
            current_messages = await self.get_conversation_history_async(chat_id)

            if len(current_messages) >= self.message_limit:
                if self.override_old_messages:
                    # Calculate how many messages to remove
                    messages_to_remove = len(current_messages) - self.message_limit + 1

                    # Clear the conversation and re-add the messages we want to keep
                    await self._clear_callback(chat_id)

                    # Keep only the most recent messages (minus the ones we need to remove)
                    messages_to_keep = current_messages[messages_to_remove:]

                    # Re-add the messages we want to keep
                    for msg in messages_to_keep:
                        await self._add_callback(chat_id, msg)
                else:
                    # Don't add message if limit reached and not overriding
                    return

        # Add the new message
        await self._add_callback(chat_id, message)

    @override
    async def clear_conversation_async(self, chat_id: str) -> None:
        """
        Clear conversation history using the user-provided clear callback.

        Args:
            chat_id: The unique identifier for the conversation to clear

        Raises:
            Any exceptions raised by the user's clear_callback function
        """
        await self._clear_callback(chat_id)
