from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage

if TYPE_CHECKING:
    from google.cloud.firestore_v1 import AsyncClient


class FirebaseConversationStore(ConversationStore):
    _client: AsyncClient
    _collection_name: str

    def __init__(
        self,
        client: AsyncClient,
        message_limit: int | None = None,
        override_old_messages: bool | None = None,
        collection_name: str = "conversations",
    ) -> None:
        super().__init__(message_limit, override_old_messages)
        self._client = client
        self._collection_name = collection_name

    @override
    async def add_message_async(
        self, chat_id: str, message: DeveloperMessage | UserMessage | AssistantMessage
    ) -> None:
        from google.cloud import firestore

        message_dict = message.model_dump()
        message_dict["timestamp"] = firestore.SERVER_TIMESTAMP

        # Get current messages to apply limits
        current_messages = await self.get_conversation_history_async(chat_id)

        # Apply message limit logic
        if self.message_limit is not None:
            if len(current_messages) >= self.message_limit:
                if self.override_old_messages:
                    # Remove oldest messages to make room
                    messages_to_remove = len(current_messages) - self.message_limit + 1
                    # Delete oldest messages from Firestore
                    doc_ref = self._client.collection(self._collection_name).document(
                        chat_id
                    )
                    messages_collection = doc_ref.collection("messages")

                    # Get oldest messages to delete
                    oldest_messages = messages_collection.order_by("timestamp").limit(
                        messages_to_remove
                    )

                    async for doc in oldest_messages.stream():
                        await doc.reference.delete()
                else:
                    # Don't add message if limit reached and not overriding
                    return

        # Add the new message
        doc_ref = self._client.collection(self._collection_name).document(chat_id)
        messages_collection = doc_ref.collection("messages")
        await messages_collection.add(message_dict)

    @override
    async def clear_conversation_async(self, chat_id: str) -> None:
        doc_ref = self._client.collection(self._collection_name).document(chat_id)
        messages_collection = doc_ref.collection("messages")

        # Delete all messages in the conversation
        async for doc in messages_collection.stream():
            await doc.reference.delete()

    @override
    async def get_conversation_history_async(
        self, chat_id: str
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]:
        doc_ref = self._client.collection(self._collection_name).document(chat_id)
        messages_collection = doc_ref.collection("messages")

        # Get messages ordered by timestamp
        query = messages_collection.order_by("timestamp")

        messages = []
        async for doc in query.stream():
            message_data = doc.to_dict()

            # Remove timestamp and firestore metadata for model reconstruction
            message_data.pop("timestamp", None)

            if "role" in message_data:
                if message_data["role"] == "developer":
                    messages.append(DeveloperMessage.model_validate(message_data))
                elif message_data["role"] == "user":
                    messages.append(UserMessage.model_validate(message_data))
                elif message_data["role"] == "assistant":
                    messages.append(AssistantMessage.model_validate(message_data))

        return messages
