from collections.abc import MutableMapping, MutableSequence, Sequence
from typing import override
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage


class LocalConversationStore(ConversationStore):
    __messages: MutableMapping[
        str, MutableSequence[DeveloperMessage | UserMessage | AssistantMessage]
    ]

    def __init__(
        self,
        message_limit: int | None = None,
        override_old_messages: bool | None = None,
    ) -> None:
        super().__init__(message_limit, override_old_messages)
        self.__messages = {}

    @override
    async def add_message_async(
        self, chat_id: str, message: DeveloperMessage | UserMessage | AssistantMessage
    ) -> None:
        if chat_id not in self.__messages:
            self.__messages[chat_id] = []

        # Apply message limit logic
        if self.message_limit is not None:
            current_count = len(self.__messages[chat_id])

            if current_count >= self.message_limit:
                if self.override_old_messages:
                    # Remove oldest messages to make room
                    messages_to_remove = current_count - self.message_limit + 1
                    self.__messages[chat_id] = self.__messages[chat_id][
                        messages_to_remove:
                    ]
                else:
                    # Don't add message if limit reached and not overriding
                    return

        self.__messages[chat_id].append(message)

    @override
    async def get_conversation_history_async(
        self, chat_id: str
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]:
        return self.__messages.get(chat_id, [])

    @override
    async def clear_conversation_async(self, chat_id: str) -> None:
        if chat_id in self.__messages:
            self.__messages[chat_id] = []
