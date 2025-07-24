import abc
from collections.abc import Sequence

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage


class ConversationStore(abc.ABC):
    message_limit: int | None
    override_old_messages: bool | None

    def __init__(
        self,
        message_limit: int | None = None,
        override_old_messages: bool | None = None,
    ) -> None:
        """__init__

        Args:
            message_limit (int | None, optional): the limit of messages to store in the store. Defaults to None.
            override_old_messages (bool | None, optional): indicate if the older messages should start to be removed in order to add new the new ones if the message limit it hit. Defaults to None.
        """
        self.message_limit = message_limit
        self.override_old_messages = override_old_messages

    @abc.abstractmethod
    async def get_conversation_history_async(
        self, chat_id: str
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]: ...

    @abc.abstractmethod
    async def add_message_async(
        self, chat_id: str, message: DeveloperMessage | UserMessage | AssistantMessage
    ) -> None: ...

    @abc.abstractmethod
    async def clear_conversation_async(self, chat_id: str) -> None: ...
