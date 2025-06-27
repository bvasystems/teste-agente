from dataclasses import dataclass, field
from typing import override
from rsb.adapters.adapter import Adapter

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.amazon.adapters.agentle_part_to_boto_content import (
    AgentlePartToBotoContent,
)
from agentle.generations.providers.amazon.models.message import Message


@dataclass(frozen=True)
class AgentleMessageToBotoMessage(
    Adapter[UserMessage | AssistantMessage | DeveloperMessage, Message]
):
    part_adapter: AgentlePartToBotoContent = field(
        default_factory=AgentlePartToBotoContent
    )

    @override
    def adapt(self, _f: UserMessage | AssistantMessage | DeveloperMessage) -> Message:
        if isinstance(_f, DeveloperMessage):
            raise ValueError("Developer messages are not supported in this API.")

        return Message(
            role="user" if isinstance(_f, UserMessage) else "assistant",
            content=[self.part_adapter.adapt(p) for p in _f.parts],
        )
