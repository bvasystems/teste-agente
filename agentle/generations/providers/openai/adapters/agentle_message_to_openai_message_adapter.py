from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.openai.adapters.agentle_part_to_openai_part_adapter import (
    AgentlePartToOpenaiPartAdapter,
)
from agentle.generations.providers.openai.adapters.agentle_tool_execution_sugestion_to_openai_tool import (
    AgentleToolExecutionSugestionToOpenaiTool,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_assistant_message_param import (
        ChatCompletionAssistantMessageParam,
    )
    from openai.types.chat.chat_completion_developer_message_param import (
        ChatCompletionDeveloperMessageParam,
    )
    from openai.types.chat.chat_completion_user_message_param import (
        ChatCompletionUserMessageParam,
    )


class AgentleMessageToOpenaiMessageAdapter(
    Adapter[
        AssistantMessage | DeveloperMessage | UserMessage,
        "ChatCompletionDeveloperMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam",
    ]
):
    def adapt(
        self, _f: AssistantMessage | DeveloperMessage | UserMessage
    ) -> (
        ChatCompletionAssistantMessageParam
        | ChatCompletionDeveloperMessageParam
        | ChatCompletionUserMessageParam
    ):
        from openai.types.chat.chat_completion_assistant_message_param import (
            ChatCompletionAssistantMessageParam,
        )
        from openai.types.chat.chat_completion_content_part_text_param import (
            ChatCompletionContentPartTextParam,
        )
        from openai.types.chat.chat_completion_developer_message_param import (
            ChatCompletionDeveloperMessageParam,
        )
        from openai.types.chat.chat_completion_user_message_param import (
            ChatCompletionUserMessageParam,
        )

        message = _f
        tool_execution_sugestion_to_openai_tool_param_adapter = (
            AgentleToolExecutionSugestionToOpenaiTool()
        )
        part_adapter = AgentlePartToOpenaiPartAdapter()

        match message:
            case AssistantMessage():
                from openai._types import NOT_GIVEN
                _message = message.without_tool_calls()
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=[
                        ChatCompletionContentPartTextParam(
                            type="text", text=str(p.text)
                        )
                        for p in _message.parts
                    ],
                    tool_calls=[
                        tool_execution_sugestion_to_openai_tool_param_adapter.adapt(t)
                        for t in message.tool_calls
                    ] if message.tool_calls else NOT_GIVEN,
                )
            case DeveloperMessage():
                return ChatCompletionDeveloperMessageParam(
                    role="developer",
                    content=[
                        ChatCompletionContentPartTextParam(
                            type="text", text=str(p.text)
                        )
                        for p in message.parts
                    ],
                )
            case UserMessage():
                return ChatCompletionUserMessageParam(
                    role="user", content=[part_adapter.adapt(p) for p in message.parts]
                )

        raise NotImplementedError
