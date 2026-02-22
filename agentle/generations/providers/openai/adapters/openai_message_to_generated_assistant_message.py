from __future__ import annotations

import json
from collections.abc import MutableSequence
from typing import TYPE_CHECKING, cast

import ujson
from pydantic import BaseModel
from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage


class OpenAIMessageToGeneratedAssistantMessageAdapter[T](
    Adapter[
        "ChatCompletionMessage | ParsedChatCompletionMessage[T]",
        GeneratedAssistantMessage[T],
    ]
):
    def adapt(
        self, _f: ChatCompletionMessage | ParsedChatCompletionMessage[T]
    ) -> GeneratedAssistantMessage[T]:
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
        )
        from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage

        if isinstance(_f, ParsedChatCompletionMessage):
            parsed = cast(T, _f.parsed)
            if parsed is None:
                raise ValueError(
                    "Could not get parsed response schema for chat completion."
                )

            return GeneratedAssistantMessage[T](
                role="assistant",
                parts=[
                    TextPart(text=ujson.dumps(cast(BaseModel, _f.parsed).model_dump()))
                ],
                parsed=parsed,
            )

        openai_message = _f
        
        tool_calls: MutableSequence[ChatCompletionMessageToolCall] = (
            openai_message.tool_calls or []
        )
        
        if openai_message.content is None and not tool_calls:
            raise ValueError("Contents and tool calls of OpenAI message are both None. Couldn't proceed.")

        tool_parts = [
            ToolExecutionSuggestion(
                id=tool_call.id,
                tool_name=tool_call.function.name,
                args=json.loads(tool_call.function.arguments or "{}"),
            )
            for tool_call in tool_calls
        ]

        parts = []
        if openai_message.content is not None:
            parts.append(TextPart(text=openai_message.content))
        
        parts.extend(tool_parts)

        return GeneratedAssistantMessage[T](
            parts=parts,
            parsed= cast(T, None),
        )
