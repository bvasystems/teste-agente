from __future__ import annotations

from typing import TYPE_CHECKING, override

import ujson
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_tool_call_param import (
        ChatCompletionMessageToolCallParam,
        Function,
    )


class AgentleToolExecutionSugestionToOpenaiTool(
    Adapter[ToolExecutionSuggestion, "ChatCompletionMessageToolCallParam"]
):
    @override
    def adapt(self, _f: ToolExecutionSuggestion) -> ChatCompletionMessageToolCallParam:
        return ChatCompletionMessageToolCallParam(
            id=_f.id,
            function=Function(arguments=ujson.dumps(_f.args), name=_f.tool_name),
            type="function",
        )
