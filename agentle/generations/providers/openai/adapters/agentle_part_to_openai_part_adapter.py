from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool import Tool
from agentle.generations.tools.tool_execution_result import ToolExecutionResult

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_content_part_image_param import (
        ChatCompletionContentPartImageParam,
    )
    from openai.types.chat.chat_completion_content_part_input_audio_param import (
        ChatCompletionContentPartInputAudioParam,
    )
    from openai.types.chat.chat_completion_content_part_param import (
        File,
        FileFile,
    )
    from openai.types.chat.chat_completion_content_part_text_param import (
        ChatCompletionContentPartTextParam,
    )


class AgentlePartToOpenaiPartAdapter(
    Adapter[
        TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion | ToolExecutionResult,
        "ChatCompletionContentPartImageParam | ChatCompletionContentPartInputAudioParam | ChatCompletionContentPartTextParam | File",
    ]
):
    @override
    def adapt(
        self,
        _f: TextPart
        | FilePart
        | Tool[Any]
        | ToolExecutionSuggestion
        | ToolExecutionResult,
    ) -> (
        ChatCompletionContentPartTextParam
        | ChatCompletionContentPartImageParam
        | ChatCompletionContentPartInputAudioParam
        | File
    ):
        part = _f

        match part:
            case (
                TextPart() | Tool() | ToolExecutionResult() | ToolExecutionSuggestion()
            ):
                return ChatCompletionContentPartTextParam(text=str(part), type="text")
            case FilePart():
                mime_type = part.mime_type
                if mime_type.startswith("image/"):
                    data = part.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    return ChatCompletionContentPartImageParam(
                        image_url={
                            "url": base64.b64encode(data).decode(),
                            "detail": "auto",
                        },
                        type="image_url",
                    )
                elif mime_type.startswith("audio/"):
                    data = part.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    return ChatCompletionContentPartInputAudioParam(
                        input_audio={
                            "data": base64.b64encode(data).decode(),
                            "format": "mp3",
                        },
                        type="input_audio",
                    )
                else:
                    return File(type="file", file=FileFile(file_data=part.base64))
