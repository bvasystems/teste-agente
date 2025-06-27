from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.providers.amazon.adapters.agentle_tool_to_bedrock_tool_adapter import (
    AgentleToolToBedrockToolAdapter,
)
from agentle.generations.providers.amazon.models.bytes_source import BytesSource
from agentle.generations.providers.amazon.models.content_block import ContentBlock
from agentle.generations.providers.amazon.models.document_block import DocumentBlock
from agentle.generations.providers.amazon.models.document_content import DocumentContent
from agentle.generations.providers.amazon.models.image_block import ImageBlock
from agentle.generations.providers.amazon.models.image_content import ImageContent
from agentle.generations.providers.amazon.models.text_content import TextContent
from agentle.generations.providers.amazon.models.tool_result_block import (
    ToolResultBlock,
)
from agentle.generations.providers.amazon.models.tool_result_content import (
    ToolResultContent,
)
from agentle.generations.providers.amazon.models.tool_result_content_block import (
    ToolResultContentBlock,
)
from agentle.generations.providers.amazon.models.tool_use_block import ToolUseBlock
from agentle.generations.providers.amazon.models.tool_use_content import ToolUseContent
from agentle.generations.providers.amazon.models.video_block import VideoBlock
from agentle.generations.providers.amazon.models.video_content import VideoContent
from agentle.generations.tools.tool import Tool
from agentle.generations.tools.tool_execution_result import ToolExecutionResult

_MIME_TYPE_MAP: Mapping[
    str,
    Literal[
        "pdf",
        "csv",
        "doc",
        "docx",
        "xls",
        "xlsx",
        "html",
        "txt",
        "md",
    ],
] = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}


@dataclass(frozen=True)
class AgentlePartToBotoContent(
    Adapter[
        TextPart | FilePart | ToolExecutionSuggestion | ToolExecutionResult,
        ContentBlock,
    ]
):
    tool_adapter: AgentleToolToBedrockToolAdapter = field(
        default_factory=AgentleToolToBedrockToolAdapter
    )

    @override
    def adapt(
        self,
        _f: TextPart | FilePart | Tool | ToolExecutionSuggestion | ToolExecutionResult,
    ) -> ContentBlock:
        match _f:
            case TextPart():
                return TextContent(text=str(_f.text))
            case FilePart():
                mime_type = _f.mime_type
                if mime_type.startswith("image/"):
                    return ImageContent(
                        image=ImageBlock(
                            format=cast(
                                Literal["jpeg", "png", "gif", "webp"],
                                mime_type.split()[1],
                            ),
                            source=BytesSource(bytes=_f.base64),
                        )
                    )
                elif mime_type.startswith("video/"):
                    return VideoContent(
                        video=VideoBlock(
                            format="mp4", source=BytesSource(bytes=_f.base64)
                        )
                    )
                elif mime_type in [
                    "application/pdf",
                    "text/csv",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "text/html",
                    "text/plain",
                    "text/markdown",
                ]:
                    return DocumentContent(
                        document=DocumentBlock(
                            format=_MIME_TYPE_MAP[mime_type],
                            name="file",
                            source=BytesSource(bytes=_f.base64),
                        )
                    )
                else:
                    raise ValueError(f"{mime_type} is not supported as mimetype")
            case Tool():
                raise ValueError("type `Tool` is not supported yet by the API.")
            case ToolExecutionSuggestion():
                return ToolUseContent(
                    toolUse=ToolUseBlock(
                        toolUseId=_f.id, name=_f.tool_name, input=_f.args
                    )
                )
            case ToolExecutionResult():
                return ToolResultContent(
                    toolResult=ToolResultBlock(
                        toolUseId=_f.suggestion.id,
                        content=[ToolResultContentBlock(text=str(_f.result))],
                    )
                )
