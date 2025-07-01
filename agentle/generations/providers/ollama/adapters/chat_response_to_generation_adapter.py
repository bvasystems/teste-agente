from collections.abc import MutableSequence
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import TYPE_CHECKING, cast, override
from uuid import uuid4

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)

if TYPE_CHECKING:
    from ollama._types import ChatResponse


@dataclass(frozen=True)
class ChatResponseToGenerationAdapter[T](Adapter["ChatResponse", Generation[T]]):
    model: str = field()
    response_schema: type[T] | None = field(default=None)

    @override
    def adapt(self, _f: ChatResponse) -> Generation[T]:
        from pydantic import BaseModel

        parsed = None

        if self.response_schema:
            bm = cast(BaseModel, self.response_schema)
            parsed = bm.model_validate(json.loads(_f.message.content or "{}"))

        parts: MutableSequence[TextPart | ToolExecutionSuggestion] = []

        text_content = _f.message.content
        if text_content:
            parts.append(TextPart(text=text_content))

        tool_calls = _f.message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                parts.append(
                    ToolExecutionSuggestion(
                        id=str(uuid4()),
                        tool_name=tool_call.function.name,
                        args=tool_call.function.arguments,
                    )
                )

        return Generation[T](
            id=uuid4(),
            object="chat.generation",
            created=datetime.now(),
            choices=[
                Choice(
                    index=0,
                    message=GeneratedAssistantMessage(
                        parts=parts, parsed=cast(T, parsed)
                    ),
                )
            ],
            model=self.model,
            usage=Usage.zero(),
        )
