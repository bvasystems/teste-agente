from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.message import Message
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool

type WithoutStructuredOutput = None


@runtime_checkable
class SupportsStreaming(Protocol):
    async def stream_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> AsyncIterator[Generation[T]]: ...
