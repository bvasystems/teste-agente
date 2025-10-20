import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal, Optional, Union, overload

from pydantic import BaseModel
from rsb.models.field import Field

from agentle.generations.tracing.otel_client import OtelClient
from agentle.prompts.models.prompt import Prompt as PromptModel
from agentle.responses._streaming.async_stream import AsyncStream
from agentle.responses.definitions.conversation_param import ConversationParam
from agentle.responses.definitions.create_response import CreateResponse
from agentle.responses.definitions.function_tool import FunctionTool
from agentle.responses.definitions.include_enum import IncludeEnum
from agentle.responses.definitions.input_item import InputItem
from agentle.responses.definitions.metadata import Metadata
from agentle.responses.definitions.prompt import Prompt
from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.definitions.response import Response
from agentle.responses.definitions.response_stream_event import ResponseStreamEvent
from agentle.responses.definitions.response_stream_options import ResponseStreamOptions
from agentle.responses.definitions.service_tier import ServiceTier
from agentle.responses.definitions.text import Text
from agentle.responses.definitions.tool import Tool
from agentle.responses.definitions.tool_choice_allowed import ToolChoiceAllowed
from agentle.responses.definitions.tool_choice_custom import ToolChoiceCustom
from agentle.responses.definitions.tool_choice_function import ToolChoiceFunction
from agentle.responses.definitions.tool_choice_mcp import ToolChoiceMCP
from agentle.responses.definitions.tool_choice_options import ToolChoiceOptions
from agentle.responses.definitions.tool_choice_types import ToolChoiceTypes
from agentle.responses.definitions.truncation import Truncation


class ResponderMixin(abc.ABC):
    otel_clients: Sequence[OtelClient] = Field(default_factory=list)

    @overload
    async def respond_async[TextFormatT = None](
        self,
        *,
        input: Optional[Union[str, list[InputItem], PromptModel]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, PromptModel]] = None,
        stream: Optional[Literal[False]] = False,
        stream_options: Optional[ResponseStreamOptions] = None,
        conversation: Optional[Union[str, ConversationParam]] = None,
        text_format: type[TextFormatT] | None = None,
        # ResponseProperties parameters
        previous_response_id: Optional[str] = None,
        reasoning: Optional[Reasoning] = None,
        background: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        text: Optional[Text] = None,
        tools: Optional[Sequence[Tool | Callable[..., Any]]] = None,
        tool_choice: Optional[
            Union[
                ToolChoiceOptions,
                ToolChoiceAllowed,
                ToolChoiceTypes,
                ToolChoiceFunction,
                ToolChoiceMCP,
                ToolChoiceCustom,
            ]
        ] = None,
        prompt: Optional[Prompt] = None,
        truncation: Optional[Truncation] = None,
        # ModelResponseProperties parameters
        metadata: Optional[Metadata] = None,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        service_tier: Optional[ServiceTier] = None,
    ) -> Response[TextFormatT]: ...

    @overload
    async def respond_async[TextFormatT = None](
        self,
        *,
        input: Optional[Union[str, list[InputItem], PromptModel]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, PromptModel]] = None,
        stream: Literal[True],
        stream_options: Optional[ResponseStreamOptions] = None,
        conversation: Optional[Union[str, ConversationParam]] = None,
        text_format: type[TextFormatT] | None = None,
        # ResponseProperties parameters
        previous_response_id: Optional[str] = None,
        reasoning: Optional[Reasoning] = None,
        background: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        text: Optional[Text] = None,
        tools: Optional[Sequence[Tool | Callable[..., Any]]] = None,
        tool_choice: Optional[
            Union[
                ToolChoiceOptions,
                ToolChoiceAllowed,
                ToolChoiceTypes,
                ToolChoiceFunction,
                ToolChoiceMCP,
                ToolChoiceCustom,
            ]
        ] = None,
        prompt: Optional[Prompt] = None,
        truncation: Optional[Truncation] = None,
        # ModelResponseProperties parameters
        metadata: Optional[Metadata] = None,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        service_tier: Optional[ServiceTier] = None,
    ) -> AsyncStream[ResponseStreamEvent, TextFormatT]: ...

    @overload
    async def respond_async[TextFormatT = None](
        self,
        *,
        input: Optional[Union[str, list[InputItem], PromptModel]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, PromptModel]] = None,
        stream: bool,
        stream_options: Optional[ResponseStreamOptions] = None,
        conversation: Optional[Union[str, ConversationParam]] = None,
        text_format: type[TextFormatT] | None = None,
        # ResponseProperties parameters
        previous_response_id: Optional[str] = None,
        reasoning: Optional[Reasoning] = None,
        background: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        text: Optional[Text] = None,
        tools: Optional[Sequence[Tool | Callable[..., Any]]] = None,
        tool_choice: Optional[
            Union[
                ToolChoiceOptions,
                ToolChoiceAllowed,
                ToolChoiceTypes,
                ToolChoiceFunction,
                ToolChoiceMCP,
                ToolChoiceCustom,
            ]
        ] = None,
        prompt: Optional[Prompt] = None,
        truncation: Optional[Truncation] = None,
        # ModelResponseProperties parameters
        metadata: Optional[Metadata] = None,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        service_tier: Optional[ServiceTier] = None,
    ) -> AsyncStream[ResponseStreamEvent, TextFormatT]: ...

    async def respond_async[TextFormatT = None](
        self,
        *,
        input: Optional[Union[str, list[InputItem], PromptModel]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, PromptModel]] = None,
        stream: Optional[Literal[False] | Literal[True]] = None,
        stream_options: Optional[ResponseStreamOptions] = None,
        conversation: Optional[Union[str, ConversationParam]] = None,
        text_format: type[TextFormatT] | None = None,
        # ResponseProperties parameters
        previous_response_id: Optional[str] = None,
        reasoning: Optional[Reasoning] = None,
        background: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        text: Optional[Text] = None,
        tools: Optional[Sequence[Tool | Callable[..., Any]]] = None,
        tool_choice: Optional[
            Union[
                ToolChoiceOptions,
                ToolChoiceAllowed,
                ToolChoiceTypes,
                ToolChoiceFunction,
                ToolChoiceMCP,
                ToolChoiceCustom,
            ]
        ] = None,
        prompt: Optional[Prompt] = None,
        truncation: Optional[Truncation] = None,
        # ModelResponseProperties parameters
        metadata: Optional[Metadata] = None,
        top_logprobs: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        service_tier: Optional[ServiceTier] = None,
    ) -> Response[TextFormatT] | AsyncStream[ResponseStreamEvent, TextFormatT]:
        _tools: list[Tool] = []
        if tools:
            for tool in tools:
                if isinstance(tool, Callable):
                    _tools.append(FunctionTool.from_callable(tool))
                else:
                    _tools.append(tool)

        create_response = CreateResponse(
            input=str(input) if isinstance(input, PromptModel) else input,
            model=model,
            include=include,
            parallel_tool_calls=parallel_tool_calls,
            store=store,
            instructions=str(instructions)
            if isinstance(instructions, PromptModel)
            else instructions,
            stream=stream,
            stream_options=stream_options,
            conversation=conversation,
            # ResponseProperties parameters
            previous_response_id=previous_response_id,
            reasoning=reasoning,
            background=background,
            max_output_tokens=max_output_tokens,
            max_tool_calls=max_tool_calls,
            text=text,
            tools=_tools,
            tool_choice=tool_choice,
            prompt=prompt,
            truncation=truncation,
            # ModelResponseProperties parameters
            metadata=metadata,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
            user=user,
            safety_identifier=safety_identifier,
            prompt_cache_key=prompt_cache_key,
            service_tier=service_tier,
        )

        if text_format:
            if not issubclass(text_format, BaseModel):
                raise ValueError(
                    "Currently, only Pydantic models are supported in text_format"
                )

            create_response.set_text_format(text_format)

            # If the caller requested structured output and did not provide tools,
            # prefer to disable tool calling explicitly so the model focuses on JSON.
            if tool_choice is None and not _tools:
                # Import locally to avoid a circular import at module import time
                from agentle.responses.definitions.tool_choice_options import (
                    ToolChoiceOptions as _ToolChoiceOptions,
                )

                create_response.tool_choice = _ToolChoiceOptions.none

            # Disable reasoning by default for structured outputs unless explicitly set,
            # to avoid consuming tokens on hidden reasoning and risking truncation
            if reasoning is None:
                create_response.reasoning = None

        return await self._respond_async(
            create_response,
            text_format=text_format,
        )

    @abc.abstractmethod
    async def _respond_async[TextFormatT = None](
        self,
        create_response: CreateResponse,
        text_format: type[TextFormatT] | None = None,
    ) -> Response[TextFormatT] | AsyncStream[ResponseStreamEvent, TextFormatT]: ...
