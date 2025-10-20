from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Literal, Optional, Type, Union, overload

import aiohttp
import orjson
from pydantic import BaseModel, TypeAdapter
from rsb.models.field import Field

from agentle.generations.tracing.otel_client_type import OtelClientType
from agentle.prompts.models.prompt import Prompt as PromptModel
from agentle.responses.async_stream import AsyncStream
from agentle.responses.definitions.conversation_param import ConversationParam
from agentle.responses.definitions.create_response import CreateResponse
from agentle.responses.definitions.function_tool import FunctionTool
from agentle.responses.definitions.include_enum import IncludeEnum
from agentle.responses.definitions.input_item import InputItem
from agentle.responses.definitions.metadata import Metadata
from agentle.responses.definitions.prompt import Prompt
from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.definitions.response import Response
from agentle.responses.definitions.response_completed_event import (
    ResponseCompletedEvent,
)
from agentle.responses.definitions.response_stream_event import ResponseStreamEvent
from agentle.responses.definitions.response_stream_options import ResponseStreamOptions
from agentle.responses.definitions.response_stream_type import ResponseStreamType
from agentle.responses.definitions.service_tier import ServiceTier
from agentle.responses.definitions.text import Text
from agentle.responses.definitions.tool import Tool
from agentle.responses.definitions.tool_choice_allowed import ToolChoiceAllowed
from agentle.responses.definitions.tool_choice_custom import ToolChoiceCustom
from agentle.responses.definitions.tool_choice_function import ToolChoiceFunction
from agentle.responses.definitions.tool_choice_mcp import ToolChoiceMCP
from agentle.responses.definitions.tool_choice_options import ToolChoiceOptions
from agentle.responses.definitions.tool_choice_options import (
    ToolChoiceOptions as _ToolChoiceOptions,
)
from agentle.responses.definitions.tool_choice_types import ToolChoiceTypes
from agentle.responses.definitions.truncation import Truncation

logger = logging.getLogger(__name__)


class Responder(BaseModel):
    otel_clients: Sequence[OtelClientType] = Field(default_factory=list)
    api_key: str | None = Field(default=None)
    base_url: str = Field(default="https://openrouter.ai/api/v1")

    # TypeAdapter for validating ResponseStreamType (discriminated union)
    _response_stream_adapter: TypeAdapter[ResponseStreamType] = TypeAdapter(
        ResponseStreamType
    )

    @classmethod
    def from_openrouter(cls, api_key: str | None = None) -> Responder:
        return cls(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    @classmethod
    def from_openai(cls, api_key: str | None = None) -> Responder:
        return cls(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )

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

                create_response.tool_choice = _ToolChoiceOptions.none

            # Disable reasoning by default for structured outputs unless explicitly set,
            # to avoid consuming tokens on hidden reasoning and risking truncation
            if reasoning is None:
                create_response.reasoning = None

        return await self._respond_async(
            create_response,
            text_format=text_format,
        )

    async def _respond_async[TextFormatT](
        self,
        create_response: CreateResponse,
        text_format: Type[TextFormatT] | None = None,
    ) -> Response[TextFormatT] | AsyncStream[ResponseStreamEvent, TextFormatT]:
        _api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not _api_key:
            raise ValueError("No API key provided")

        # Build request payload
        request_payload = create_response.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
            by_alias=True,
        )

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {_api_key}",
            "Content-Type": "application/json",
        }

        # Determine if streaming
        is_streaming = create_response.stream or False

        # Make API request
        base_url = "https://api.openai.com/v1"
        url = f"{base_url}/responses"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request_payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"OpenRouter API error (status {response.status}): {error_text}"
                    )

                if is_streaming:
                    # Read all content within the session context to avoid connection closure
                    content_lines: list[bytes] = []
                    async for line in response.content:
                        content_lines.append(line)

                    # Wrap the buffered content in AsyncStreamImpl
                    return AsyncStream(
                        self._stream_events_from_buffer(
                            content_lines, text_format=text_format
                        ),
                        text_format=text_format,
                    )
                else:
                    return await self._handle_non_streaming_response(
                        response, text_format=text_format
                    )

    async def _handle_non_streaming_response[TextFormatT](
        self,
        response: aiohttp.ClientResponse,
        text_format: Type[TextFormatT] | None = None,
    ) -> Response[TextFormatT]:
        """Handle non-streaming response from OpenRouter Responses API."""
        # Read raw text for debugging, then parse JSON
        response_text = await response.text()
        response_data = orjson.loads(response_text)

        # Parse the response using Pydantic
        parsed_response = (
            Response[TextFormatT]
            .model_validate(response_data)
            .set_text_format(text_format)
        )

        # Avoid forcing access to parsed output here; caller may inspect if available

        # If text_format is provided, parse structured output
        if text_format and issubclass(text_format, BaseModel):
            found_parsed = False
            if parsed_response.output:
                for output_item in parsed_response.output:
                    if output_item.type == "message":
                        for content in output_item.content:
                            if content.type == "output_text" and content.text:
                                # Try to parse as JSON if text_format is provided
                                try:
                                    parsed_data = orjson.loads(content.text)
                                    content.parsed = text_format.model_validate(
                                        parsed_data
                                    )
                                    found_parsed = True
                                except Exception:
                                    # If parsing fails, leave parsed as None
                                    pass

            # Fallback: some models populate output_text at the top level
            if not found_parsed and parsed_response.output_text:
                try:
                    parsed_data = orjson.loads(parsed_response.output_text)
                    # Inject into the first message/output_text content if available
                    for output_item in parsed_response.output:
                        if output_item.type == "message":
                            for content in output_item.content:
                                if content.type == "output_text":
                                    content.parsed = text_format.model_validate(
                                        parsed_data
                                    )
                                    break
                    found_parsed = True
                except Exception:
                    pass

            # If we still don't have parsed content and the response is incomplete
            # due to max_output_tokens, raise a helpful error message so users know
            # it's a token budget/reasoning issue rather than a provider failure.
            status_value = getattr(
                parsed_response.status, "value", parsed_response.status
            )
            incomplete_reason = (
                getattr(
                    parsed_response.incomplete_details.reason,
                    "value",
                    parsed_response.incomplete_details.reason,
                )
                if parsed_response.incomplete_details
                else None
            )

            if (
                not found_parsed
                and status_value == "incomplete"
                and incomplete_reason == "max_output_tokens"
            ):
                raise ValueError(
                    "Structured output not returned: the response was truncated due to max_output_tokens. "
                    + "When text_format is set and reasoning is enabled (especially high), the model may spend the entire budget on reasoning. "
                    + "Increase max_output_tokens or lower reasoning effort to ensure the JSON can be emitted."
                )

        return parsed_response

    async def _stream_events_from_buffer[TextFormatT](
        self,
        content_lines: list[bytes],
        text_format: Type[TextFormatT] | None = None,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Stream events from buffered content lines.

        Parses Server-Sent Events (SSE) format from pre-buffered content:
        event: response.created
        data: {"type":"response.created",...}
        """

        accumulated_text = ""

        for line in content_lines:
            line_str = line.decode("utf-8").strip()

            if not line_str:
                continue

            # Parse SSE format
            if line_str.startswith("event: "):
                # Event type line (we can ignore this as type is in data)
                continue
            elif line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove 'data: ' prefix

                if data_str == "[DONE]":
                    break

                try:
                    event_data = orjson.loads(data_str)
                    event_type = event_data.get("type")

                    # Map OpenRouter event types to our event types
                    # The type field uses format like "response.output_text.delta"
                    # but our discriminator expects "ResponseTextDeltaEvent"
                    event_data = self._normalize_event_type(event_data)

                    # Parse event using Pydantic discriminated union
                    event: ResponseStreamType = (
                        self._response_stream_adapter.validate_python(event_data)
                    )

                    # Ensure response objects inside events know the requested text_format
                    if text_format:
                        resp_obj = getattr(event, "response", None)
                        if resp_obj is not None:
                            try:
                                # Call setter on the response object (no reassignment needed)
                                resp_obj.set_text_format(text_format)
                            except Exception:
                                pass

                    # Accumulate text for structured output parsing
                    if event_type == "response.output_text.delta":
                        accumulated_text += event_data.get("delta", "")

                    # On completion, try to parse structured output
                    if (
                        event_type == "response.completed"
                        and text_format
                        and accumulated_text
                        and isinstance(event, ResponseCompletedEvent)
                    ):
                        if issubclass(text_format, BaseModel):
                            try:
                                parsed_data = orjson.loads(accumulated_text)
                                # Inject parsed data into the event
                                if event.response.output:
                                    for output_item in event.response.output:
                                        if output_item.type == "message":
                                            for content in output_item.content:
                                                if content.type == "output_text":
                                                    content.parsed = (
                                                        text_format.model_validate(
                                                            parsed_data
                                                        )
                                                    )
                                logger.info(
                                    f"Injected parsed content: {event.response.output_parsed}"
                                )
                            except Exception:
                                pass

                    logger.info(f"Yielding event: {event.type}")
                    yield event

                except orjson.JSONDecodeError:
                    # Skip malformed JSON
                    continue
                except Exception as e:
                    # Log but don't crash on validation errors
                    logger.warning(f"Failed to parse event: {e}")
                    continue

    def _normalize_event_type(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize OpenRouter event type to match our discriminated union.

        OpenRouter uses: "response.output_text.delta"
        We expect: "ResponseTextDeltaEvent"
        """
        event_type = event_data.get("type", "")

        # Mapping from OpenRouter event types to our event class names
        type_mapping = {
            "response.created": "ResponseCreatedEvent",
            "response.in_progress": "ResponseInProgressEvent",
            "response.completed": "ResponseCompletedEvent",
            "response.failed": "ResponseFailedEvent",
            "response.incomplete": "ResponseIncompleteEvent",
            "response.queued": "ResponseQueuedEvent",
            "response.error": "ResponseErrorEvent",
            "response.output_item.added": "ResponseOutputItemAddedEvent",
            "response.output_item.done": "ResponseOutputItemDoneEvent",
            "response.content_part.added": "ResponseContentPartAddedEvent",
            "response.content_part.done": "ResponseContentPartDoneEvent",
            "response.output_text.delta": "ResponseTextDeltaEvent",
            "response.output_text.done": "ResponseTextDoneEvent",
            "response.output_text.annotation.added": "ResponseOutputTextAnnotationAddedEvent",
            "response.reasoning.delta": "ResponseReasoningTextDeltaEvent",
            "response.reasoning.done": "ResponseReasoningTextDoneEvent",
            "response.reasoning_summary_part.added": "ResponseReasoningSummaryPartAddedEvent",
            "response.reasoning_summary_part.done": "ResponseReasoningSummaryPartDoneEvent",
            "response.reasoning_summary_text.delta": "ResponseReasoningSummaryTextDeltaEvent",
            "response.reasoning_summary_text.done": "ResponseReasoningSummaryTextDoneEvent",
            "response.refusal.delta": "ResponseRefusalDeltaEvent",
            "response.refusal.done": "ResponseRefusalDoneEvent",
            "response.function_call_arguments.delta": "ResponseFunctionCallArgumentsDeltaEvent",
            "response.function_call_arguments.done": "ResponseFunctionCallArgumentsDoneEvent",
            "response.audio.delta": "ResponseAudioDeltaEvent",
            "response.audio.done": "ResponseAudioDoneEvent",
            "response.audio_transcript.delta": "ResponseAudioTranscriptDeltaEvent",
            "response.audio_transcript.done": "ResponseAudioTranscriptDoneEvent",
            "response.web_search_call.in_progress": "ResponseWebSearchCallInProgressEvent",
            "response.web_search_call.searching": "ResponseWebSearchCallSearchingEvent",
            "response.web_search_call.completed": "ResponseWebSearchCallCompletedEvent",
            "response.file_search_call.in_progress": "ResponseFileSearchCallInProgressEvent",
            "response.file_search_call.searching": "ResponseFileSearchCallSearchingEvent",
            "response.file_search_call.completed": "ResponseFileSearchCallCompletedEvent",
            "response.code_interpreter_call.in_progress": "ResponseCodeInterpreterCallInProgressEvent",
            "response.code_interpreter_call.interpreting": "ResponseCodeInterpreterCallInterpretingEvent",
            "response.code_interpreter_call.completed": "ResponseCodeInterpreterCallCompletedEvent",
            "response.code_interpreter_call.code.delta": "ResponseCodeInterpreterCallCodeDeltaEvent",
            "response.code_interpreter_call.code.done": "ResponseCodeInterpreterCallCodeDoneEvent",
            "response.image_gen_call.in_progress": "ResponseImageGenCallInProgressEvent",
            "response.image_gen_call.generating": "ResponseImageGenCallGeneratingEvent",
            "response.image_gen_call.completed": "ResponseImageGenCallCompletedEvent",
            "response.image_gen_call.partial_image": "ResponseImageGenCallPartialImageEvent",
            "response.mcp_call.in_progress": "ResponseMCPCallInProgressEvent",
            "response.mcp_call.arguments.delta": "ResponseMCPCallArgumentsDeltaEvent",
            "response.mcp_call.arguments.done": "ResponseMCPCallArgumentsDoneEvent",
            "response.mcp_call.completed": "ResponseMCPCallCompletedEvent",
            "response.mcp_call.failed": "ResponseMCPCallFailedEvent",
            "response.mcp_list_tools.in_progress": "ResponseMCPListToolsInProgressEvent",
            "response.mcp_list_tools.completed": "ResponseMCPListToolsCompletedEvent",
            "response.mcp_list_tools.failed": "ResponseMCPListToolsFailedEvent",
            "response.custom_tool_call.input.delta": "ResponseCustomToolCallInputDeltaEvent",
            "response.custom_tool_call.input.done": "ResponseCustomToolCallInputDoneEvent",
        }

        normalized_type = type_mapping.get(event_type, event_type)
        event_data["type"] = normalized_type

        return event_data
