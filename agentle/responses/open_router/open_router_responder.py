import json
import logging
import os
from typing import Any, AsyncIterator, Literal, Type, override

from pydantic import BaseModel, Field, TypeAdapter

import aiohttp
from agentle.responses._streaming.async_stream import AsyncStream
from agentle.responses._streaming.async_stream_impl import AsyncStreamImpl
from agentle.responses.definitions.create_response import CreateResponse
from agentle.responses.definitions.response import Response
from agentle.responses.definitions.response_completed_event import (
    ResponseCompletedEvent,
)
from agentle.responses.definitions.response_stream_event import ResponseStreamEvent
from agentle.responses.definitions.response_stream_type import ResponseStreamType
from agentle.responses.responder_mixin import ResponderMixin

logger = logging.getLogger(__name__)


class OpenRouterResponder(BaseModel, ResponderMixin):
    """
    OpenRouter Responder for the Responses API (Beta).

    This responder implements OpenRouter's new Responses API, which is OpenAI-compatible
    and provides a stateless transformation layer with support for:
    - Basic text generation (streaming and non-streaming)
    - Reasoning capabilities with configurable effort levels
    - Tool/function calling with parallel execution
    - Web search integration with citation annotations
    - Structured output parsing with Pydantic models

    The Responses API is stateless - each request is independent and no conversation
    state is persisted between requests. You must include the full conversation history
    in each request to maintain context.

    Attributes:
        type: Literal type identifier for this responder ("openrouter")
        api_key: Optional OpenRouter API key (falls back to OPENROUTER_API_KEY env var)

    Example:
        ```python
        from agentle.responses.open_router.open_router_responder import OpenRouterResponder

        responder = OpenRouterResponder(api_key="your-key")

        # Basic usage
        response = await responder.respond_async(
            input="What is the meaning of life?",
            model="openai/o4-mini",
            max_output_tokens=500,
        )

        # Streaming
        stream = await responder.respond_async(
            input="Write a story",
            model="openai/o4-mini",
            stream=True,
        )
        async for event in stream:
            if event.type == "ResponseTextDeltaEvent":
                print(event.delta, end="")

        # With reasoning
        response = await responder.respond_async(
            input="Solve this problem...",
            model="openai/o4-mini",
            reasoning=Reasoning(effort=Effort.high),
        )

        # With web search
        response = await responder.respond_async(
            input="What's the latest news?",
            model="openai/o4-mini",
        )
        ```

    See Also:
        - OpenRouter Responses API docs: https://openrouter.ai/docs/api-reference/responses
        - Example usage: examples/openrouter_responses_example.py
    """

    type: Literal["openrouter"] = Field(default="openrouter")
    api_key: str | None = Field(default=None)

    # TypeAdapter for validating ResponseStreamType (discriminated union)
    _response_stream_adapter: TypeAdapter[ResponseStreamType] = TypeAdapter(
        ResponseStreamType
    )

    @override
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
                    return AsyncStreamImpl(
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
        try:
            response_data = json.loads(response_text)
        except Exception:
            logger.error("Failed to decode response JSON: %s", response_text)
            raise

        # Debug: log the raw JSON response to help diagnose missing structured outputs
        try:
            print("HTTP response JSON:")
            print(response_data)
        except Exception:
            pass

        # Parse the response using Pydantic
        parsed_response = (
            Response[TextFormatT]
            .model_validate(response_data)
            .set_text_format(text_format)
        )

        # Avoid forcing access to parsed output here; caller may inspect if available

        # If text_format is provided, parse structured output
        if text_format and issubclass(text_format, BaseModel):
            if parsed_response.output:
                for output_item in parsed_response.output:
                    if output_item.type == "message":
                        for content in output_item.content:
                            if content.type == "output_text" and content.text:
                                # Try to parse as JSON if text_format is provided
                                try:
                                    parsed_data = json.loads(content.text)
                                    content.parsed = text_format.model_validate(
                                        parsed_data
                                    )
                                except Exception:
                                    # If parsing fails, leave parsed as None
                                    pass

            # Fallback: some models populate output_text at the top level
            if not parsed_response.output_parsed and parsed_response.output_text:
                try:
                    parsed_data = json.loads(parsed_response.output_text)
                    # Inject into the first message/output_text content if available
                    for output_item in parsed_response.output:
                        if output_item.type == "message":
                            for content in output_item.content:
                                if content.type == "output_text":
                                    content.parsed = text_format.model_validate(
                                        parsed_data
                                    )
                                    break
                except Exception:
                    pass

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
                    event_data = json.loads(data_str)
                    event_type = event_data.get("type")

                    # Map OpenRouter event types to our event types
                    # The type field uses format like "response.output_text.delta"
                    # but our discriminator expects "ResponseTextDeltaEvent"
                    event_data = self._normalize_event_type(event_data)

                    # Parse event using Pydantic discriminated union
                    event: ResponseStreamType = (
                        self._response_stream_adapter.validate_python(event_data)
                    )

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
                                parsed_data = json.loads(accumulated_text)
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

                except json.JSONDecodeError:
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
