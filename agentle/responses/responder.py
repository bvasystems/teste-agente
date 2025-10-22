from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime
from typing import Any, Literal, Optional, Type, Union, overload

import aiohttp
import orjson
from pydantic import BaseModel, TypeAdapter

# from rsb.coroutines import fire_and_forget
from rsb.models.field import Field

from agentle.generations.tracing.otel_client_type import OtelClientType
from agentle.prompts.models.prompt import Prompt as AgentlePromptType
from agentle.responses.async_stream import AsyncStream
from agentle.responses.pricing.default_pricing_service import (
    DefaultPricingService as DefaultPricingServiceImpl,
)

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
from agentle.responses.pricing.openrouter_pricing_service import (
    OpenRouterPricingService,
)
from agentle.responses.pricing.pricing_service import PricingService

logger = logging.getLogger(__name__)


class Responder(BaseModel):
    """
    Client for interacting with OpenRouter and OpenAI Responses API with built-in observability.

    The Responder class provides a high-level interface for making API calls to OpenRouter
    and OpenAI's Responses API, with automatic tracing, metrics collection, and cost tracking
    through OpenTelemetry (OTel) integration.

    Key Features:
        - Automatic tracing of all API calls through OtelClient integration
        - Cost and usage metrics calculation for all requests
        - Support for both streaming and non-streaming responses
        - Structured output parsing with Pydantic models
        - Resilient error handling that doesn't impact core functionality
        - Support for multiple observability backends simultaneously

    Attributes:
        otel_clients: Sequence of OtelClient instances for observability integration.
            When provided, the Responder will automatically create trace and generation
            contexts for each client, track usage metrics, calculate costs, and handle
            cleanup. Multiple clients can be configured to send telemetry to different
            backends (e.g., Langfuse, custom observability platforms). Tracing operations
            are non-blocking and failures won't impact API calls.

        api_key: API key for authentication. If not provided, will attempt to read from
            OPENROUTER_API_KEY or OPENAI_API_KEY environment variables depending on the
            base_url.

        base_url: Base URL for the API endpoint. Defaults to OpenRouter's API endpoint.
            Use openrouter() or openai() class methods for convenience.

        pricing_service: Service for looking up model pricing to calculate costs.
            Defaults to DefaultPricingService which includes pricing for common models.
            Can be customized with a custom PricingService implementation for additional
            models or dynamic pricing updates.

    Tracing Behavior:
        When otel_clients are configured, the Responder automatically:
        1. Creates trace and generation contexts before making API calls
        2. Tracks request parameters, model settings, and metadata
        3. Extracts token usage from responses (input, output, reasoning tokens)
        4. Calculates costs using the pricing_service
        5. Updates contexts with results, usage, and cost information
        6. Handles errors and records them in tracing contexts
        7. Properly cleans up contexts after completion or failure

        For streaming responses, metrics are accumulated throughout the stream and
        reported when the stream completes. Structured outputs are parsed and included
        in trace metadata.

        All tracing operations are designed to be non-blocking and resilient - failures
        in observability won't cause API calls to fail.

    Example:
        Basic usage without tracing:
        >>> responder = Responder.openrouter(api_key="your-key")
        >>> response = await responder.respond_async(
        ...     input="What is the capital of France?",
        ...     model="openai/gpt-4"
        ... )

        With observability integration:
        >>> from agentle.generations.tracing.langfuse_otel_client import LangfuseOtelClient
        >>> otel_client = LangfuseOtelClient()
        >>> responder = Responder.openrouter(otel_clients=[otel_client])
        >>> response = await responder.respond_async(
        ...     input="What is the capital of France?",
        ...     model="openai/gpt-4"
        ... )
        # Automatically traces the request with usage and cost metrics

        Adding observability dynamically:
        >>> responder = Responder.openrouter()
        >>> # Later, add observability
        >>> otel_client = LangfuseOtelClient()
        >>> responder.append_otel_client(otel_client)
        >>> # Now all subsequent calls will be traced

        With structured output:
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     capital: str
        ...     country: str
        >>> response = await responder.respond_async(
        ...     input="What is the capital of France?",
        ...     model="openai/gpt-4",
        ...     text_format=Answer
        ... )
        >>> parsed = response.output_parsed
        # Structured output is automatically parsed and included in traces

    See Also:
        - OtelClient: Abstract interface for observability clients
        - PricingService: Interface for model pricing lookups
        - Response: Response object returned by non-streaming calls
        - AsyncStream: Stream object returned by streaming calls
    """

    otel_clients: list[OtelClientType] = Field(default_factory=list)
    api_key: str | None = Field(default=None)
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    pricing_service: PricingService = Field(
        default_factory=lambda: DefaultPricingServiceImpl()
    )

    # TypeAdapter for validating ResponseStreamType (discriminated union)
    _response_stream_adapter: TypeAdapter[ResponseStreamType] = TypeAdapter(
        ResponseStreamType
    )

    @classmethod
    def openrouter(
        cls,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClientType] | None = None,
    ) -> Responder:
        """
        Create a Responder configured for OpenRouter API.

        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
            pricing_service: Custom pricing service for cost calculations. Defaults to DefaultPricingService.
            otel_clients: Sequence of OtelClient instances for observability integration.

        Returns:
            Configured Responder instance for OpenRouter.
        """
        pricing_service = OpenRouterPricingService()
        if otel_clients is not None:
            return cls(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                pricing_service=pricing_service,
                otel_clients=list(otel_clients),
            )
        else:
            return cls(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )

    @classmethod
    def openai(
        cls,
        api_key: str | None = None,
        pricing_service: PricingService | None = None,
        otel_clients: list[OtelClientType] | None = None,
    ) -> Responder:
        """
        Create a Responder configured for OpenAI API.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            pricing_service: Custom pricing service for cost calculations. Defaults to DefaultPricingService.
            otel_clients: Sequence of OtelClient instances for observability integration.

        Returns:
            Configured Responder instance for OpenAI.
        """
        if pricing_service is not None and otel_clients is not None:
            return cls(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
                pricing_service=pricing_service,
                otel_clients=list(otel_clients),
            )
        elif pricing_service is not None:
            return cls(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
                pricing_service=pricing_service,
            )
        elif otel_clients is not None:
            return cls(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
                otel_clients=list(otel_clients),
            )
        else:
            return cls(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
            )

    def append_otel_client(self, client: OtelClientType) -> None:
        """
        Add an OtelClient to the responder for observability integration.

        This method allows adding observability clients after the Responder has been
        initialized. Useful for dynamic configuration or conditional observability setup.

        Args:
            client: An OtelClient instance to add to the responder's client list.

        Example:
            >>> responder = Responder.openrouter()
            >>> langfuse_client = LangfuseOtelClient()
            >>> responder.append_otel_client(langfuse_client)
            >>> # Now all API calls will be traced to Langfuse
        """
        self.otel_clients.append(client)

    @overload
    async def respond_async[TextFormatT = None](
        self,
        *,
        input: Optional[Union[str, list[InputItem], AgentlePromptType]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, AgentlePromptType]] = None,
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
        input: Optional[Union[str, list[InputItem], AgentlePromptType]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, AgentlePromptType]] = None,
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
        input: Optional[Union[str, list[InputItem], AgentlePromptType]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, AgentlePromptType]] = None,
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
        input: Optional[Union[str, list[InputItem], AgentlePromptType]] = None,
        model: Optional[str] = None,
        include: Optional[list[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[Union[str, AgentlePromptType]] = None,
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
            input=str(input) if isinstance(input, AgentlePromptType) else input,
            model=model,
            include=include,
            parallel_tool_calls=parallel_tool_calls,
            store=store,
            instructions=str(instructions)
            if isinstance(instructions, AgentlePromptType)
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

        # Initialize tracing
        start_time = datetime.now()
        active_contexts: list[dict[str, Any]] = []
        model = create_response.model or "unknown"

        # Prepare generation config for tracing
        generation_config: dict[str, Any] = {
            "metadata": create_response.metadata.model_dump()
            if create_response.metadata
            else {},
        }

        try:
            # Create tracing contexts if otel_clients are present
            if self.otel_clients:
                try:
                    active_contexts = await self._create_tracing_contexts(
                        model=model,
                        request_payload=request_payload,
                        generation_config=generation_config,
                    )
                except Exception as e:
                    # Log error but don't fail the request
                    logger.error(
                        f"Failed to create tracing contexts, continuing without tracing: {e}",
                        exc_info=True,
                    )

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

                        # Wrap the buffered content with tracing if contexts are active
                        if active_contexts:
                            stream_generator = self._stream_events_with_tracing(
                                content_lines=content_lines,
                                text_format=text_format,
                                active_contexts=active_contexts,
                                start_time=start_time,
                                model=model,
                            )
                        else:
                            stream_generator = self._stream_events_from_buffer(
                                content_lines, text_format=text_format
                            )

                        # Wrap in AsyncStream
                        return AsyncStream(
                            stream_generator,
                            text_format=text_format,
                        )
                    else:
                        # Handle non-streaming response
                        parsed_response = await self._handle_non_streaming_response(
                            response, text_format=text_format
                        )

                        # Update tracing with success
                        if active_contexts:
                            try:
                                await self._update_tracing_success(
                                    active_contexts=active_contexts,
                                    response=parsed_response,
                                    start_time=start_time,
                                    model=model,
                                    text_format=text_format,
                                )
                            except Exception as e:
                                # Log error but don't fail the request
                                logger.error(
                                    f"Failed to update tracing with success: {e}",
                                    exc_info=True,
                                )

                        return parsed_response

        except Exception as e:
            # Update tracing with error
            if active_contexts:
                try:
                    trace_metadata = self._prepare_trace_metadata(
                        model=model,
                        base_url=self.base_url,
                        generation_config=generation_config,
                    )
                    await self._update_tracing_error(
                        active_contexts=active_contexts,
                        error=e,
                        start_time=start_time,
                        metadata=trace_metadata,
                    )
                except Exception as trace_error:
                    # Log error but don't fail the request - we're already handling an error
                    logger.error(
                        f"Failed to update tracing with error: {trace_error}",
                        exc_info=True,
                    )
            # Re-raise the original error
            raise

        finally:
            # Cleanup tracing contexts
            # Note: For streaming, cleanup happens in _stream_events_with_tracing
            # after the stream is consumed. For non-streaming, cleanup happens here.
            if active_contexts and not is_streaming:
                try:
                    await self._cleanup_tracing_contexts(active_contexts)
                except Exception as e:
                    # Log error but don't fail the request
                    logger.error(
                        f"Failed to cleanup tracing contexts: {e}",
                        exc_info=True,
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

    async def _stream_events_with_tracing[TextFormatT](
        self,
        content_lines: list[bytes],
        text_format: Type[TextFormatT] | None,
        active_contexts: list[dict[str, Any]],
        start_time: datetime,
        model: str,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """
        Stream events with tracing integration.

        Wraps the _stream_events_from_buffer generator to accumulate text deltas
        for metrics and update tracing contexts on completion or error. Also handles
        cleanup of tracing contexts after streaming completes.

        Args:
            content_lines: Buffered content lines from the API response
            text_format: The structured output format type (if applicable)
            active_contexts: List of active tracing contexts
            start_time: Timestamp when the request started
            model: The model identifier

        Yields:
            ResponseStreamEvent objects as they are parsed from the stream
        """
        accumulated_text = ""
        final_event: ResponseCompletedEvent[TextFormatT] | None = None

        try:
            async for event in self._stream_events_from_buffer(
                content_lines, text_format=text_format
            ):
                # Accumulate text deltas for metrics
                if hasattr(event, "delta"):
                    delta = getattr(event, "delta", "")
                    if delta:
                        accumulated_text += delta

                # Track final ResponseCompletedEvent
                if isinstance(event, ResponseCompletedEvent):
                    final_event = event

                yield event

            # Update tracing with success after streaming completes
            if active_contexts and final_event:
                try:
                    await self._update_tracing_success(
                        active_contexts=active_contexts,
                        response=final_event.response,
                        accumulated_text=accumulated_text,
                        start_time=start_time,
                        model=model,
                        text_format=text_format,
                    )
                except Exception as trace_error:
                    # Log error but don't fail the stream
                    logger.error(
                        f"Failed to update tracing with success after streaming: {trace_error}",
                        exc_info=True,
                    )

        except Exception as e:
            # Update tracing with error
            if active_contexts:
                try:
                    generation_config: dict[str, Any] = {"metadata": {}}
                    trace_metadata = self._prepare_trace_metadata(
                        model=model,
                        base_url=self.base_url,
                        generation_config=generation_config,
                    )
                    await self._update_tracing_error(
                        active_contexts=active_contexts,
                        error=e,
                        start_time=start_time,
                        metadata=trace_metadata,
                    )
                except Exception as trace_error:
                    # Log error but don't fail - we're already handling an error
                    logger.error(
                        f"Failed to update tracing with error during streaming: {trace_error}",
                        exc_info=True,
                    )
            # Re-raise the error
            raise

        finally:
            # Cleanup tracing contexts after streaming completes or fails
            if active_contexts:
                try:
                    await self._cleanup_tracing_contexts(active_contexts)
                except Exception as e:
                    # Log error but don't fail
                    logger.error(
                        f"Failed to cleanup tracing contexts after streaming: {e}",
                        exc_info=True,
                    )

    def _prepare_trace_input_data(
        self,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Prepare input data for trace context from request payload.

        Extracts relevant fields like input/messages, model, tools, reasoning settings,
        temperature, top_p, etc. for observability tracking.

        Args:
            request_payload: The request payload dictionary sent to the API

        Returns:
            Dictionary containing structured input data for tracing
        """
        # Extract basic input
        input_data = request_payload.get("input")

        # Extract model
        model = request_payload.get("model")

        # Extract tools information
        tools = request_payload.get("tools", [])
        has_tools = len(tools) > 0
        tools_count = len(tools)

        # Extract structured output information
        text_config = request_payload.get("text", {})
        has_structured_output = (
            text_config.get("type") == "json_schema" if text_config else False
        )

        # Extract reasoning information
        reasoning = request_payload.get("reasoning")
        reasoning_enabled = reasoning is not None
        reasoning_effort = None
        if reasoning and isinstance(reasoning, dict):
            reasoning_effort = reasoning.get("effort")

        # Extract model parameters
        temperature = request_payload.get("temperature")
        top_p = request_payload.get("top_p")
        max_output_tokens = request_payload.get("max_output_tokens")

        # Extract streaming flag
        stream = request_payload.get("stream", False)

        return {
            "input": input_data,
            "model": model,
            "has_tools": has_tools,
            "tools_count": tools_count,
            "has_structured_output": has_structured_output,
            "reasoning_enabled": reasoning_enabled,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "stream": stream,
        }

    def _prepare_trace_metadata(
        self,
        *,
        model: str,
        base_url: str,
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Prepare metadata for trace context.

        Extracts model name, provider, base_url, and merges custom metadata
        from generation_config for observability tracking.

        Args:
            model: The model identifier
            base_url: The API base URL
            generation_config: Configuration dictionary that may contain custom metadata

        Returns:
            Dictionary containing metadata for tracing
        """
        # Determine provider from base_url
        provider = "openai"
        if "openrouter" in base_url.lower():
            provider = "openrouter"

        # Start with base metadata
        metadata = {
            "model": model,
            "provider": provider,
            "base_url": base_url,
        }

        # Merge custom metadata from generation_config if present
        custom_metadata: dict[str, Any] = generation_config.get("metadata", {})
        if custom_metadata:
            metadata.update(custom_metadata)

        return metadata

    def _extract_usage_from_response(
        self,
        response: Response[Any],
    ) -> dict[str, Any] | None:
        """
        Extract token usage information from a Response object.

        Handles missing usage data gracefully and extracts detailed token counts
        including reasoning tokens if available.

        Args:
            response: The Response object from the API

        Returns:
            Dictionary with usage details (input, output, total tokens, reasoning tokens),
            or None if usage data is not available
        """
        if not response.usage:
            return None

        usage = response.usage

        # Extract basic token counts
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        total_tokens = usage.total_tokens

        # Extract reasoning tokens if available
        reasoning_tokens = None
        if usage.output_tokens_details:
            reasoning_tokens = usage.output_tokens_details.reasoning_tokens

        # Build usage dictionary
        usage_dict = {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
            "unit": "TOKENS",
        }

        # Add reasoning tokens if present
        if reasoning_tokens is not None and reasoning_tokens > 0:
            usage_dict["reasoning_tokens"] = reasoning_tokens

        return usage_dict

    async def _calculate_costs(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict[str, Any] | None:
        """
        Calculate cost metrics based on token usage.

        Uses the PricingService to get model pricing and calculates input and output costs.
        Handles unknown models gracefully by returning None.

        Args:
            model: The model identifier
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used

        Returns:
            Dictionary with cost breakdown (input, output, total costs with currency),
            or None if pricing is not available for the model
        """
        try:
            # Get pricing from the pricing service
            input_price_per_million = (
                await self.pricing_service.get_input_price_per_million(
                    model, modality="text"
                )
            )
            output_price_per_million = (
                await self.pricing_service.get_output_price_per_million(
                    model, modality="text"
                )
            )

            # If pricing is not available, return None
            if input_price_per_million is None or output_price_per_million is None:
                logger.debug(f"Pricing not available for model: {model}")
                return None

            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * input_price_per_million
            output_cost = (output_tokens / 1_000_000) * output_price_per_million
            total_cost = input_cost + output_cost

            return {
                "input": input_cost,
                "output": output_cost,
                "total": total_cost,
                "currency": "USD",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except Exception as e:
            # Log error but don't fail the request
            logger.warning(f"Failed to calculate costs for model {model}: {e}")
            return None

    async def _create_tracing_contexts(
        self,
        *,
        model: str,
        request_payload: dict[str, Any],
        generation_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Create trace and generation contexts for all configured OtelClients.

        Iterates through all otel_clients and creates both trace and generation contexts
        for each client. Handles context creation errors gracefully to ensure that
        tracing failures don't impact the main API call.

        Args:
            model: The model identifier
            request_payload: The request payload dictionary sent to the API
            generation_config: Configuration dictionary containing metadata and settings

        Returns:
            List of context dictionaries, each containing:
            - client: The OtelClient instance
            - trace_gen: The trace context generator
            - trace_ctx: The trace context object
            - generation_gen: The generation context generator
            - generation_ctx: The generation context object
        """
        active_contexts: list[dict[str, Any]] = []

        if not self.otel_clients:
            logger.debug(
                "No OTel clients configured, skipping tracing context creation"
            )
            return active_contexts

        logger.debug(
            f"Creating tracing contexts for {len(self.otel_clients)} OTel client(s) with model: {model}"
        )

        # Prepare input data and metadata for tracing
        input_data = self._prepare_trace_input_data(request_payload)
        metadata = self._prepare_trace_metadata(
            model=model,
            base_url=self.base_url,
            generation_config=generation_config,
        )

        # Create contexts for each client
        for client in self.otel_clients:
            client_name = type(client).__name__
            try:
                logger.debug(f"Creating trace context for client: {client_name}")

                # Create trace context
                trace_gen = client.trace_context(
                    name="responder_api_call",
                    input_data=input_data,
                    metadata=metadata,
                )
                trace_ctx = await trace_gen.__anext__()

                logger.debug(f"Trace context created for client: {client_name}")

                # Create generation context
                logger.debug(f"Creating generation context for client: {client_name}")
                generation_gen = client.generation_context(
                    trace_context=trace_ctx,
                    name="response_generation",
                    model=model,
                    provider=metadata.get("provider", "unknown"),
                    input_data=input_data,
                    metadata=metadata,
                )
                generation_ctx = await generation_gen.__anext__()

                logger.debug(f"Generation context created for client: {client_name}")

                # Store contexts
                active_contexts.append(
                    {
                        "client": client,
                        "trace_gen": trace_gen,
                        "trace_ctx": trace_ctx,
                        "generation_gen": generation_gen,
                        "generation_ctx": generation_ctx,
                    }
                )

                logger.debug(
                    f"Successfully created tracing contexts for client: {client_name}"
                )

            except Exception as e:
                # Log error but continue with other clients
                logger.error(
                    f"Failed to create tracing contexts for client {client_name}: {e}",
                    exc_info=True,
                )
                continue

        logger.debug(
            f"Created {len(active_contexts)} tracing context(s) out of {len(self.otel_clients)} client(s)"
        )

        return active_contexts

    async def _update_tracing_success(
        self,
        *,
        active_contexts: list[dict[str, Any]],
        response: Response[Any] | None = None,
        accumulated_text: str = "",
        start_time: datetime,
        model: str,
        text_format: Type[Any] | None = None,
    ) -> None:
        """
        Update all tracing contexts with successful response data.

        Extracts usage and calculates costs, then updates both generation and trace
        contexts with the results. Handles structured output parsing and uses
        non-blocking operations for non-critical updates.

        Args:
            active_contexts: List of active tracing contexts
            response: The Response object from the API (if available)
            accumulated_text: Accumulated text from streaming (for structured output parsing)
            start_time: Timestamp when the request started
            model: The model identifier
            text_format: The structured output format type (if applicable)
        """
        if not active_contexts:
            logger.debug("No active tracing contexts to update with success")
            return

        logger.debug(
            f"Updating {len(active_contexts)} tracing context(s) with successful response"
        )

        # Calculate latency
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()

        logger.debug(f"Request latency: {latency:.3f}s")

        # Extract usage from response
        usage_details = None
        if response:
            usage_details = self._extract_usage_from_response(response)
            if usage_details:
                logger.debug(
                    f"Extracted usage: {usage_details.get('input', 0)} input tokens, {usage_details.get('output', 0)} output tokens"
                )
            else:
                logger.debug("No usage details available in response")

        # Calculate costs if usage is available
        cost_details = None
        if usage_details:
            cost_details = await self._calculate_costs(
                model=model,
                input_tokens=usage_details.get("input", 0),
                output_tokens=usage_details.get("output", 0),
            )
            if cost_details:
                logger.debug(
                    f"Calculated costs: ${cost_details.get('total', 0):.6f} (input: ${cost_details.get('input', 0):.6f}, output: ${cost_details.get('output', 0):.6f})"
                )
            else:
                logger.debug(f"Cost calculation not available for model: {model}")

        # Prepare output data
        output_data: dict[str, Any] = {}

        # Add response data if available
        if response:
            output_data["response_id"] = response.id
            if response.status:
                output_data["status"] = (
                    response.status.value
                    if hasattr(response.status, "value")
                    else response.status
                )

            # Add output text if available
            if response.output_text:
                output_data["output_text"] = response.output_text

            # Handle structured output parsing
            if text_format and issubclass(text_format, BaseModel):
                # Try to get parsed output from response
                parsed_output = None
                if response.output_parsed:
                    parsed_output = response.output_parsed
                elif accumulated_text:
                    # Try to parse accumulated text
                    try:
                        parsed_data = orjson.loads(accumulated_text)
                        parsed_output = text_format.model_validate(parsed_data)
                    except Exception as e:
                        logger.debug(f"Failed to parse accumulated text: {e}")

                if parsed_output:
                    output_data["parsed_output"] = parsed_output.model_dump()

        # Add latency
        output_data["latency_seconds"] = latency

        # Prepare metadata with costs and usage
        metadata: dict[str, Any] = {
            "latency_seconds": latency,
            "model": model,
        }

        if usage_details:
            metadata["usage"] = usage_details

        if cost_details:
            metadata["cost"] = cost_details

        # Update contexts for each client
        for context_dict in active_contexts:
            client_name = type(context_dict.get("client", "unknown")).__name__
            try:
                client = context_dict["client"]
                generation_ctx = context_dict["generation_ctx"]
                trace_ctx = context_dict["trace_ctx"]

                logger.debug(f"Updating tracing contexts for client: {client_name}")

                # Update generation context
                if generation_ctx:
                    try:
                        logger.debug(
                            f"Updating generation context for client: {client_name}"
                        )
                        await client.update_generation(
                            generation_ctx,
                            output_data=output_data,
                            usage_details=usage_details,
                            cost_details=cost_details,
                            metadata=metadata,
                            end_time=end_time,
                        )
                        logger.debug(
                            f"Successfully updated generation context for client: {client_name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to update generation context for client {client_name}: {e}",
                            exc_info=True,
                        )

                # Update trace context
                if trace_ctx:
                    try:
                        logger.debug(
                            f"Updating trace context for client: {client_name}"
                        )
                        await client.update_trace(
                            trace_ctx,
                            output_data=output_data,
                            success=True,
                            metadata=metadata,
                            end_time=end_time,
                        )
                        logger.debug(
                            f"Successfully updated trace context for client: {client_name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to update trace context for client {client_name}: {e}",
                            exc_info=True,
                        )

            except Exception as e:
                # Log error but continue with other clients
                logger.error(
                    f"Failed to update tracing for client {client_name}: {e}",
                    exc_info=True,
                )
                continue

        logger.debug(
            f"Completed updating {len(active_contexts)} tracing context(s) with success"
        )

    async def _update_tracing_error(
        self,
        *,
        active_contexts: list[dict[str, Any]],
        error: Exception,
        start_time: datetime,
        metadata: dict[str, Any],
    ) -> None:
        """
        Update all tracing contexts with error information.

        Records error details in both generation and trace contexts. Handles errors
        in error handling gracefully to ensure that tracing failures don't compound
        the original error.

        Args:
            active_contexts: List of active tracing contexts
            error: The exception that occurred
            start_time: Timestamp when the request started
            metadata: Additional metadata to include in the error trace
        """
        if not active_contexts:
            logger.debug("No active tracing contexts to update with error")
            return

        logger.debug(
            f"Updating {len(active_contexts)} tracing context(s) with error: {type(error).__name__}"
        )

        # Calculate latency until error
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()

        logger.debug(f"Latency until error: {latency:.3f}s")

        # Prepare error metadata
        error_metadata = dict(metadata)
        error_metadata.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "latency_until_error": latency,
                "success": False,
            }
        )

        # Update contexts for each client
        for context_dict in active_contexts:
            client_name = type(context_dict.get("client", "unknown")).__name__
            try:
                client = context_dict["client"]
                generation_ctx = context_dict.get("generation_ctx")
                trace_ctx = context_dict.get("trace_ctx")

                logger.debug(f"Recording error in tracing for client: {client_name}")

                # Use the client's handle_error method if available
                try:
                    await client.handle_error(
                        trace_context=trace_ctx,
                        generation_context=generation_ctx,
                        error=error,
                        start_time=start_time,
                        metadata=error_metadata,
                    )
                    logger.debug(
                        f"Successfully recorded error in tracing for client: {client_name}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to record error in tracing for client {client_name}: {e}",
                        exc_info=True,
                    )

            except Exception as e:
                # Log error but don't re-raise - we're already handling an error
                logger.error(
                    f"Failed to update error tracing for client {client_name}: {e}",
                    exc_info=True,
                )
                continue

        logger.debug(
            f"Completed updating {len(active_contexts)} tracing context(s) with error"
        )

    async def _cleanup_tracing_contexts(
        self,
        active_contexts: list[dict[str, Any]],
    ) -> None:
        """
        Cleanup all tracing contexts by closing generators.

        Properly closes all trace and generation context generators to ensure
        resources are released. Handles cleanup errors gracefully to ensure
        all contexts are attempted even if some fail.

        Args:
            active_contexts: List of active tracing contexts to cleanup
        """
        if not active_contexts:
            logger.debug("No active tracing contexts to cleanup")
            return

        logger.debug(f"Cleaning up {len(active_contexts)} tracing context(s)")

        for context_dict in active_contexts:
            client_name = type(context_dict.get("client", "unknown")).__name__
            try:
                logger.debug(f"Cleaning up tracing contexts for client: {client_name}")

                # Close generation context generator
                generation_gen = context_dict.get("generation_gen")
                if generation_gen:
                    try:
                        logger.debug(
                            f"Closing generation context generator for client: {client_name}"
                        )
                        await generation_gen.aclose()
                        logger.debug(
                            f"Closed generation context generator for client: {client_name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to close generation context for client {client_name}: {e}",
                            exc_info=True,
                        )

                # Close trace context generator
                trace_gen = context_dict.get("trace_gen")
                if trace_gen:
                    try:
                        logger.debug(
                            f"Closing trace context generator for client: {client_name}"
                        )
                        await trace_gen.aclose()
                        logger.debug(
                            f"Closed trace context generator for client: {client_name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to close trace context for client {client_name}: {e}",
                            exc_info=True,
                        )

            except Exception as e:
                # Log error but continue with other contexts
                logger.error(
                    f"Failed to cleanup tracing context for client {client_name}: {e}",
                    exc_info=True,
                )
                continue

        logger.debug(f"Completed cleanup of {len(active_contexts)} tracing context(s)")

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
