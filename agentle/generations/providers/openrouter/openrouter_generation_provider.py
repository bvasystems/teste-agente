# Placeholder for OpenRouterGenerationProvider implementation
"""
OpenRouter provider implementation for the Agentle framework.

This module provides the OpenRouterGenerationProvider class, which enables Agentle
to interact with multiple AI models through OpenRouter's unified API. OpenRouter
acts as a gateway to various providers including OpenAI, Anthropic, Google, and
many others, with automatic fallback and routing capabilities.

The provider supports:
- Multiple model routing with automatic fallbacks
- API key authentication
- Message-based interactions with multimodal content (images, PDFs, audio)
- Structured output parsing via response schemas
- Tool/function calling
- Provider preferences and routing configuration
- Custom HTTP client configuration
- Usage statistics tracking

This implementation transforms Agentle's unified message format into OpenRouter's
request format and adapts responses back into Agentle's Generation objects.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from os import getenv
from typing import TYPE_CHECKING, Any, cast, override

import httpx

from agentle.generations.json.json_schema_builder import JsonSchemaBuilder
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.decorators.model_kind_mapper import (
    override_model_kind,
)
from agentle.generations.providers.openrouter._adapters.agentle_message_to_openrouter_message_adapter import (
    AgentleMessageToOpenRouterMessageAdapter,
)
from agentle.generations.providers.openrouter._adapters.agentle_tool_to_openrouter_tool_adapter import (
    AgentleToolToOpenRouterToolAdapter,
)
from agentle.generations.providers.openrouter._adapters.openrouter_response_to_generation_adapter import (
    OpenRouterResponseToGenerationAdapter,
)
from agentle.generations.providers.openrouter._types import (
    OpenRouterRequest,
    OpenRouterResponse,
    OpenRouterProviderPreferences,
    OpenRouterResponseFormat,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing import observe
from agentle.utils.raise_error import raise_error

if TYPE_CHECKING:
    from agentle.generations.tracing.otel_client import OtelClient


logger = logging.getLogger(__name__)
type WithoutStructuredOutput = None


class OpenRouterGenerationProvider(GenerationProvider):
    """
    Provider implementation for OpenRouter services.

    This class implements the GenerationProvider interface for OpenRouter's unified API,
    allowing seamless integration with multiple AI providers through a single interface.
    It handles conversion of Agentle messages to OpenRouter format, manages API
    communication, and processes responses back into the standardized Agentle format.

    The provider supports API key authentication, custom HTTP configuration, provider
    routing preferences, multimodal inputs, tool calling, and structured output parsing.

    Attributes:
        otel_clients: Optional clients for observability and tracing.
        api_key: API key for authentication with OpenRouter.
        base_url: Optional custom base URL for the OpenRouter API.
        timeout: Optional timeout for API requests.
        max_retries: Maximum number of retries for failed requests.
        default_headers: Optional default HTTP headers for requests.
        http_client: Optional custom HTTP client for requests.
        provider_preferences: Optional provider routing preferences.
        message_adapter: Adapter to convert Agentle messages to OpenRouter format.
        tool_adapter: Adapter to convert Agentle tools to OpenRouter format.
    """

    otel_clients: Sequence[OtelClient]
    api_key: str
    base_url: str
    max_retries: int
    default_headers: Mapping[str, str] | None
    http_client: httpx.AsyncClient | None
    provider_preferences: OpenRouterProviderPreferences | None
    message_adapter: AgentleMessageToOpenRouterMessageAdapter
    tool_adapter: AgentleToolToOpenRouterToolAdapter

    def __init__(
        self,
        *,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | OtelClient | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        provider_preferences: OpenRouterProviderPreferences | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ):
        """
        Initialize the OpenRouter Generation Provider.

        Args:
            api_key: API key for authentication with OpenRouter.
            otel_clients: Optional clients for observability and tracing.
            provider_id: Optional custom provider identifier.
            base_url: Base URL for the OpenRouter API.
            max_retries: Maximum number of retries for failed requests.
            default_headers: Optional default HTTP headers for requests.
            http_client: Optional custom HTTP client for requests.
            provider_preferences: Optional provider routing preferences.
            message_adapter: Optional adapter to convert Agentle messages.
            tool_adapter: Optional adapter to convert Agentle tools.
        """
        super().__init__(otel_clients=otel_clients, provider_id=provider_id)
        self.api_key = (
            api_key
            or getenv("OPENROUTER_API_KEY")
            or raise_error(
                "any of api_key of OPENROUTER_API_KEY must be set to use OpenRouter provider."
            )
        )
        self.base_url = base_url
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.http_client = http_client
        self.provider_preferences = provider_preferences
        self.message_adapter = (
            message_adapter or AgentleMessageToOpenRouterMessageAdapter()
        )
        self.tool_adapter = tool_adapter or AgentleToolToOpenRouterToolAdapter()

    @property
    @override
    def organization(self) -> str:
        """
        Get the provider organization identifier.

        Returns:
            str: The organization identifier, which is "openrouter" for this provider.
        """
        return "openrouter"

    @property
    @override
    def default_model(self) -> str:
        """
        The default model to use for generation.

        Returns:
            str: Default model identifier for OpenRouter.
        """
        return "anthropic/claude-sonnet-4.5"

    @override
    async def stream_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> AsyncGenerator[Generation[WithoutStructuredOutput], None]:
        """
        Stream generations asynchronously (not yet implemented).

        Args:
            model: The model identifier or kind to use.
            messages: The sequence of messages to send.
            response_schema: Optional schema for structured output.
            generation_config: Optional generation configuration.
            tools: Optional tools for function calling.

        Yields:
            Generation objects as they are produced.

        Raises:
            NotImplementedError: Streaming is not yet implemented.
        """
        raise NotImplementedError("Streaming is not yet implemented for OpenRouter.")
        if False:  # pragma: no cover
            yield cast(Generation[WithoutStructuredOutput], None)

    @override
    @observe
    @override_model_kind
    async def generate_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation asynchronously using OpenRouter.

        This method handles the conversion of Agentle messages to OpenRouter's format,
        sends the request to OpenRouter's API, and processes the response into Agentle's
        standardized Generation format.

        Args:
            model: The model identifier to use (or list of models for fallback).
            messages: A sequence of Agentle messages to send to the model.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling.

        Returns:
            Generation[T]: An Agentle Generation object containing the model's response,
                potentially with structured output if a response_schema was provided.
        """
        _generation_config = self._normalize_generation_config(generation_config)

        # Convert messages
        openrouter_messages = [
            self.message_adapter.adapt(message) for message in messages
        ]

        # Convert tools if provided
        openrouter_tools = (
            [self.tool_adapter.adapt(tool) for tool in tools] if tools else None
        )

        # Build response format for structured outputs
        response_format = None
        if response_schema:
            response_format = OpenRouterResponseFormat(
                type="json_schema",
                json_schema={
                    "name": "response_schema",
                    "strict": True,
                    "schema": JsonSchemaBuilder(
                        cast(type[Any], response_schema),
                        use_defs_instead_of_definitions=True,
                        clean_output=True,
                        strict_mode=True,
                    ).build(dereference=True),
                },
            )

        # Build the request
        request_body: OpenRouterRequest = {
            "model": model or self.default_model,
            "messages": openrouter_messages,
        }

        # Add optional parameters
        if openrouter_tools:
            request_body["tools"] = openrouter_tools

        if response_format:
            request_body["response_format"] = response_format

        if self.provider_preferences:
            request_body["provider"] = self.provider_preferences

        # Add generation config parameters
        if _generation_config.temperature is not None:
            request_body["temperature"] = _generation_config.temperature
        if _generation_config.max_output_tokens is not None:
            request_body["max_tokens"] = _generation_config.max_output_tokens
        if _generation_config.top_p is not None:
            request_body["top_p"] = _generation_config.top_p

        # Make the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **(self.default_headers or {}),
        }

        # Configure timeout for httpx client
        # Use the generation config timeout or default to 300 seconds (5 minutes) for vision/PDF tasks
        timeout_seconds = _generation_config.timeout_in_seconds or 300.0
        client = self.http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=timeout_seconds,
                connect=30.0,  # Keep connection timeout reasonable
            )
        )
        url = f"{self.base_url}/chat/completions"

        try:
            async with asyncio.timeout(_generation_config.timeout_in_seconds):
                response = await client.post(
                    url,
                    json=request_body,
                    headers=headers,
                )
                response.raise_for_status()
                openrouter_response: OpenRouterResponse = response.json()

        except asyncio.TimeoutError as e:
            e.add_note(
                f"Content generation timed out after {_generation_config.timeout_in_seconds}s"
            )
            raise
        finally:
            if not self.http_client:
                await client.aclose()

        # Convert response to Generation
        return OpenRouterResponseToGenerationAdapter[T](
            response_schema=response_schema,
        ).adapt(openrouter_response)

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        """
        Map a ModelKind to a specific OpenRouter model identifier.

        Args:
            model_kind: The model kind category to map.

        Returns:
            str: The corresponding OpenRouter model identifier.
        """
        mapping: Mapping[ModelKind, str] = {
            "category_nano": "google/gemini-2.5-flash-lite-preview-09-2025",
            "category_mini": "anthropic/claude-3.5-haiku",
            "category_standard": "anthropic/claude-sonnet-4.5",
            "category_pro": "anthropic/claude-opus-4.1",
            "category_flagship": "anthropic/claude-opus-4.1",
            "category_reasoning": "deepseek/deepseek-v3.2-exp",
            "category_vision": "google/gemini-2.5-flash-preview-09-2025",
            "category_coding": "deepseek/deepseek-v3.2-exp",
            "category_instruct": "anthropic/claude-sonnet-4.5",
            # Experimental variants
            "category_nano_experimental": "google/gemini-2.5-flash-lite-preview-09-2025",
            "category_mini_experimental": "anthropic/claude-3.5-haiku",
            "category_standard_experimental": "anthropic/claude-sonnet-4.5",
            "category_pro_experimental": "anthropic/claude-opus-4.1",
            "category_flagship_experimental": "anthropic/claude-opus-4.1",
            "category_reasoning_experimental": "deepseek/deepseek-v3.2-exp",
            "category_vision_experimental": "google/gemini-2.5-flash-preview-09-2025",
            "category_coding_experimental": "deepseek/deepseek-v3.2-exp",
            "category_instruct_experimental": "anthropic/claude-sonnet-4.5",
        }

        return mapping[model_kind]

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for input/prompt tokens.

        Note: OpenRouter pricing varies by provider and model. This provides
        approximate pricing for common models. For exact pricing, check
        https://openrouter.ai/models

        Args:
            model: The model identifier.
            estimate_tokens: Optional estimate of token count (not used).

        Returns:
            float: The approximate price per million input tokens.
        """
        # Sample pricing for common models
        # Actual pricing may vary and should be checked on OpenRouter
        input_prices: Mapping[str, float] = {
            "anthropic/claude-opus-4.1": 15.0,
            "anthropic/claude-sonnet-4.5": 3.0,
            "anthropic/claude-3.5-haiku": 1.0,
            "google/gemini-2.5-flash-preview-09-2025": 0.075,
            "google/gemini-2.5-flash-lite-preview-09-2025": 0.0375,
            "deepseek/deepseek-v3.2-exp": 0.27,
            "openai/gpt-4o": 2.5,
            "openai/gpt-4o-mini": 0.15,
        }

        price = input_prices.get(model)
        if price is None:
            logger.warning(
                f"OpenRouter model {model} not found in pricing table. "
                + "Returning 0.0. Check https://openrouter.ai/models for actual pricing."
            )
            return 0.0
        return price

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for output/completion tokens.

        Note: OpenRouter pricing varies by provider and model. This provides
        approximate pricing for common models. For exact pricing, check
        https://openrouter.ai/models

        Args:
            model: The model identifier.
            estimate_tokens: Optional estimate of token count (not used).

        Returns:
            float: The approximate price per million output tokens.
        """
        # Sample pricing for common models
        # Actual pricing may vary and should be checked on OpenRouter
        output_prices: Mapping[str, float] = {
            "anthropic/claude-opus-4.1": 75.0,
            "anthropic/claude-sonnet-4.5": 15.0,
            "anthropic/claude-3.5-haiku": 5.0,
            "google/gemini-2.5-flash-preview-09-2025": 0.3,
            "google/gemini-2.5-flash-lite-preview-09-2025": 0.15,
            "deepseek/deepseek-v3.2-exp": 1.10,
            "openai/gpt-4o": 10.0,
            "openai/gpt-4o-mini": 0.6,
        }

        price = output_prices.get(model)
        if price is None:
            logger.warning(
                f"OpenRouter model {model} not found in pricing table. "
                + "Returning 0.0. Check https://openrouter.ai/models for actual pricing."
            )
            return 0.0
        return price
