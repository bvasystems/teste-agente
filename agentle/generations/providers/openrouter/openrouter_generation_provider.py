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
- Streaming responses with Server-Sent Events (SSE)
- Provider preferences and routing configuration (ZDR, sort, max_price, only, ignore)
- Message transforms (middle-out context compression)
- Plugins (file parser for PDFs, web search)
- Prompt caching (cache_control on text parts)
- Reasoning output (for models that support it)
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
from typing import TYPE_CHECKING, Any, Literal, cast, override

import httpx

from agentle.generations.json.json_schema_builder import JsonSchemaBuilder
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.message_parts.text import TextPart
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
    OpenRouterPlugin,
    OpenRouterMaxPrice,
    OpenRouterWebSearchPlugin,
    OpenRouterFileParserPlugin,
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
    routing preferences, multimodal inputs, tool calling, structured output parsing,
    streaming, message transforms, plugins, prompt caching, and reasoning output.

    Attributes:
        otel_clients: Optional clients for observability and tracing.
        api_key: API key for authentication with OpenRouter.
        base_url: Optional custom base URL for the OpenRouter API.
        timeout: Optional timeout for API requests.
        max_retries: Maximum number of retries for failed requests.
        default_headers: Optional default HTTP headers for requests.
        http_client: Optional custom HTTP client for requests.
        provider_preferences: Optional provider routing preferences (ZDR, sort, max_price, etc).
        plugins: Optional plugins configuration (file parser, web search).
        transforms: Optional transforms (e.g., middle-out context compression).
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
    plugins: Sequence[OpenRouterPlugin] | None
    transforms: Sequence[Literal["middle-out"]] | None
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
        plugins: Sequence[OpenRouterPlugin] | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
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
            plugins: Optional plugins configuration (e.g., file parser, web search).
            transforms: Optional transforms (e.g., ["middle-out"] for context compression).
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
        self.plugins = plugins
        self.transforms = transforms
        self.message_adapter = (
            message_adapter or AgentleMessageToOpenRouterMessageAdapter()
        )
        self.tool_adapter = tool_adapter or AgentleToolToOpenRouterToolAdapter()

    # ==================== Factory Methods ====================

    @classmethod
    def with_cheapest_routing(
        cls,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        plugins: Sequence[OpenRouterPlugin] | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider configured to always use the cheapest available provider.
        
        Equivalent to setting provider_preferences with sort="price".
        
        Args:
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            plugins: Optional plugins (web search, file parser)
            transforms: Optional transforms (middle-out compression)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
        """
        preferences: OpenRouterProviderPreferences = {"sort": "price"}
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=preferences,
            plugins=plugins,
            transforms=transforms,
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    @classmethod
    def with_fastest_routing(
        cls,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        plugins: Sequence[OpenRouterPlugin] | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider configured to prioritize highest throughput (Nitro mode).
        
        Equivalent to setting provider_preferences with sort="throughput".
        
        Args:
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            plugins: Optional plugins (web search, file parser)
            transforms: Optional transforms (middle-out compression)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
        """
        preferences: OpenRouterProviderPreferences = {"sort": "throughput"}
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=preferences,
            plugins=plugins,
            transforms=transforms,
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    @classmethod
    def with_lowest_latency(
        cls,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        plugins: Sequence[OpenRouterPlugin] | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider configured to prioritize lowest latency.
        
        Equivalent to setting provider_preferences with sort="latency".
        
        Args:
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            plugins: Optional plugins (web search, file parser)
            transforms: Optional transforms (middle-out compression)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
        """
        preferences: OpenRouterProviderPreferences = {"sort": "latency"}
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=preferences,
            plugins=plugins,
            transforms=transforms,
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    @classmethod
    def with_zdr_enforcement(
        cls,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        plugins: Sequence[OpenRouterPlugin] | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider with Zero Data Retention enforcement.
        
        Only routes to endpoints that do not retain prompts.
        
        Args:
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            plugins: Optional plugins (web search, file parser)
            transforms: Optional transforms (middle-out compression)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
        """
        preferences: OpenRouterProviderPreferences = {"zdr": True}
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=preferences,
            plugins=plugins,
            transforms=transforms,
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    @classmethod
    def with_privacy_mode(
        cls,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        plugins: Sequence[OpenRouterPlugin] | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider that only uses providers which don't collect user data.
        
        Combines ZDR enforcement with data_collection="deny".
        
        Args:
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            plugins: Optional plugins (web search, file parser)
            transforms: Optional transforms (middle-out compression)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
        """
        preferences: OpenRouterProviderPreferences = {
            "zdr": True,
            "data_collection": "deny",
        }
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=preferences,
            plugins=plugins,
            transforms=transforms,
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    @classmethod
    def with_specific_providers(
        cls,
        providers: Sequence[str],
        allow_fallbacks: bool = True,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        plugins: Sequence[OpenRouterPlugin] | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider that tries specific providers in order.
        
        Args:
            providers: List of provider slugs to try (e.g., ["anthropic", "openai"])
            allow_fallbacks: Whether to allow other providers if specified ones fail
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            plugins: Optional plugins (web search, file parser)
            transforms: Optional transforms (middle-out compression)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
            
        Example:
            >>> provider = OpenRouterGenerationProvider.with_specific_providers(
            ...     providers=["anthropic", "openai"],
            ...     allow_fallbacks=False
            ... )
        """
        preferences: OpenRouterProviderPreferences = {
            "order": providers,
            "allow_fallbacks": allow_fallbacks,
        }
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=preferences,
            plugins=plugins,
            transforms=transforms,
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    @classmethod
    def with_web_search(
        cls,
        api_key: str | None = None,
        max_results: int = 5,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        provider_preferences: OpenRouterProviderPreferences | None = None,
        transforms: Sequence[Literal["middle-out"]] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider with web search plugin enabled.
        
        Args:
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            max_results: Maximum number of search results to return
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            provider_preferences: Optional provider routing preferences
            transforms: Optional transforms (middle-out compression)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
        """
        plugins: list[OpenRouterPlugin] = [
            {"id": "web", "max_results": max_results}  # type: ignore
        ]
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=provider_preferences,
            plugins=plugins,
            transforms=transforms,
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    @classmethod
    def with_context_compression(
        cls,
        api_key: str | None = None,
        otel_clients: Sequence[OtelClient] | None = None,
        provider_id: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        provider_preferences: OpenRouterProviderPreferences | None = None,
        plugins: Sequence[OpenRouterPlugin] | None = None,
        message_adapter: AgentleMessageToOpenRouterMessageAdapter | None = None,
        tool_adapter: AgentleToolToOpenRouterToolAdapter | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Create provider with middle-out context compression enabled.
        
        Useful for long contexts that exceed model limits.
        
        Args:
            api_key: Optional API key (uses OPENROUTER_API_KEY env var if not provided)
            otel_clients: Optional OpenTelemetry clients for tracing
            provider_id: Optional provider identifier for tracing
            base_url: Base URL for OpenRouter API
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in all requests
            http_client: Optional httpx client to reuse
            provider_preferences: Optional provider routing preferences
            plugins: Optional plugins (web search, file parser)
            message_adapter: Optional custom message adapter
            tool_adapter: Optional custom tool adapter
            
        Returns:
            Configured OpenRouterGenerationProvider instance
        """
        return cls(
            api_key=api_key,
            otel_clients=otel_clients,
            provider_id=provider_id,
            base_url=base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
            provider_preferences=provider_preferences,
            plugins=plugins,
            transforms=["middle-out"],
            message_adapter=message_adapter,
            tool_adapter=tool_adapter,
        )

    # ==================== Builder-Style Methods ====================

    def order_by_cheapest(self) -> "OpenRouterGenerationProvider":
        """Configure to always use the cheapest provider (floor pricing).
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["sort"] = "price"
        return self

    def order_by_fastest(self) -> "OpenRouterGenerationProvider":
        """Configure to prioritize highest throughput (Nitro mode).
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["sort"] = "throughput"
        return self

    def order_by_lowest_latency(self) -> "OpenRouterGenerationProvider":
        """Configure to prioritize lowest latency.
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["sort"] = "latency"
        return self

    def enable_zdr(self) -> "OpenRouterGenerationProvider":
        """Enable Zero Data Retention enforcement.
        
        Only routes to endpoints that do not retain prompts.
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["zdr"] = True
        return self

    def deny_data_collection(self) -> "OpenRouterGenerationProvider":
        """Only use providers that don't collect user data.
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["data_collection"] = "deny"
        return self

    def allow_data_collection(self) -> "OpenRouterGenerationProvider":
        """Allow providers that may collect user data (default).
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["data_collection"] = "allow"
        return self

    def set_provider_order(self, providers: Sequence[str]) -> "OpenRouterGenerationProvider":
        """Set the order of providers to try.
        
        Args:
            providers: List of provider slugs (e.g., ["anthropic", "openai"])
            
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["order"] = providers
        return self

    def allow_only_providers(self, providers: Sequence[str]) -> "OpenRouterGenerationProvider":
        """Only allow specific providers for requests.
        
        Args:
            providers: List of provider slugs to allow
            
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["only"] = providers
        return self

    def ignore_providers(self, providers: Sequence[str]) -> "OpenRouterGenerationProvider":
        """Ignore specific providers for requests.
        
        Args:
            providers: List of provider slugs to ignore
            
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["ignore"] = providers
        return self

    def disable_fallbacks(self) -> "OpenRouterGenerationProvider":
        """Disable fallback providers.
        
        Request will fail if primary provider is unavailable.
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["allow_fallbacks"] = False
        return self

    def enable_fallbacks(self) -> "OpenRouterGenerationProvider":
        """Enable fallback providers (default).
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["allow_fallbacks"] = True
        return self

    def require_all_parameters(self) -> "OpenRouterGenerationProvider":
        """Only use providers that support all request parameters.
        
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["require_parameters"] = True
        return self

    def set_max_price(
        self,
        prompt: float | None = None,
        completion: float | None = None,
        request: float | None = None,
        image: float | None = None,
    ) -> "OpenRouterGenerationProvider":
        """Set maximum pricing constraints.
        
        Args:
            prompt: Max price per million prompt tokens
            completion: Max price per million completion tokens
            request: Max price per request
            image: Max price per image
            
        Returns:
            Self for method chaining
            
        Example:
            >>> provider.set_max_price(prompt=1.0, completion=2.0)
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        
        max_price: OpenRouterMaxPrice = {}
        if prompt is not None:
            max_price["prompt"] = prompt
        if completion is not None:
            max_price["completion"] = completion
        if request is not None:
            max_price["request"] = request
        if image is not None:
            max_price["image"] = image
            
        self.provider_preferences["max_price"] = max_price
        return self

    def filter_by_quantization(self, quantizations: Sequence[str]) -> "OpenRouterGenerationProvider":
        """Filter providers by quantization levels.
        
        Args:
            quantizations: List of quantization levels (e.g., ["fp8", "int4"])
                Valid values: int4, int8, fp4, fp6, fp8, fp16, bf16, fp32, unknown
            
        Returns:
            Self for method chaining
        """
        if self.provider_preferences is None:
            self.provider_preferences = {}
        self.provider_preferences["quantizations"] = quantizations
        return self

    def enable_web_search(self, max_results: int = 5, engine: Literal["native", "exa"] = "exa") -> "OpenRouterGenerationProvider":
        """Enable web search plugin.
        
        Args:
            max_results: Maximum number of search results
            engine: Search engine to use ("native" or "exa")
            
        Returns:
            Self for method chaining
        """
        plugin: OpenRouterWebSearchPlugin = {
            "id": "web",
            "max_results": max_results,
            "engine": engine,
        }
        if self.plugins is None:
            self.plugins = [plugin]  # type: ignore
        else:
            # Remove existing web search plugin if any
            self.plugins = [p for p in self.plugins if p.get("id") != "web"]  # type: ignore
            self.plugins.append(plugin)  # type: ignore
        return self

    def enable_pdf_parsing(
        self,
        engine: Literal["pdf-text", "mistral-ocr", "native"] = "native"
    ) -> "OpenRouterGenerationProvider":
        """Enable PDF parsing plugin.
        
        Args:
            engine: PDF parsing engine to use
            
        Returns:
            Self for method chaining
        """
        plugin: OpenRouterFileParserPlugin = {
            "id": "file-parser",
            "pdf": {"engine": engine},
        }
        if self.plugins is None:
            self.plugins = [plugin]  # type: ignore
        else:
            # Remove existing file parser plugin if any
            self.plugins = [p for p in self.plugins if p.get("id") != "file-parser"]  # type: ignore
            self.plugins.append(plugin)  # type: ignore
        return self

    def enable_context_compression(self) -> "OpenRouterGenerationProvider":
        """Enable middle-out context compression.
        
        Useful for long contexts that exceed model limits.
        
        Returns:
            Self for method chaining
        """
        if self.transforms is None:
            self.transforms = ["middle-out"]
        elif "middle-out" not in self.transforms:
            self.transforms = list(self.transforms) + ["middle-out"]
        return self

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
        Stream generations asynchronously from OpenRouter.

        This method streams responses from OpenRouter's API using Server-Sent Events (SSE).
        Each chunk is converted to a Generation object and yielded as it arrives.

        Note: When response_schema is provided, the model is instructed via system prompt
        to output JSON matching the schema. The JSON is streamed naturally and can be
        parsed from the final accumulated content.

        Args:
            model: The model identifier or kind to use.
            messages: The sequence of messages to send.
            response_schema: Optional schema for structured output (via prompt instruction).
            generation_config: Optional generation configuration.
            tools: Optional tools for function calling.

        Yields:
            Generation objects as they are produced.
        """
        from textwrap import dedent
        from agentle.generations.providers.openrouter._adapters.openrouter_stream_to_generation_adapter import (
            OpenRouterStreamToGenerationAdapter,
        )
        from agentle.utils.describe_model_for_llm import describe_model_for_llm

        _generation_config = self._normalize_generation_config(generation_config)

        # Handle structured output via system prompt instruction
        messages_list = list(messages)
        if response_schema:
            model_description = describe_model_for_llm(response_schema)  # type: ignore[reportArgumentType]
            json_instruction = "Your Output must be a valid JSON string. Do not include any other text. You must provide an answer following the following json structure:"
            conditional_prefix = (
                "If, and only if, not calling any tools, " if tools else ""
            )

            instruction_text = (
                f"{conditional_prefix}{json_instruction}\n{model_description}"
            )

            # Check if first message is a DeveloperMessage
            if messages_list and isinstance(messages_list[0], DeveloperMessage):
                # Append to existing system instruction
                existing_instruction = messages_list[0].text
                messages_list[0] = DeveloperMessage(
                    parts=[TextPart(text=existing_instruction + dedent(f"""\n\n{instruction_text}"""))]
                )
            else:
                # Prepend new DeveloperMessage
                messages_list.insert(
                    0,
                    DeveloperMessage(
                        parts=[TextPart(text=dedent(f"""You are a helpful assistant. {instruction_text}"""))]
                    )
                )

        # Convert messages
        openrouter_messages = [
            self.message_adapter.adapt(message) for message in messages_list
        ]

        # Convert tools if provided
        openrouter_tools = (
            [self.tool_adapter.adapt(tool) for tool in tools] if tools else None
        )

        # Build the request
        request_body: OpenRouterRequest = {
            "model": model or self.default_model,
            "messages": openrouter_messages,
            "stream": True,
        }

        # Add optional parameters
        if openrouter_tools:
            request_body["tools"] = openrouter_tools

        if self.provider_preferences:
            request_body["provider"] = self.provider_preferences

        # Add generation config parameters
        if _generation_config.temperature is not None:
            request_body["temperature"] = _generation_config.temperature
        if _generation_config.max_output_tokens is not None:
            request_body["max_tokens"] = _generation_config.max_output_tokens
        if _generation_config.top_p is not None:
            request_body["top_p"] = _generation_config.top_p

        # Add plugins if configured
        if self.plugins:
            request_body["plugins"] = self.plugins

        # Add transforms if configured
        if self.transforms:
            request_body["transforms"] = self.transforms  # type: ignore

        # Make the streaming API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **(self.default_headers or {}),
        }

        timeout_seconds = _generation_config.timeout_in_seconds or 300.0
        client = self.http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=timeout_seconds,
                connect=30.0,
            )
        )
        url = f"{self.base_url}/chat/completions"

        try:
            async with asyncio.timeout(_generation_config.timeout_in_seconds):
                async with client.stream(
                    "POST",
                    url,
                    json=request_body,
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    # Create async generator from response content
                    async def content_generator() -> AsyncGenerator[bytes, None]:
                        async for chunk in response.aiter_bytes():
                            yield chunk

                    # Use the streaming adapter to process the response
                    adapter = OpenRouterStreamToGenerationAdapter[T](
                        response_schema=response_schema,  # Pass schema for dynamic parsing 
                        model=model or self.default_model,
                    )

                    async for generation in adapter.adapt(content_generator()):
                        yield generation

        except asyncio.TimeoutError as e:
            e.add_note(
                f"Streaming timed out after {_generation_config.timeout_in_seconds}s"
            )
            raise
        finally:
            if not self.http_client:
                await client.aclose()

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
                        cast(type[Any], response_schema),  # pyright: ignore[reportGeneralTypeIssues]
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

        # Add plugins if configured
        if self.plugins:
            request_body["plugins"] = self.plugins

        # Add transforms if configured
        if self.transforms:
            request_body["transforms"] = self.transforms  # type: ignore

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
