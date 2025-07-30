from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Sequence, override

import httpx

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.decorators import override_model_kind
from agentle.generations.providers.openai.adapters.agentle_message_to_openai_message_adapter import (
    AgentleMessageToOpenaiMessageAdapter,
)
from agentle.generations.providers.openai.adapters.agentle_tool_to_openai_tool_adapter import (
    AgentleToolToOpenaiToolAdapter,
)
from agentle.generations.providers.openai.adapters.chat_completion_to_generation_adapter import (
    ChatCompletionToGenerationAdapter,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing import observe

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agentle.generations.tracing.otel_client import OtelClient


type WithoutStructuredOutput = None


class NotGivenSentinel:
    def __bool__(self) -> Literal[False]:
        return False


NOT_GIVEN = NotGivenSentinel()


class OpenaiGenerationProvider(GenerationProvider):
    """
    OpenAI generation provider.
    """

    client: AsyncOpenAI

    def __init__(
        self,
        api_key: str | None = None,
        *,
        otel_clients: Sequence[OtelClient] | OtelClient | None = None,
        organization_name: str | None = None,
        project_name: str | None = None,
        base_url: str | httpx.URL | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        super().__init__(otel_clients=otel_clients)

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            organization=organization_name,
            project=project_name,
        )

    @property
    @override
    def default_model(self) -> str:
        return "gpt-4o"

    @observe
    @override
    @override_model_kind
    async def generate_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool[Any]] | None = None,
    ) -> Generation[T]:
        """
        Create a generation asynchronously using an OpenAI model.

        This method sends the provided messages to the OpenAI API and processes
        the response. With the @observe decorator, all the observability and tracing
        is handled automatically.

        Args:
            model: The OpenAI model to use for generation (e.g., "gpt-4o")
            messages: The sequence of messages to send to the model
            response_schema: Optional schema for structured output parsing
            generation_config: Optional configuration for the generation
            tools: Optional tools for function calling

        Returns:
            Generation[T]: An Agentle Generation object containing the response
        """
        from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN
        from openai.types.chat.chat_completion import ChatCompletion
        from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

        _generation_config = self._normalize_generation_config(generation_config)

        input_message_adapter = AgentleMessageToOpenaiMessageAdapter()
        openai_tool_adapter = AgentleToolToOpenaiToolAdapter()

        try:
            async with asyncio.timeout(_generation_config.timeout_in_seconds):
                chat_completion: ChatCompletion | ParsedChatCompletion[T] = (
                    await self._client.chat.completions.create(
                        model=self._resolve_model(model),
                        messages=[
                            input_message_adapter.adapt(message) for message in messages
                        ],
                        tools=[openai_tool_adapter.adapt(tool) for tool in tools]
                        if tools
                        else OPENAI_NOT_GIVEN,
                    )
                    if not bool(response_schema)
                    else await self._client.chat.completions.parse(
                        model=self._resolve_model(model),
                        messages=[
                            input_message_adapter.adapt(message) for message in messages
                        ],
                        tools=[openai_tool_adapter.adapt(tool) for tool in tools]
                        if tools
                        else OPENAI_NOT_GIVEN,
                        response_format=response_schema,
                    )
                )
        except asyncio.TimeoutError as e:
            e.add_note(
                f"Content generation timed out after {_generation_config.timeout_in_seconds}s"
            )
            raise

        output_adapter = ChatCompletionToGenerationAdapter[T]()
        return output_adapter.adapt(chat_completion)

    @property
    @override
    def organization(self) -> str:
        """
        Get the provider organization identifier.

        Returns:
            str: The organization identifier, which is "openai" for this provider.
        """
        return "openai"

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        mapping: Mapping[ModelKind, str] = {
            "category_nano": "gpt-4.1-nano",  # smallest, cost-effective nano model [7]
            "category_mini": "o4-mini",  # fast, cost-efficient reasoning model [3]
            "category_standard": "gpt-4.1",  # balanced, standard GPT-4.1 model [7][6]
            "category_pro": "gpt-4.5",  # high performance, latest GPT-4.5 research preview [2][3]
            "category_flagship": "o3",  # most powerful reasoning model, SOTA on coding/math/science [3][8]
            "category_reasoning": "o3",  # same as flagship, specialized for complex reasoning [3]
            "category_vision": "o3",  # strong visual perception capabilities [3]
            "category_coding": "o3",  # excels at coding tasks [3]
            "category_instruct": "gpt-4.1",  # instruction-following optimized [6][7]
            # Experimental fallback to stable (no distinct experimental models)
            "category_nano_experimental": "gpt-4.1-nano",
            "category_mini_experimental": "o4-mini",
            "category_standard_experimental": "gpt-4.1",
            "category_pro_experimental": "gpt-4.5",
            "category_flagship_experimental": "o3",
            "category_reasoning_experimental": "o3",
            "category_vision_experimental": "o3",
            "category_coding_experimental": "o3",
            "category_instruct_experimental": "gpt-4.1",
        }

        return mapping[model_kind]

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for input/prompt tokens.

        Uses OpenAI's pricing structure.

        Args:
            model: The model identifier
            estimate_tokens: Optional estimate of token count

        Returns:
            float: Price per million tokens for the specified model
        """
        # Pricing data from official OpenAI sources and industry analysis
        model_pricing = {
            # Nano models
            "gpt-4.1-nano": 2.50,  # Cost-effective nano model
            "gpt-4.o-mini": 2.50,  # GPT-4o mini pricing
            # Mid-tier models
            "o4-mini": 10.00,  # Comparable to GPT-4 Turbo pricing
            "gpt-4.o": 5.00,  # Standard GPT-4o pricing
            # Standard models
            "gpt-4.1": 30.00,  # Standard GPT-4.1 pricing
            # Pro models
            "gpt-4.5": 50.00,  # High-performance GPT-4.5
            # Flagship models
            "o3": 20.00,  # Premium reasoning model
        }
        return model_pricing.get(model, 0.0)

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for output/completion tokens.

        Uses OpenAI's pricing structure.

        Args:
            model: The model identifier
            estimate_tokens: Optional estimate of token count

        Returns:
            float: Price per million tokens for the specified model
        """
        # Pricing data from official OpenAI sources and industry analysis
        model_pricing = {
            # Nano models
            "gpt-4.1-nano": 5.00,  # Nano output pricing
            "gpt-4.o-mini": 10.00,  # GPT-4o mini output
            # Mid-tier models
            "o4-mini": 30.00,  # GPT-4 Turbo equivalent
            "gpt-4.o": 15.00,  # Standard GPT-4o output
            # Standard models
            "gpt-4.1": 60.00,  # Standard GPT-4.1 output
            # Pro models
            "gpt-4.5": 150.00,  # High-performance output
            # Flagship models
            "o3": 60.00,  # Premium output pricing
        }
        return model_pricing.get(model, 0.0)
