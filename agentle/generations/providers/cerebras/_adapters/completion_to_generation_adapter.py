"""
Adapter for converting Cerebras chat completion responses to Agentle Generation objects.

This module provides the CerebrasCompletionToGenerationAdapter class, which transforms
responses from Cerebras's Cloud SDK (ChatCompletionResponse) into the standardized Generation
format used throughout the Agentle framework. The adapter handles conversion of response
content, choices, metadata, and usage statistics.

This adapter is a critical component of Agentle's provider abstraction layer, enabling
the framework to present a unified interface regardless of which underlying AI provider
is being used. It processes all the provider-specific details of Cerebras's response format
and normalizes them to Agentle's internal representation.

The adapter supports structured output parsing for type-safe responses through its
generic type parameter.
"""

from __future__ import annotations
import datetime
import logging
import uuid
from typing import TYPE_CHECKING

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.pricing import Pricing
from agentle.generations.models.generation.usage import Usage
from agentle.generations.providers.cerebras._adapters.cerebras_message_to_generated_assistant_message_adapter import (
    CerebrasMessageToGeneratedAssistantMessageAdapter,
)
from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatCompletionResponse
    from agentle.generations.providers.cerebras.cerebras_generation_provider import CerebrasGenerationProvider

logger = logging.getLogger(__name__)


class CerebrasCompletionToGenerationAdapter[T](
    Adapter["ChatCompletionResponse", Generation[T]]
):
    """
    Adapter for converting Cerebras ChatCompletionResponse objects to Agentle Generation objects.

    This adapter transforms the response format from Cerebras's Cloud SDK into Agentle's
    standardized Generation format. It processes elements such as candidate responses,
    usage statistics, and structured output data.

    The adapter is generic over type T, which represents the optional structured data
    format that can be extracted from the model's response when a response schema is provided.

    This class plays a key role in Agentle's provider abstraction layer by normalizing
    Cerebras-specific response formats into the framework's unified representation.

    Attributes:
        response_schema: Optional Pydantic model class for parsing structured data from
            the response.
        start_time: The timestamp when the generation request was initiated, used to
            calculate elapsed time for the generation.
        model: The name of the model that was used to generate the response.
        message_to_generated_assistant_message_adapter: Adapter for converting Cerebras
            message objects to Agentle message objects.
        preferred_id: Optional UUID to use for the resulting Generation object.
    """

    response_schema: type[T] | None
    model: str
    message_to_generated_assistant_message_adapter: (
        CerebrasMessageToGeneratedAssistantMessageAdapter[T]
    )
    preferred_id: uuid.UUID | None
    provider: "CerebrasGenerationProvider | None"

    def __init__(
        self,
        *,
        response_schema: type[T] | None = None,
        model: str,
        message_to_generated_assistant_message_adapter: CerebrasMessageToGeneratedAssistantMessageAdapter[
            T
        ]
        | None = None,
        preferred_id: uuid.UUID | None = None,
        provider: "CerebrasGenerationProvider | None" = None,
    ):
        """
        Initialize the adapter with the necessary configuration.

        Args:
            response_schema: Optional Pydantic model class for parsing structured data
                from the response.
            start_time: The timestamp when the generation request was initiated.
            model: The name of the model that generated the response.
            message_to_generated_assistant_message_adapter: Optional adapter for converting
                Cerebras message objects to Agentle message objects.
            preferred_id: Optional UUID to use for the resulting Generation object.
            provider: Optional provider instance for pricing calculation.
        """
        self.response_schema = response_schema
        self.model = model
        self.message_to_generated_assistant_message_adapter = (
            message_to_generated_assistant_message_adapter
            or CerebrasMessageToGeneratedAssistantMessageAdapter(
                response_schema=response_schema
            )
        )
        self.preferred_id = preferred_id
        self.provider = provider

    def adapt(self, _f: ChatCompletionResponse) -> Generation[T]:
        """
        Convert a Cerebras ChatCompletionResponse to an Agentle Generation object.

        This method processes the response from Cerebras's API, extracting choices,
        usage statistics, and any structured data. It then constructs a standardized
        Generation object that can be used consistently throughout the Agentle framework.

        Args:
            _f: The Cerebras ChatCompletionResponse object to adapt, typically received
                directly from Cerebras's Cloud SDK.

        Returns:
            Generation[T]: An Agentle Generation object containing the normalized response
                data, including any parsed structured output if a response_schema was provided.
        """
        choices: list[Choice[T]] = [
            Choice(
                index=index,
                message=self.message_to_generated_assistant_message_adapter.adapt(
                    choice.message
                ),
            )
            for index, choice in enumerate(_f.choices)
        ]

        usage = Usage(
            prompt_tokens=_f.usage.prompt_tokens or 0,
            completion_tokens=_f.usage.completion_tokens or 0,
        )

        return Generation(
            id=self.preferred_id or uuid.uuid4(),
            choices=choices,
            object="chat.generation",
            created=datetime.datetime.now(),
            model=_f.model,
            usage=usage,
        )
    
    async def adapt_async(self, _f: ChatCompletionResponse) -> Generation[T]:
        """
        Convert a Cerebras ChatCompletionResponse to an Agentle Generation object asynchronously.
        
        This async version calculates pricing information if provider is available.

        Args:
            _f: The Cerebras ChatCompletionResponse object to adapt.

        Returns:
            Generation[T]: An Agentle Generation object with pricing information.
        """
        choices: list[Choice[T]] = [
            Choice(
                index=index,
                message=self.message_to_generated_assistant_message_adapter.adapt(
                    choice.message
                ),
            )
            for index, choice in enumerate(_f.choices)
        ]

        usage = Usage(
            prompt_tokens=_f.usage.prompt_tokens or 0,
            completion_tokens=_f.usage.completion_tokens or 0,
        )
        
        # Calculate pricing if provider is available
        pricing = Pricing()
        if self.provider is not None:
            provider = self.provider
            model = self.model
            try:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                if input_tokens > 0 or output_tokens > 0:
                    input_price_per_million = await provider.price_per_million_tokens_input(
                        model, input_tokens
                    )
                    output_price_per_million = await provider.price_per_million_tokens_output(
                        model, output_tokens
                    )
                    
                    input_cost = input_price_per_million * (input_tokens / 1_000_000)
                    output_cost = output_price_per_million * (output_tokens / 1_000_000)
                    total_cost = input_cost + output_cost
                    
                    pricing = Pricing(
                        input_pricing=round(input_cost, 8),
                        output_pricing=round(output_cost, 8),
                        total_pricing=round(total_cost, 8),
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to calculate pricing: {e}")
                pricing = Pricing()

        return Generation(
            id=self.preferred_id or uuid.uuid4(),
            choices=choices,
            object="chat.generation",
            created=datetime.datetime.now(),
            model=_f.model,
            usage=usage,
            pricing=pricing,
        )
