from __future__ import annotations

import datetime
import logging
import uuid
from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.pricing import Pricing
from agentle.generations.models.generation.usage import Usage
from agentle.generations.providers.openai.adapters.openai_choice_to_choice_adapter import (
    OpenaiChoiceToChoiceAdapter,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
    from agentle.generations.providers.openai.openai import OpenaiGenerationProvider

logger = logging.getLogger(__name__)


class ChatCompletionToGenerationAdapter[T](
    Adapter["ChatCompletion | ParsedChatCompletion[T]", Generation[T]]
):
    provider: "OpenaiGenerationProvider | None"
    model: str | None
    
    def __init__(
        self,
        *,
        provider: "OpenaiGenerationProvider | None" = None,
        model: str | None = None,
    ):
        """Initialize the adapter.
        
        Args:
            provider: Optional provider instance for pricing calculation.
            model: Optional model identifier for pricing calculation.
        """
        self.provider = provider
        self.model = model
    
    def adapt(self, _f: ChatCompletion | ParsedChatCompletion[T]) -> Generation[T]:
        from openai.types.completion_usage import CompletionUsage

        completion = _f
        choice_adapter = OpenaiChoiceToChoiceAdapter[T]()

        usage = completion.usage or CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )

        return Generation(
            id=uuid.uuid4(),
            object="chat.generation",
            created=datetime.datetime.fromtimestamp(completion.created),
            model=completion.model,
            choices=[choice_adapter.adapt(choice) for choice in completion.choices],
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ),
        )
    
    async def adapt_async(self, _f: ChatCompletion | ParsedChatCompletion[T]) -> Generation[T]:
        """Convert OpenAI completion to Generation asynchronously with pricing.
        
        Args:
            _f: The OpenAI completion to convert.
            
        Returns:
            Generation object with normalized data and pricing information.
        """
        from openai.types.completion_usage import CompletionUsage

        completion = _f
        choice_adapter = OpenaiChoiceToChoiceAdapter[T]()

        usage = completion.usage or CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=0
        )
        
        # Calculate pricing if provider and model are available
        pricing = Pricing()
        if self.provider is not None and self.model is not None:
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
            id=uuid.uuid4(),
            object="chat.generation",
            created=datetime.datetime.fromtimestamp(completion.created),
            model=completion.model,
            choices=[choice_adapter.adapt(choice) for choice in completion.choices],
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ),
            pricing=pricing,
        )
