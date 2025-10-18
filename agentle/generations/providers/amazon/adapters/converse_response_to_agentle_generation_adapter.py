from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import datetime
import logging
from typing import TYPE_CHECKING, Any, cast, override
import uuid

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.pricing import Pricing
from agentle.generations.models.generation.usage import Usage

from agentle.generations.providers.amazon.adapters.boto_message_to_agentle_message_adapter import (
    BotoMessageToAgentleMessageAdapter,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef
    from agentle.generations.providers.amazon.bedrock_generation_provider import BedrockGenerationProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConverseResponseToAgentleGenerationAdapter[T](
    Adapter["ConverseResponseTypeDef", Generation[T]]
):
    model: str
    response_schema: type[T] | None = field(default=None)
    provider: "BedrockGenerationProvider | None" = field(default=None)

    @override
    def adapt(self, _f: ConverseResponseTypeDef) -> Generation[T]:
        # Without Structured Outputs
        # b'{"metrics":{"latencyMs":1501},"output":{"message":{"content":[{"text":"Hello! It\'s nice to meet you. How are you doing today? Is there anything I can help you with?"}],"role":"assistant"}},"stopReason":"end_turn","usage":{"cacheReadInputTokenCount":0,"cacheReadInputTokens":0,"cacheWriteInputTokenCount":0,"cacheWriteInputTokens":0,"inputTokens":14,"outputTokens":27,"totalTokens":41}}'

        amazon_usage: Mapping[str, Any] = cast(Mapping[str, Any], _f["usage"])

        usage = Usage(
            prompt_tokens=amazon_usage["inputTokens"],
            completion_tokens=amazon_usage["outputTokens"],
        )

        _message_adater = BotoMessageToAgentleMessageAdapter(
            response_schema=self.response_schema
        )

        _bedrock_output = _f.get("output")
        _bedrock_message = _bedrock_output.get("message")

        if _bedrock_message is None:
            raise ValueError(
                "Could not get message from bedrock response."
                + f"Bedrock response: {_f}"
            )

        return Generation(
            id=uuid.uuid4(),
            model=self.model,
            object="chat.generation",
            choices=[Choice(index=0, message=_message_adater.adapt(_bedrock_message))],
            created=datetime.datetime.now(),
            usage=usage,
        )
    
    async def adapt_async(self, _f: ConverseResponseTypeDef) -> Generation[T]:
        """Convert Bedrock response to Generation asynchronously with pricing.
        
        Args:
            _f: The Bedrock response to convert.
            
        Returns:
            Generation object with normalized data and pricing information.
        """
        amazon_usage: Mapping[str, Any] = cast(Mapping[str, Any], _f["usage"])

        usage = Usage(
            prompt_tokens=amazon_usage["inputTokens"],
            completion_tokens=amazon_usage["outputTokens"],
        )

        _message_adater = BotoMessageToAgentleMessageAdapter(
            response_schema=self.response_schema
        )

        _bedrock_output = _f.get("output")
        _bedrock_message = _bedrock_output.get("message")

        if _bedrock_message is None:
            raise ValueError(
                "Could not get message from bedrock response."
                + f"Bedrock response: {_f}"
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
            id=uuid.uuid4(),
            model=self.model,
            object="chat.generation",
            choices=[Choice(index=0, message=_message_adater.adapt(_bedrock_message))],
            created=datetime.datetime.now(),
            usage=usage,
            pricing=pricing,
        )
