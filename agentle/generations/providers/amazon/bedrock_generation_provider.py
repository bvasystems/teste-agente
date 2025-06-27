from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Sequence, cast, override

from rsb.coroutines.run_async import run_async

from agentle.generations.collections.message_sequence import MessageSequence
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.amazon.adapters.agentle_message_to_boto_message import (
    AgentleMessageToBotoMessage,
)
from agentle.generations.providers.amazon.adapters.agentle_tool_to_bedrock_tool_adapter import (
    AgentleToolToBedrockToolAdapter,
)
from agentle.generations.providers.amazon.adapters.converse_response_to_agentle_generation_adapter import (
    ConverseResponseToAgentleGenerationAdapter,
)
from agentle.generations.providers.amazon.adapters.generation_config_to_inference_config import (
    GenerationConfigToInferenceConfigAdapter,
)
from agentle.generations.providers.amazon.adapters.response_schema_to_bedrock_tool_adapter import (
    ResponseSchemaToBedrockToolAdapter,
)
from agentle.generations.providers.amazon.boto_config import BotoConfig
from agentle.generations.providers.amazon.models.specific_tool import SpecificTool
from agentle.generations.providers.amazon.models.text_content import TextContent
from agentle.generations.providers.amazon.models.tool_choice import ToolChoice
from agentle.generations.providers.amazon.models.tool_config import ToolConfig
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.generations.tracing.decorators.observe import observe

logger = logging.getLogger(__name__)


class BedrockGenerationProvider(GenerationProvider):
    _client: Any
    region_name: str
    access_key_id: str | None
    secret_access_key: str | None
    config: BotoConfig | None

    def __init__(
        self,
        *,
        tracing_client: StatefulObservabilityClient | None = None,
        region_name: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        config: BotoConfig | None = None,
    ):
        import boto3

        super().__init__(tracing_client=tracing_client)

        self._client = self._client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_access_key
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
            config=config,
        )

    @property
    @override
    def default_model(self) -> str:
        return "anthropic.claude-sonnet-4-20250514-v1:0"

    @property
    @override
    def organization(self) -> str:
        return "aws"

    @observe
    @override
    async def create_generation_async[T](
        self,
        *,
        model: str | None | ModelKind = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool[Any]] | None = None,
    ) -> Generation[T]:
        from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef

        message_adapter = AgentleMessageToBotoMessage()

        message_sequence = MessageSequence(messages)

        messages_without_system = message_sequence.without_developer_prompt().elements

        system_message: DeveloperMessage | None = (
            messages_without_system[0]
            if isinstance(messages_without_system[0], DeveloperMessage)
            else None
        )

        conversation = [
            message_adapter.adapt(message) for message in messages_without_system
        ]

        inference_config_adapter = GenerationConfigToInferenceConfigAdapter()
        tool_adapter = AgentleToolToBedrockToolAdapter()

        _inference_config = inference_config_adapter.adapt(
            generation_config
            if isinstance(generation_config, GenerationConfig)
            else GenerationConfig()
            if not isinstance(generation_config, dict)
            else GenerationConfig(**generation_config)
        )

        # TODO: rs_tool = response_schema_to_bedrock_tool(response_schema)
        rs_tool = (
            ResponseSchemaToBedrockToolAdapter().adapt(response_schema)
            if response_schema
            else None
        )

        extra_tools = [rs_tool] if rs_tool else []

        _tools = tools or []

        _tool_config = (
            ToolConfig(
                tools=[tool_adapter.adapt(tool) for tool in _tools] + extra_tools,
                toolChoice=ToolChoice(auto={})
                if response_schema is None
                else ToolChoice(tool=SpecificTool(name=response_schema.__name__)),
            )
            if _tools or extra_tools
            else None
        )

        _system = [
            TextContent(
                text=system_message.text
                if system_message
                else "You are a helpful assistant"
            )
        ]

        _model = model or self.default_model

        if _tool_config:
            response: ConverseResponseTypeDef = cast(
                ConverseResponseTypeDef,
                await run_async(
                    self._client.converse,
                    modelId=_model,
                    system=_system,
                    messages=conversation,
                    inferenceConfig=_inference_config,
                    toolConfig=_tool_config,
                ),
            )
        else:
            response: ConverseResponseTypeDef = cast(
                ConverseResponseTypeDef,
                await run_async(
                    self._client.converse,
                    modelId=_model,
                    system=_system,
                    messages=conversation,
                    inferenceConfig=_inference_config,
                ),
            )

        logger.debug(f"Received Bedrock Response: {response}")

        return ConverseResponseToAgentleGenerationAdapter(
            response_schema=response_schema
        ).adapt(response)

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float: ...

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float: ...

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        mapping: Mapping[ModelKind, str] = {
            # Stable models
            "category_nano": "",
            "category_mini": "",
            "category_standard": "",
            "category_pro": "",
            "category_flagship": "",
            "category_reasoning": "",
            "category_vision": "",
            "category_coding": "",
            "category_instruct": "",
            # Experimental models
            "category_nano_experimental": "",
            "category_mini_experimental": "",
            "category_standard_experimental": "",
            "category_pro_experimental": "",
            "category_flagship_experimental": "",
            "category_reasoning_experimental": "",
            "category_vision_experimental": "",
            "category_coding_experimental": "",
            "category_instruct_experimental": "",
        }

        return mapping[model_kind]
