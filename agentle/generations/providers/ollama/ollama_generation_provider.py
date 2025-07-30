from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Sequence, cast, override

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.ollama.adapters.chat_response_to_generation_adapter import (
    ChatResponseToGenerationAdapter,
)
from agentle.generations.providers.ollama.adapters.message_to_ollama_message_adapter import (
    MessageToOllamaMessageAdapter,
)
from agentle.generations.providers.ollama.adapters.tool_to_ollama_tool_adapter import (
    ToolToOllamaToolAdapter,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing import observe

if TYPE_CHECKING:
    from ollama._types import Options

    from agentle.generations.tracing.otel_client import OtelClient


class OllamaGenerationProvider(GenerationProvider):
    def __init__(
        self,
        *,
        otel_clients: Sequence[OtelClient] | OtelClient | None = None,
        options: Mapping[str, Any] | Options | None = None,
        think: bool | None = None,
        host: str | None = None,
    ) -> None:
        from ollama._client import AsyncClient

        super().__init__(otel_clients=otel_clients)
        self._client = AsyncClient(host=host)
        self.options = options
        self.think = think

    @property
    @override
    def default_model(self) -> str:
        return "gemma3n:e4b"

    @property
    @override
    def organization(self) -> str:
        return "Ollama"

    @observe
    @override
    async def generate_async[T](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool[Any]] | None = None,
    ) -> Generation[T]:
        from pydantic import BaseModel

        tool_adapter = ToolToOllamaToolAdapter()

        bm = cast(BaseModel, response_schema) if response_schema else None

        _generation_config = self._normalize_generation_config(generation_config)

        _model = self._resolve_model(model)
        message_adapter = MessageToOllamaMessageAdapter()
        _messages = [message_adapter.adapt(m) for m in messages]

        _tools = [tool_adapter.adapt(tool) for tool in tools] if tools else None

        try:
            async with asyncio.timeout(_generation_config.timeout_in_seconds):
                response = await self._client.chat(
                    model=_model,
                    messages=_messages,
                    tools=_tools,
                    format=bm.model_json_schema() if bm else None,
                    options=self.options,
                    think=self.think,
                )
        except asyncio.TimeoutError as e:
            e.add_note(
                f"Content generation timed out after {_generation_config.timeout_in_seconds}s"
            )
            raise

        return ChatResponseToGenerationAdapter(
            model=_model, response_schema=response_schema
        ).adapt(response)  # type: ignore

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        """
        Maps abstract ModelKind categories to specific Ollama model names.

        This mapping is based on the latest available Ollama models as of July 2025,
        focusing on the most capable and well-supported models in each category.
        """
        mapping: Mapping[ModelKind, str] = {
            # Nano: Smallest, fastest, most cost-effective models
            "category_nano": "llama3.2:1b",
            "category_nano_experimental": "smollm2:135m",
            # Mini: Small but capable models
            "category_mini": "llama3.2:3b",
            "category_mini_experimental": "phi4:mini",
            # Standard: Mid-range, balanced performance models
            "category_standard": "llama3.1:8b",
            "category_standard_experimental": "qwen2.5:7b",
            # Pro: High performance models
            "category_pro": "llama3.1:70b",
            "category_pro_experimental": "qwen2.5:14b",
            # Flagship: Best available models from provider
            "category_flagship": "llama3.3:70b",
            "category_flagship_experimental": "qwen3:235b",
            # Reasoning: Specialized for complex reasoning
            "category_reasoning": "deepseek-r1:32b",
            "category_reasoning_experimental": "qwq:32b",
            # Vision: Multimodal capabilities for image/video processing
            "category_vision": "llama3.2-vision:11b",
            "category_vision_experimental": "qwen2-vl:7b",
            # Coding: Specialized for programming tasks
            "category_coding": "codellama:13b",
            "category_coding_experimental": "qwen2.5-coder:7b",
            # Instruct: Fine-tuned for instruction following
            "category_instruct": "dolphin-llama3:8b",
            "category_instruct_experimental": "openhermes:7b",
        }

        return mapping.get(model_kind, "llama3.1:8b")  # Default fallback
