"""
Adapter components for the OpenRouter provider integration in the Agentle framework.

This module contains adapter classes that transform requests to and responses from
OpenRouter's API into standardized formats used throughout the Agentle framework.
These adapters serve as part of Agentle's provider abstraction layer, allowing the
framework to present a unified interface regardless of the underlying AI provider.

The adapters in this package handle various conversion tasks, including:
- Converting Agentle message objects to OpenRouter message format
- Transforming OpenRouter responses to Agentle Generation objects
- Supporting structured output parsing for type-safe responses
- Handling multimodal content (images, PDFs, audio)
- Converting tool definitions and tool calls

These adapters ensure that all provider-specific details of OpenRouter's request and
response formats are processed and normalized to Agentle's internal representation,
maintaining consistency across different AI providers within the framework.
"""

from .agentle_message_to_openrouter_message_adapter import (
    AgentleMessageToOpenRouterMessageAdapter,
)
from .agentle_part_to_openrouter_part_adapter import (
    AgentlePartToOpenRouterPartAdapter,
)
from .agentle_tool_to_openrouter_tool_adapter import (
    AgentleToolToOpenRouterToolAdapter,
)
from .openrouter_response_to_generation_adapter import (
    OpenRouterResponseToGenerationAdapter,
)
from .openrouter_message_to_generated_assistant_message_adapter import (
    OpenRouterMessageToGeneratedAssistantMessageAdapter,
)

__all__ = [
    "AgentleMessageToOpenRouterMessageAdapter",
    "AgentlePartToOpenRouterPartAdapter",
    "AgentleToolToOpenRouterToolAdapter",
    "OpenRouterResponseToGenerationAdapter",
    "OpenRouterMessageToGeneratedAssistantMessageAdapter",
]
