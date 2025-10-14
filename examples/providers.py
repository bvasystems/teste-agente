"""
Providers Example

This example demonstrates how to use different model providers with the Agentle framework.
"""

from dotenv import load_dotenv

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.openrouter.openrouter_generation_provider import (
    OpenRouterGenerationProvider,
)

load_dotenv()


def add_numbers(a: float, b: float) -> float:
    return a + b


# Example 1: Create an agent with Google's Gemini model
provider: GenerationProvider = OpenRouterGenerationProvider()

# Run the Google agent
generation = provider.generate(
    model="meta-llama/llama-3.3-70b-instruct:free",
    messages=[
        UserMessage(
            parts=[
                TextPart(
                    text="what is 2+2?",
                )
            ]
        )
    ],
)

print(generation)
