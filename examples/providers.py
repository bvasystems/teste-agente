"""
Providers Example

This example demonstrates how to use different model providers with the Agentle framework.
"""

from dotenv import load_dotenv

from agentle.generations.models.message_parts.file import FilePart
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
provider: GenerationProvider = OpenRouterGenerationProvider().with_fallback_models(
    "openai/gpt-5-nano"
)

example_file = FilePart.from_local_file(
    "/Users/arthurbrenno/Documents/Dev/Paragon/agentle/examples/harvard.wav",
    "audio/mp3",
)

# Run the Google agent
generation = provider.generate(
    model="google/gemini-2.5-flash",
    messages=[
        UserMessage(
            parts=[
                example_file,
                TextPart(
                    text="O que tem nesse audio?",
                ),
            ]
        )
    ],
)

print(generation)
