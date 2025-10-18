import asyncio
from pydantic import BaseModel

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.openrouter.openrouter_generation_provider import (
    OpenRouterGenerationProvider,
)
from dotenv import load_dotenv

load_dotenv()


class Answer(BaseModel):
    reasoning: str
    answer: str
    confidence: float


provider = OpenRouterGenerationProvider()


async def main():
    async for generation in provider.stream_async(
        model="google/gemini-2.5-flash-preview-09-2025",
        messages=[UserMessage(parts=[TextPart(text="What is 2+2?")])],
        response_schema=Answer,  # ✅ Now supported!
    ):
        # Stream the JSON as it's generated
        print(generation)
        print(generation.parsed)


asyncio.run(main())
