import asyncio

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.generations.tools.tool import Tool
from pydantic import BaseModel


class Response(BaseModel):
    reasoning: str
    response: str


async def sum(a: float, b: float) -> float:
    return a + b


async def main():
    provider = GoogleGenerationProvider()
    stream = provider.stream_async(
        messages=[
            UserMessage(parts=[TextPart(text="quanto é 2+2?")])
        ],
        tools=[Tool.from_callable(sum)],
    )

    full_text = ""
    chunk_count = 0

    async for generation in stream:
        print(f"\n{generation}\n\n\n")
        chunk_count += 1
        # chunk_text = generation.text
        # full_text += chunk_text

        # print(f"Chunk {chunk_count}: {chunk_text!r}")
        # print(f"Tokens so far: {generation.usage.completion_tokens}")

    print(f"\nFull response: {full_text}")
    print(f"Total chunks: {chunk_count}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(main())
