import asyncio

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider


async def main():
    provider = GoogleGenerationProvider()
    stream = await provider.stream_async(
        messages=[
            UserMessage(
                parts=[
                    TextPart(
                        text="hello! write a long poem about the yanomami community"
                    )
                ]
            )
        ]
    )

    full_text = ""
    chunk_count = 0

    async for generation in stream:
        chunk_count += 1
        chunk_text = generation.text
        full_text += chunk_text

        print(f"Chunk {chunk_count}: {chunk_text!r}")
        print(f"Tokens so far: {generation.usage.completion_tokens}")

        # # You can check if this is the final chunk
        # if generation.is_final_chunk:
        #     print("Final chunk received!")

    print(f"\nFull response: {full_text}")
    print(f"Total chunks: {chunk_count}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(main())
