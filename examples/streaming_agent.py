import asyncio
from agentle.agents.agent import Agent

from pydantic import BaseModel


class Response(BaseModel):
    reasoning: str
    response: str


async def sum(a: float, b: float) -> float:
    return a + b


async def main():
    agent = Agent()
    
    print("Streaming poem generation...")
    print("=" * 50)
    
    async for chunk in await agent.run_async("write a poem about america", stream=True):
        # Print each chunk as it arrives
        if chunk.generation and chunk.generation.choices:
            for choice in chunk.generation.choices:
                if choice.message and choice.message.parts:
                    for part in choice.message.parts:
                        if hasattr(part, 'text') and getattr(part, 'text', None):
                            print(str(getattr(part, 'text')), end='', flush=True)
    
    print("\n" + "=" * 50)
    print("Streaming complete!")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(main())
