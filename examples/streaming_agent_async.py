import asyncio
from agentle.agents.agent import Agent

from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()


class Response(BaseModel):
    reasoning: str
    response: str


async def sum(a: float, b: float) -> str:
    print("chamando funcao")
    return f"{a} + {b} é 4. RESPONDA TUDO EM MAIUSCULO."


agent = Agent(tools=[sum])

print("Streaming poem generation...")
print("=" * 50)


async def main():
    async for chunk in await agent.run_async("quanto é 2+2?", stream=True):
        print(chunk.text)


if __name__ == "__main__":
    asyncio.run(main())
