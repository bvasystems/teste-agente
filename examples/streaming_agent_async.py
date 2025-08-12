from agentle.agents.agent import Agent

from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()


class Response(BaseModel):
    reasoning: str
    response: str


async def sum(a: float, b: float) -> float:
    return a + b


agent = Agent()

print("Streaming poem generation...")
print("=" * 50)

for chunk in agent.run("write a poem about america", stream=True):
    print(chunk.text)
