"""
Example usage of OpenRouterResponder with the new Responses API.

This demonstrates:
- Basic text generation
- Streaming responses
- Structured output with Pydantic models
- Tool calling
- Web search integration
- Reasoning capabilities
"""

import asyncio
import os

from dotenv import load_dotenv
from rsb.models.base_model import BaseModel

from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.definitions.reasoning_effort import ReasoningEffort
from agentle.responses.open_router.open_router_responder import OpenRouterResponder

load_dotenv()


class MathResponse(BaseModel):
    math_result: int


async def main():
    """Basic text generation example."""
    responder = OpenRouterResponder(api_key=os.getenv("OPENAI_API_KEY"))

    print("Starting...")
    response = await responder.respond_async(
        input="What is 2+2?",
        model="gpt-5-nano",
        max_output_tokens=500,
        text_format=MathResponse,
        reasoning=Reasoning(
            effort=ReasoningEffort.high,
        ),
        stream=True,
    )

    async for event in response:
        print(event)

    print("Response: ")
    print(response)

    print("Output parsed: ")
    print(response.output_parsed)


if __name__ == "__main__":
    asyncio.run(main())
