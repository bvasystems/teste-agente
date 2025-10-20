import asyncio

from dotenv import load_dotenv
from rsb.models.base_model import BaseModel


from agentle.responses.responder import Responder

load_dotenv()


class MathResponse(BaseModel):
    math_result: int


def add(a: int, b: int) -> int:
    return a + b


async def main():
    """Basic text generation example."""
    responder = Responder.from_openai()

    print("Starting...")
    response = await responder.respond_async(
        input="What is 2+2? call the function and also return structured output",
        model="gpt-5-nano",
        max_output_tokens=5000,
        text_format=MathResponse,
    )

    print(response)
    print(response.output_parsed)
    print(response.function_calls)


if __name__ == "__main__":
    asyncio.run(main())
