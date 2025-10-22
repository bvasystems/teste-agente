import asyncio
import time

from dotenv import load_dotenv
from rsb.models.base_model import BaseModel


from agentle.generations.tracing.langfuse_otel_client import LangfuseOtelClient
from agentle.responses.responder import Responder

load_dotenv(override=True)


class MathResponse(BaseModel):
    math_result: int


def add(a: int, b: int) -> int:
    return a + b


async def without_langfuse(run_num: int):
    """Test without Langfuse tracing."""
    print(f"\n=== WITHOUT LANGFUSE (Run {run_num}) ===")

    responder = Responder.openrouter()

    start_time = time.time()
    response = await responder.respond_async(
        input="What is 2+2?",
        model="gpt-5-nano",
        max_output_tokens=5000,
        text_format=MathResponse,
    )
    elapsed_time = time.time() - start_time

    print(f"Parsed output: {response.output_parsed}")
    print(f"Time taken: {elapsed_time:.3f}s")

    return elapsed_time


async def with_langfuse(run_num: int):
    """Test with Langfuse tracing."""
    print(f"\n=== WITH LANGFUSE (Run {run_num}) ===")

    responder = Responder.openrouter()
    responder.append_otel_client(LangfuseOtelClient())

    start_time = time.time()
    response = await responder.respond_async(
        input="What is 2+2?",
        model="gpt-5-nano",
        max_output_tokens=5000,
        text_format=MathResponse,
    )
    elapsed_time = time.time() - start_time

    print(f"Parsed output: {response.output_parsed}")
    print(f"Time taken: {elapsed_time:.3f}s")

    return elapsed_time


async def main():
    """Compare performance with and without Langfuse over 4 runs."""

    num_runs = 4
    times_without: list[float] = []
    times_with: list[float] = []

    for i in range(1, num_runs + 1):
        # Run without Langfuse
        time_without = await without_langfuse(i)
        times_without.append(time_without)

        # Run with Langfuse
        time_with = await with_langfuse(i)
        times_with.append(time_with)

    # Calculate averages
    avg_without = sum(times_without) / len(times_without)
    avg_with = sum(times_with) / len(times_with)
    avg_overhead = avg_with - avg_without
    avg_overhead_pct = (avg_overhead / avg_without * 100) if avg_without > 0 else 0

    # Show comparison
    print("\n" + "=" * 50)
    print("=== FINAL RESULTS ===")
    print("=" * 50)
    print("\nWithout Langfuse:")
    print(f"  Individual runs: {[f'{t:.3f}s' for t in times_without]}")
    print(f"  Average: {avg_without:.3f}s")

    print("\nWith Langfuse:")
    print(f"  Individual runs: {[f'{t:.3f}s' for t in times_with]}")
    print(f"  Average: {avg_with:.3f}s")

    print("\nOverhead:")
    print(f"  Average overhead: {avg_overhead:.3f}s ({avg_overhead_pct:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
