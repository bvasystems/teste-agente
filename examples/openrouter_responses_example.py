import asyncio
import time

from dotenv import load_dotenv
from rsb.models.base_model import BaseModel

from agentle.generations.models.generation.trace_params import TraceParams
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
        model="google/gemma-3-27b-it",
        max_output_tokens=5000,
        text_format=MathResponse,
    )
    elapsed_time = time.time() - start_time

    print(f"Parsed output: {response.output_parsed}")
    print(f"Time taken: {elapsed_time:.3f}s")

    return elapsed_time


async def with_langfuse(run_num: int):
    """Test with Langfuse tracing and TraceParams."""
    print(f"\n=== WITH LANGFUSE + TraceParams (Run {run_num}) ===")

    responder = Responder.openrouter()
    responder.append_otel_client(LangfuseOtelClient())

    # Create trace params with metadata
    trace_params = TraceParams(
        name=f"math_calculation_run_{run_num}",
        user_id="test_user_123",
        session_id=f"test_session_{run_num // 3}",  # Group every 3 runs
        tags=["math", "testing", "performance"],
        version="1.0.0",
        release="development",
        metadata={
            "run_number": run_num,
            "test_type": "performance_comparison",
            "model_type": "gemma",
            "expected_answer": 4,
        },
    )

    start_time = time.time()
    response = await responder.respond_async(
        input="What is 2+2?",
        model="google/gemma-3-27b-it",
        max_output_tokens=5000,
        text_format=MathResponse,
        trace_params=trace_params,
    )
    elapsed_time = time.time() - start_time

    print(f"Parsed output: {response.output_parsed}")
    print(f"Time taken: {elapsed_time:.3f}s")
    print(f"Trace name: {trace_params.get('name')}")
    print(f"Session ID: {trace_params.get('session_id')}")
    print(f"Tags: {trace_params.get('tags')}")

    return elapsed_time


async def main():
    """Compare performance with and without Langfuse over multiple runs."""

    num_runs = 10
    times_without: list[float] = []
    times_with: list[float] = []

    print("=" * 60)
    print("Testing TraceParams Integration with Responder")
    print("=" * 60)
    print(f"\nRunning {num_runs} iterations to compare performance...")
    print("Check your Langfuse dashboard to see the trace metadata!")

    for i in range(1, num_runs + 1):
        # Run without Langfuse
        time_without = await without_langfuse(i)
        times_without.append(time_without)

        # Run with Langfuse + TraceParams
        time_with = await with_langfuse(i)
        times_with.append(time_with)

    # Calculate averages
    avg_without = sum(times_without) / len(times_without)
    avg_with = sum(times_with) / len(times_with)
    avg_overhead = avg_with - avg_without
    avg_overhead_pct = (avg_overhead / avg_without * 100) if avg_without > 0 else 0

    # Show comparison
    print("\n" + "=" * 60)
    print("=== FINAL RESULTS ===")
    print("=" * 60)
    print("\nWithout Langfuse:")
    print(f"  Individual runs: {[f'{t:.3f}s' for t in times_without]}")
    print(f"  Average: {avg_without:.3f}s")

    print("\nWith Langfuse + TraceParams:")
    print(f"  Individual runs: {[f'{t:.3f}s' for t in times_with]}")
    print(f"  Average: {avg_with:.3f}s")

    print("\nOverhead:")
    print(f"  Average overhead: {avg_overhead:.3f}s ({avg_overhead_pct:.1f}%)")

    print("\n" + "=" * 60)
    print("TraceParams Features Tested:")
    print("=" * 60)
    print("✓ name: Custom trace names for each run")
    print("✓ user_id: User identification")
    print("✓ session_id: Session grouping (every 3 runs)")
    print("✓ tags: Categorization with multiple tags")
    print("✓ version: Version tracking")
    print("✓ release: Release environment tracking")
    print("✓ metadata: Custom metadata (run_number, test_type, etc.)")
    print("\nCheck your Langfuse dashboard to see all this metadata!")


if __name__ == "__main__":
    asyncio.run(main())
