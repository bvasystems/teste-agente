"""
OpenRouter Advanced Features Example

Demonstrates the improved OpenRouter provider with:
1. Dynamic pricing from OpenRouter API
2. Fallback models as method parameters
3. Full GenerationConfig support (including top_k)
4. GenerationConfig overriding constructor parameters
"""

import asyncio
from agentle.generations.providers.openrouter import OpenRouterGenerationProvider
from agentle.generations.models.messages import UserMessage
from agentle.generations.models.message_parts import TextPart
from agentle.generations.models.generation.generation_config import GenerationConfig


async def example_dynamic_pricing():
    """Example: Dynamic pricing fetched from OpenRouter API."""
    print("\n=== Dynamic Pricing Example ===")
    
    provider = OpenRouterGenerationProvider()
    
    # Pricing is automatically fetched from OpenRouter /models endpoint
    model = "anthropic/claude-sonnet-4.5"
    
    input_price = await provider.price_per_million_tokens_input(model)
    output_price = await provider.price_per_million_tokens_output(model)
    
    print(f"Model: {model}")
    print(f"Input price: ${input_price:.4f} per million tokens")
    print(f"Output price: ${output_price:.4f} per million tokens")
    
    # Try another model
    model2 = "google/gemini-2.5-flash-preview-09-2025"
    input_price2 = await provider.price_per_million_tokens_input(model2)
    output_price2 = await provider.price_per_million_tokens_output(model2)
    
    print(f"\nModel: {model2}")
    print(f"Input price: ${input_price2:.4f} per million tokens")
    print(f"Output price: ${output_price2:.4f} per million tokens")


async def example_fallback_as_parameter():
    """Example: Pass fallback models as method parameter instead of constructor."""
    print("\n=== Fallback Models as Parameter ===")
    
    # No fallbacks in constructor
    provider = OpenRouterGenerationProvider()
    
    # Pass fallbacks directly in the method call
    generation = await provider.generate_async(
        model="openai/gpt-4o",
        messages=[
            UserMessage(parts=[TextPart(text="What is 2+2?")])
        ],
        fallback_models=[
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-flash-preview-09-2025"
        ]
    )
    
    print(f"Primary model: openai/gpt-4o")
    print(f"Fallbacks: claude-3.5-sonnet, gemini-2.5-flash")
    print(f"Actually used: {generation.model}")
    print(f"Response: {generation.text}")


async def example_generation_config_full():
    """Example: Full GenerationConfig support including top_k."""
    print("\n=== Full GenerationConfig Support ===")
    
    provider = OpenRouterGenerationProvider()
    
    config = GenerationConfig(
        temperature=0.7,
        max_output_tokens=500,
        top_p=0.9,
        top_k=40.0,  # Now supported!
    )
    
    generation = await provider.generate_async(
        model="anthropic/claude-sonnet-4.5",
        messages=[
            UserMessage(parts=[TextPart(text="Write a creative story about a robot.")])
        ],
        generation_config=config
    )
    
    print(f"Config: temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")
    print(f"Response: {generation.text[:200]}...")


async def example_config_overrides_constructor():
    """Example: GenerationConfig parameters override constructor settings."""
    print("\n=== GenerationConfig Overrides Constructor ===")
    
    # Constructor sets provider preferences
    provider = (
        OpenRouterGenerationProvider()
        .order_by_cheapest()  # Constructor preference
        .enable_zdr()
    )
    
    # But GenerationConfig parameters take precedence
    config = GenerationConfig(
        temperature=0.9,  # This overrides any default
        max_output_tokens=1000,
        top_p=0.95,
    )
    
    generation = await provider.generate_async(
        model="anthropic/claude-sonnet-4.5",
        messages=[
            UserMessage(parts=[TextPart(text="Explain quantum entanglement.")])
        ],
        generation_config=config
    )
    
    print("Constructor: order_by_cheapest + ZDR")
    print(f"GenerationConfig: temp={config.temperature}, top_p={config.top_p}")
    print(f"Result: GenerationConfig parameters are used in the request")
    print(f"Response: {generation.text[:200]}...")


async def example_streaming_with_fallbacks():
    """Example: Streaming with fallback models as parameter."""
    print("\n=== Streaming with Fallback Parameter ===")
    
    provider = OpenRouterGenerationProvider()
    
    print("Streaming with fallbacks...")
    async for generation in provider.stream_async(
        model="openai/gpt-4o",
        messages=[
            UserMessage(parts=[TextPart(text="Count from 1 to 5 slowly.")])
        ],
        fallback_models=["anthropic/claude-3.5-sonnet"],
        generation_config=GenerationConfig(temperature=0.5)
    ):
        if generation.text:
            print(generation.text, end="", flush=True)
    
    print(f"\n\nUsed model: {generation.model}")


async def example_combined_features():
    """Example: Combining all features together."""
    print("\n=== Combined Features ===")
    
    provider = (
        OpenRouterGenerationProvider()
        .order_by_fastest()
        .enable_zdr()
        .set_max_price(prompt=5.0, completion=10.0)
    )
    
    config = GenerationConfig(
        temperature=0.8,
        max_output_tokens=800,
        top_p=0.92,
        top_k=50.0,
    )
    
    generation = await provider.generate_async(
        model="anthropic/claude-sonnet-4.5",
        messages=[
            UserMessage(parts=[TextPart(text="Explain the benefits of async programming.")])
        ],
        generation_config=config,
        fallback_models=[
            "openai/gpt-4o",
            "google/gemini-2.5-flash-preview-09-2025"
        ]
    )
    
    print(f"Provider: fastest routing, ZDR, price cap")
    print(f"Config: temp={config.temperature}, top_k={config.top_k}")
    print(f"Fallbacks: gpt-4o, gemini-2.5-flash")
    print(f"Used model: {generation.model}")
    print(f"Response: {generation.text[:200]}...")
    
    # Show pricing
    input_price = await provider.price_per_million_tokens_input(generation.model)
    output_price = await provider.price_per_million_tokens_output(generation.model)
    
    estimated_cost = (
        (generation.usage.prompt_tokens * input_price / 1_000_000) +
        (generation.usage.completion_tokens * output_price / 1_000_000)
    )
    
    print(f"\nUsage: {generation.usage.prompt_tokens} in, {generation.usage.completion_tokens} out")
    print(f"Estimated cost: ${estimated_cost:.6f}")


async def main():
    """Run all examples."""
    print("OpenRouter Advanced Features Examples")
    print("=" * 60)
    
    # Note: Uncomment the examples you want to run
    # Make sure OPENROUTER_API_KEY is set in your environment
    
    # await example_dynamic_pricing()
    # await example_fallback_as_parameter()
    # await example_generation_config_full()
    # await example_config_overrides_constructor()
    # await example_streaming_with_fallbacks()
    # await example_combined_features()
    
    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
