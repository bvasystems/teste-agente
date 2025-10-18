"""
Example demonstrating OpenRouter error handling with detailed, actionable error messages.

This script shows how the OpenRouter provider now raises custom exceptions with:
- Clear descriptions of what went wrong
- Possible causes of the error
- Actionable solutions to fix the issue
"""

import asyncio
from dotenv import load_dotenv

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.openrouter.openrouter_generation_provider import (
    OpenRouterGenerationProvider,
)
from agentle.generations.providers.openrouter.exceptions import (
    ModelNotFoundError,
    InvalidCredentialsError,
    ContextLengthExceededError,
    OpenRouterError,
)

load_dotenv()


async def test_model_not_found():
    """Test error handling for non-existent model."""
    print("\n" + "=" * 80)
    print("TEST 1: Model Not Found Error")
    print("=" * 80)

    provider = OpenRouterGenerationProvider()

    try:
        await provider.generate_async(
            model="this-model-does-not-exist",
            messages=[UserMessage(parts=[TextPart(text="Hello")])],
        )
    except ModelNotFoundError as e:
        print(f"\n✅ Caught ModelNotFoundError:\n{e}")
    except OpenRouterError as e:
        print(f"\n✅ Caught OpenRouterError:\n{e}")


async def test_invalid_credentials():
    """Test error handling for invalid API key."""
    print("\n" + "=" * 80)
    print("TEST 2: Invalid Credentials Error")
    print("=" * 80)

    provider = OpenRouterGenerationProvider(api_key="invalid-key-12345")

    try:
        await provider.generate_async(
            model="openai/gpt-4o-mini",
            messages=[UserMessage(parts=[TextPart(text="Hello")])],
        )
    except InvalidCredentialsError as e:
        print(f"\n✅ Caught InvalidCredentialsError:\n{e}")
    except OpenRouterError as e:
        print(f"\n✅ Caught OpenRouterError:\n{e}")


async def test_context_length():
    """Test error handling for context length exceeded."""
    print("\n" + "=" * 80)
    print("TEST 3: Context Length Exceeded Error")
    print("=" * 80)

    provider = OpenRouterGenerationProvider()

    # Create a very long message to exceed context
    very_long_text = "Hello " * 50000  # This should exceed most model contexts

    try:
        await provider.generate_async(
            model="openai/gpt-4o-mini",
            messages=[UserMessage(parts=[TextPart(text=very_long_text)])],
        )
    except ContextLengthExceededError as e:
        print(f"\n✅ Caught ContextLengthExceededError:\n{e}")
    except OpenRouterError as e:
        print(f"\n✅ Caught OpenRouterError:\n{e}")


async def test_successful_request():
    """Test a successful request (no error)."""
    print("\n" + "=" * 80)
    print("TEST 4: Successful Request (No Error)")
    print("=" * 80)

    provider = OpenRouterGenerationProvider()

    try:
        generation = await provider.generate_async(
            model="openai/gpt-4o-mini",
            messages=[
                UserMessage(parts=[TextPart(text="Say 'Hello, World!' in one word")])
            ],
        )
        print(f"\n✅ Success! Response: {generation.choices[0].message.parts[0].text}")
    except OpenRouterError as e:
        print(f"\n❌ Unexpected error:\n{e}")


async def main():
    """Run all error handling tests."""
    print("\n🧪 OpenRouter Error Handling Examples")
    print("=" * 80)
    print("This demonstrates the new detailed error messages with:")
    print("  • Clear descriptions of what went wrong")
    print("  • Possible causes")
    print("  • Actionable solutions")
    print("=" * 80)

    # Test 1: Model not found
    await test_model_not_found()

    # Test 2: Invalid credentials (uncomment to test)
    # await test_invalid_credentials()

    # Test 3: Context length exceeded (uncomment to test - may take time/credits)
    # await test_context_length()

    # Test 4: Successful request
    await test_successful_request()

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
