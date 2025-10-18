"""
Example demonstrating the Tool Leakage Validator.

This example shows how to prevent the AI from leaking internal tool definitions
or hallucinating tool call syntax in its responses.
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.google import GoogleGenerationProvider
from agentle.guardrails.validators.tool_leakage_validator import ToolLeakageValidator
from agentle.guardrails.core.guardrail_config import GuardrailConfig
from agentle.generations.tools.tool import Tool


# Define some example tools
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72°F."


def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the internal database."""
    return [{"id": 1, "title": "Result 1"}]


def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to a user."""
    return True


def example_basic_tool_leakage_detection():
    """Example: Detect and block tool leakage"""
    print("\n=== Example 1: Basic Tool Leakage Detection ===\n")

    # Create agent with tools
    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant. Never expose internal tool names or function signatures.",
        tools=[get_weather, search_database, send_email],
        # Add tool leakage validator
        output_guardrails=[
            ToolLeakageValidator(
                priority=5,  # Run early
                enabled=True,
                block_on_detection=True,  # Block responses with leakage
                redact_leakage=False,  # Don't redact, just block
            ),
        ],
        guardrail_config=GuardrailConfig(
            fail_on_output_violation=True,  # Fail if tool leakage detected
            log_violations=True,
        ),
    )

    # This should work fine
    result = agent.run("What's the weather like in Tokyo?")
    print(f"✓ Safe response: {result.generation.text[:100]}...")

    # Simulate AI hallucinating tool syntax (in practice, this would come from the model)
    # Note: In real usage, the model might accidentally output this
    print("\nIf the AI were to output tool syntax, it would be blocked.")


def example_with_tools():
    """Example: Pass tools directly (recommended approach)"""
    print("\n=== Example 2: Pass Tools Directly ===\n")

    # Define the tools you want to use
    my_tools = [get_weather, search_database, send_email]

    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        tools=my_tools,
        output_guardrails=[
            ToolLeakageValidator(
                priority=5,
                enabled=True,
                # Pass the actual tool callables - names will be extracted automatically!
                tools=my_tools,
                block_on_detection=True,
                redact_leakage=False,
            ),
        ],
        guardrail_config=GuardrailConfig(
            fail_on_output_violation=True,
            log_violations=True,
        ),
    )

    result = agent.run("How can you help me?")
    print(f"Response: {result.generation.text[:150]}...")
    print(f"✓ Monitoring tools: {[tool.__name__ for tool in my_tools]}")


def example_redact_instead_of_block():
    """Example: Redact tool leakage instead of blocking"""
    print("\n=== Example 3: Redact Tool Leakage ===\n")

    my_tools = [get_weather, search_database]

    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        tools=my_tools,
        output_guardrails=[
            ToolLeakageValidator(
                priority=5,
                enabled=True,
                tools=my_tools,  # Pass tools directly
                block_on_detection=False,  # Don't block
                redact_leakage=True,  # Redact instead
            ),
        ],
        guardrail_config=GuardrailConfig(
            fail_on_output_violation=False,  # Don't fail, just modify
            log_violations=True,
        ),
    )

    result = agent.run("What tools do you have access to?")
    print(f"Response (with redaction if needed): {result.generation.text[:200]}...")


def example_with_tool_objects():
    """Example: Works with Tool objects too"""
    print("\n=== Example 4: Using Tool Objects ===\n")

    # Convert callables to Tool objects
    tool_objects = [
        Tool.from_callable(get_weather),
        Tool.from_callable(search_database),
        Tool.from_callable(send_email),
    ]

    print(f"Monitoring tools: {[tool.name for tool in tool_objects]}")

    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        tools=tool_objects,
        output_guardrails=[
            ToolLeakageValidator(
                priority=5,
                enabled=True,
                tools=tool_objects,  # Works with Tool objects too!
                block_on_detection=True,
            ),
        ],
        guardrail_config=GuardrailConfig(
            fail_on_output_violation=True,
            log_violations=True,
        ),
    )

    result = agent.run("What can you do?")
    print(f"Response: {result.generation.text[:150]}...")


def example_patterns_detected():
    """Example: Show what patterns are detected"""
    print("\n=== Example 5: Detected Patterns ===\n")

    validator = ToolLeakageValidator(tool_names=["get_weather", "search_database"])

    # Examples of text that would be flagged
    test_cases = [
        '{"get_weather": {"location": "Tokyo"}}',
        "def get_weather(location: str) -> str:",
        'Tool: search_database, Parameters: {"query": "test"}',
        "I will call the get_weather function",
        "function search_database(query: string) -> array",
    ]

    print("Patterns that would be detected:\n")
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test}")

    print("\n✓ All these patterns would be caught by the validator")


def example_with_streaming():
    """Example: Streaming responses with guardrails"""
    print("\n=== Example 6: Streaming with Guardrails ===\n")

    my_tools = [get_weather]

    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        tools=my_tools,
        output_guardrails=[
            ToolLeakageValidator(
                priority=5,
                enabled=True,
                tools=my_tools,  # Pass tools directly
                block_on_detection=False,
                redact_leakage=True,
            ),
        ],
        guardrail_config=GuardrailConfig(
            fail_on_output_violation=False,
            log_violations=True,
        ),
    )

    print("Streaming response (with tool leakage protection):")
    for chunk in agent.run("Tell me about weather services.", stream=True):
        if chunk.generation and chunk.generation.text:
            print(chunk.generation.text, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("🛡️  Tool Leakage Validator Examples")
    print("=" * 60)

    try:
        example_basic_tool_leakage_detection()
    except Exception as e:
        print(f"Example 1 error: {e}")

    try:
        example_with_tools()
    except Exception as e:
        print(f"Example 2 error: {e}")

    try:
        example_redact_instead_of_block()
    except Exception as e:
        print(f"Example 3 error: {e}")

    try:
        example_with_tool_objects()
    except Exception as e:
        print(f"Example 4 error: {e}")

    try:
        example_patterns_detected()
    except Exception as e:
        print(f"Example 5 error: {e}")

    try:
        example_with_streaming()
    except Exception as e:
        print(f"Example 6 error: {e}")

    print("\n" + "=" * 60)
    print("✓ Examples completed!")
