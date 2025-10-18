"""
Simple example: Tool Leakage Validator with tools parameter.

This shows the cleanest way to use the validator - just pass your tools!
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.google import GoogleGenerationProvider
from agentle.guardrails.validators.tool_leakage_validator import ToolLeakageValidator
from agentle.guardrails.core.guardrail_config import GuardrailConfig


# Define your tools
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72°F."


def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the internal database."""
    return [{"id": 1, "title": "Result 1"}]


def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to a user."""
    return True


# Create your agent
my_tools = [get_weather, search_database, send_email]

agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash-exp",
    instructions="You are a helpful assistant. Never expose internal tool information.",
    tools=my_tools,
    # Add the tool leakage validator - just pass the same tools!
    output_guardrails=[
        ToolLeakageValidator(
            tools=my_tools,  # That's it! Names are extracted automatically
            block_on_detection=True,
            redact_leakage=False,
        ),
    ],
    guardrail_config=GuardrailConfig(
        fail_on_output_violation=True,
        log_violations=True,
    ),
)

# Use the agent
if __name__ == "__main__":
    print("🛡️  Simple Tool Leakage Protection\n")

    # This will work fine
    result = agent.run("What's the weather like in Tokyo?")
    print(f"✓ Response: {result.generation.text[:150]}...\n")

    # If the AI accidentally leaks tool info, it will be blocked
    print("✓ Tool leakage protection is active!")
    print(f"✓ Monitoring: {[tool.__name__ for tool in my_tools]}")
