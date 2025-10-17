"""
Example demonstrating the Guardrails system in Agentle.

This example shows how to:
1. Add input guardrails to validate user input
2. Add output guardrails to validate model responses
3. Configure guardrail behavior
4. Handle guardrail violations
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.google import GoogleGenerationProvider
from agentle.guardrails.validators.toxicity_validator import ToxicityValidator
from agentle.guardrails.validators.pii_detection_validator import PIIDetectionValidator
from agentle.guardrails.core.guardrail_config import (
    GuardrailConfig,
    DEVELOPMENT_GUARDRAIL_CONFIG,
    PRODUCTION_GUARDRAIL_CONFIG,
)
from agentle.guardrails.core.guardrail_error import GuardrailViolationError


def example_basic_guardrails():
    """Example: Basic guardrails setup"""
    print("\n=== Example 1: Basic Guardrails ===\n")
    
    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        
        # Add input validators
        input_guardrails=[
            ToxicityValidator(
                priority=10,
                enabled=True,
                toxicity_threshold=0.7  # Block input with toxicity > 0.7
            ),
        ],
        
        # Add output validators
        output_guardrails=[
            ToxicityValidator(
                priority=10,
                enabled=True,
                toxicity_threshold=0.6  # Stricter for outputs
            ),
        ],
        
        # Configure guardrail behavior
        guardrail_config=GuardrailConfig(
            fail_on_input_violation=True,   # Block toxic input
            fail_on_output_violation=False, # Warn on toxic output but allow
            log_violations=True,
            include_metrics=True,
        )
    )
    
    # Test with safe input
    try:
        result = agent.run("Hello! How are you today?")
        print(f"Safe input response: {result.generation.text[:100]}...")
    except GuardrailViolationError as e:
        print(f"Input blocked: {e}")
    
    # Test with toxic input (will be blocked)
    try:
        result = agent.run("You are an idiot and stupid!")
        print(f"Toxic input response: {result.generation.text}")
    except GuardrailViolationError as e:
        print(f"✓ Input blocked by guardrails: {e}")


def example_pii_detection():
    """Example: PII detection and redaction"""
    print("\n=== Example 2: PII Detection ===\n")
    
    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        
        input_guardrails=[
            PIIDetectionValidator(
                priority=20,
                enabled=True,
                redact_pii=True,  # Automatically redact PII
                pii_threshold=0.8
            ),
        ],
        
        guardrail_config=GuardrailConfig(
            fail_on_input_violation=False,  # Don't block, just redact
            log_violations=True,
        )
    )
    
    # Test with PII in input
    result = agent.run(
        "My email is john.doe@example.com and my phone is 555-123-4567"
    )
    print(f"Response (PII should be redacted): {result.generation.text}")


def example_multiple_validators():
    """Example: Using multiple validators"""
    print("\n=== Example 3: Multiple Validators ===\n")
    
    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        
        input_guardrails=[
            ToxicityValidator(priority=10, enabled=True, toxicity_threshold=0.7),
            PIIDetectionValidator(priority=20, enabled=True, redact_pii=True),
        ],
        
        output_guardrails=[
            ToxicityValidator(priority=10, enabled=True, toxicity_threshold=0.5),
            PIIDetectionValidator(priority=20, enabled=True, redact_pii=True),
        ],
        
        guardrail_config=GuardrailConfig(
            fail_on_input_violation=True,
            fail_on_output_violation=False,
            log_violations=True,
            parallel_execution=True,  # Run validators in parallel
            fail_fast=True,           # Stop at first violation
        )
    )
    
    try:
        result = agent.run("Tell me about data privacy best practices.")
        print(f"Response: {result.generation.text[:150]}...")
    except GuardrailViolationError as e:
        print(f"Blocked: {e}")


def example_environment_configs():
    """Example: Using pre-configured environments"""
    print("\n=== Example 4: Environment Configurations ===\n")
    
    # Development mode - lenient, detailed logging
    dev_agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        input_guardrails=[ToxicityValidator()],
        output_guardrails=[ToxicityValidator()],
        guardrail_config=DEVELOPMENT_GUARDRAIL_CONFIG
    )
    
    print("Development mode: Lenient validation, detailed logging")
    result = dev_agent.run("Hello!")
    print(f"Response: {result.generation.text[:50]}...")
    
    # Production mode - strict, optimized
    prod_agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        input_guardrails=[ToxicityValidator()],
        output_guardrails=[ToxicityValidator()],
        guardrail_config=PRODUCTION_GUARDRAIL_CONFIG
    )
    
    print("\nProduction mode: Strict validation, minimal logging")
    result = prod_agent.run("Hello!")
    print(f"Response: {result.generation.text[:50]}...")


def example_custom_config():
    """Example: Custom guardrail configuration"""
    print("\n=== Example 5: Custom Configuration ===\n")
    
    custom_config: GuardrailConfig = {
        "fail_on_input_violation": True,
        "fail_on_output_violation": True,
        "log_violations": True,
        "include_metrics": True,
        "fail_fast": False,           # Check all validators
        "parallel_execution": True,   # Run in parallel for speed
        "cache_enabled": True,
        "max_cache_size": 500,
        "timeout_seconds": 15.0,
    }
    
    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        input_guardrails=[
            ToxicityValidator(priority=10),
            PIIDetectionValidator(priority=20),
        ],
        output_guardrails=[
            ToxicityValidator(priority=10),
        ],
        guardrail_config=custom_config
    )
    
    result = agent.run("What's the weather like?")
    print(f"Response: {result.generation.text[:100]}...")


def example_streaming_with_guardrails():
    """Example: Streaming responses with guardrails"""
    print("\n=== Example 6: Streaming with Guardrails ===\n")
    
    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash-exp",
        instructions="You are a helpful assistant.",
        
        output_guardrails=[
            ToxicityValidator(priority=10, enabled=True, toxicity_threshold=0.6),
        ],
        
        guardrail_config=GuardrailConfig(
            fail_on_output_violation=False,  # Allow but log
            log_violations=True,
        )
    )
    
    print("Streaming response with output validation:")
    for chunk in agent.run("Tell me a short story.", stream=True):
        if chunk.generation and chunk.generation.text:
            print(chunk.generation.text, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("🛡️  Agentle Guardrails Examples")
    print("=" * 50)
    
    # Run all examples
    try:
        example_basic_guardrails()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_pii_detection()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_multiple_validators()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_environment_configs()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_custom_config()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    try:
        example_streaming_with_guardrails()
    except Exception as e:
        print(f"Example 6 error: {e}")
    
    print("\n" + "=" * 50)
    print("✓ Examples completed!")
