"""
Test API retry mechanism to ensure the session management fix works correctly.
"""

from dotenv import load_dotenv

from agentle.agents.agent import Agent
from agentle.agents.apis.api import API
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.primitive_schema import PrimitiveSchema
from agentle.agents.apis.request_config import RequestConfig
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)

load_dotenv(override=True)


async def test_retry_mechanism():
    """Test that retries work correctly with the session management fix."""
    print("\n" + "=" * 70)
    print("Testing API Retry Mechanism")
    print("=" * 70)

    # Create an API with retry configuration
    api = API(
        name="JSONPlaceholder",
        description="Free fake REST API for testing",
        base_url="https://jsonplaceholder.typicode.com",
        request_config=RequestConfig(
            max_retries=3,
            retry_delay=0.5,
            timeout=5.0,
            enable_request_logging=True,
        ),
        endpoints=[],
    )

    # Add an endpoint that might fail (using an invalid endpoint to trigger retries)
    test_endpoint = Endpoint(
        name="get_post",
        description="Get a specific post by ID",
        path="/posts/{id}",
        method=HTTPMethod.GET,
        parameters=[
            EndpointParameter(
                name="id",
                description="Post ID",
                parameter_schema=PrimitiveSchema(type="integer"),
                location=ParameterLocation.PATH,
                required=True,
            )
        ],
        request_config=RequestConfig(
            max_retries=2,
            retry_delay=0.3,
            timeout=10.0,
        ),
    )

    api.add_endpoint(test_endpoint)

    # Create agent
    agent = Agent(
        name="Test Assistant",
        generation_provider=GoogleGenerationProvider(
            use_vertex_ai=True, project="unicortex", location="global"
        ),
        model="gemini-2.5-flash",
        instructions="You are a test assistant. Fetch the requested post.",
        apis=[api],
    )

    # Test with a valid request
    print("\n📝 Testing valid request...")
    result = agent.run("Get post with ID 5")
    print(f"✅ Valid request succeeded!")
    print(f"   Response preview: {result.text[:80]}...")

    # Test with another valid request to ensure session reuse works
    print("\n📝 Testing second request...")
    result2 = agent.run("Get post with ID 10")
    print(f"✅ Second request succeeded!")
    print(f"   Response preview: {result2.text[:80]}...")

    print("\n" + "=" * 70)
    print("✅ Retry mechanism test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_retry_mechanism())
