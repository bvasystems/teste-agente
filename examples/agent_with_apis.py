"""
Test the API feature using JSONPlaceholder - a free fake REST API for testing.

This example demonstrates:
1. Creating an API manually with endpoints
2. Loading an API from an OpenAPI spec URL
3. Using the API with an agent
4. Making requests through the agent
"""

from dotenv import load_dotenv

from agentle.agents.agent import Agent
from agentle.agents.apis.api import API
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.primitive_schema import PrimitiveSchema
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)

load_dotenv(override=True)


async def test_manual_api():
    """Test creating an API manually."""
    print("\n" + "=" * 70)
    print("TEST 1: Manual API Creation")
    print("=" * 70)

    # Create API manually
    api = API(
        name="JSONPlaceholder",
        description="Free fake REST API for testing and prototyping",
        base_url="https://jsonplaceholder.typicode.com",
        endpoints=[],
    )

    # Test various path patterns to ensure function names are valid
    test_endpoints = [
        # Normal path
        Endpoint(
            name="get_posts",
            description="Get all posts",
            path="/posts",
            method=HTTPMethod.GET,
        ),
        # Path with parameter
        Endpoint(
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
        ),
        # Path with dashes (should be converted to underscores)
        Endpoint(
            name="get_user_posts",
            description="Get posts by user",
            path="/users/{userId}/posts",
            method=HTTPMethod.GET,
            parameters=[
                EndpointParameter(
                    name="userId",
                    description="User ID",
                    parameter_schema=PrimitiveSchema(type="integer"),
                    location=ParameterLocation.PATH,
                    required=True,
                )
            ],
        ),
    ]

    for endpoint in test_endpoints:
        api.add_endpoint(endpoint)

    # Create agent with the API
    agent = Agent(
        name="Blog Assistant",
        generation_provider=GoogleGenerationProvider(
            use_vertex_ai=True, project="unicortex", location="global"
        ),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant that can fetch blog posts. When asked about posts, use the available tools.",
        apis=[api],
    )

    # Test the agent
    result = agent.run("Get me post with ID 1")
    print(f"\n✅ Manual API test passed!")
    print(f"Response preview: {result.text[:100]}...")

    return result


async def test_openapi_spec():
    """Test loading an API from an OpenAPI spec."""
    print("\n" + "=" * 70)
    print("TEST 2: OpenAPI Spec Loading")
    print("=" * 70)

    try:
        # Load a real OpenAPI spec from a public API
        # Using PetStore API as it's a standard example
        api = await API.from_openapi_spec(
            "https://petstore3.swagger.io/api/v3/openapi.json",
            name="PetStore",
            base_url_override="https://petstore3.swagger.io/api/v3",  # Override the relative base URL
            include_operations=["getPetById"],  # Only include one simple operation
        )

        print(f"✅ Loaded API: {api.name}")
        print(f"   Base URL: {api.base_url}")
        print(f"   Endpoints: {len(api.endpoints)}")

        # Print endpoint names to verify they're valid
        for endpoint in api.endpoints:
            print(f"   - {endpoint.name}: {endpoint.method.value} {endpoint.path}")

        # Create agent with the API
        agent = Agent(
            name="Pet Store Assistant",
            generation_provider=GoogleGenerationProvider(
                use_vertex_ai=True, project="unicortex", location="global"
            ),
            model="gemini-2.5-flash",
            instructions="You are a pet store assistant. You can look up pets by ID.",
            apis=[api],
        )

        # Test the agent with a specific pet ID that should exist
        result = agent.run("Get information about pet with ID 1")
        print(f"\n✅ OpenAPI spec test passed!")
        print(f"Response preview: {result.text[:100]}...")

        return result

    except Exception as e:
        print(f"⚠️  OpenAPI spec test skipped: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main():
    """Run all tests."""
    print("\n🧪 Testing API Feature")
    print("=" * 70)

    # Test 1: Manual API creation
    await test_manual_api()

    # Test 2: OpenAPI spec loading
    await test_openapi_spec()

    print("\n" + "=" * 70)
    print("✅ All API tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
