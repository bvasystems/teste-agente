"""
Real-world examples using public test APIs to demonstrate the endpoint functionality.

This shows how to integrate with actual public APIs using both simple and complex parameters.
"""

from dotenv import load_dotenv

from agentle.agents.agent import Agent
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.params.integer_param import integer_param
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)

load_dotenv()

facts_agent = Agent(
    name="Fun Facts Assistant",
    generation_provider=GoogleGenerationProvider(
        use_vertex_ai=True, project="unicortex", location="global"
    ),
    model="gemini-2.5-flash",
    instructions="""You are a fun facts assistant. You can provide:
        - Random cat facts and trivia
        - Information about cat breeds
        
        Always make the information engaging and fun!""",
    endpoints=[
        Endpoint(
            name="get_cat_fact",
            description="Get a random cat fact",
            call_condition="when user asks about cats, cat facts, or wants cat trivia",
            url="https://catfact.ninja/fact",
            method=HTTPMethod.GET,
            parameters=[
                integer_param(
                    name="max_length",
                    description="Maximum length of the cat fact",
                    required=False,
                    minimum=1,
                    maximum=1000,
                    location=ParameterLocation.QUERY,
                )
            ],
        ),
        Endpoint(
            name="get_cat_breeds",
            description="Get information about cat breeds",
            call_condition="when user asks about cat breeds or types of cats",
            url="https://catfact.ninja/breeds",
            method=HTTPMethod.GET,
            parameters=[
                integer_param(
                    name="limit",
                    description="Number of breeds to return",
                    required=False,
                    minimum=1,
                    maximum=100,
                    default=10,
                    location=ParameterLocation.QUERY,
                )
            ],
        ),
    ],
)

if __name__ == "__main__":
    response = facts_agent.run("Give me two cat breeds")

    print(f"Cat Facts Response: {response.pretty_formatted()}")
