from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)

agent = Agent(
    name="Research Assistant",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.5-flash",
    instructions="You help analysing websites.",
    # Array of string-based knowledge sources (no caching)
    static_knowledge=[
        # URLs as strings
        "https://monowave.store/",
    ],
    debug=True,
)

print(agent.run("What kind of products does monowave have?"))
