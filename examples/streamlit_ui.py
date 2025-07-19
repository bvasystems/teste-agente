from agentle.agents.agent import Agent

from agentle.agents.ui.streamlit import AgentToStreamlit
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)

travel_agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.5-flash",
    instructions="""Você é um especialista que analisa currículos e retorna as informações mais relevantes sobre eles.""",
)

streamlit_app = AgentToStreamlit(
    title="Analisador de currículos",
    description="Ask me anything about travel destinations and planning!",
    initial_mode="presentation",
).adapt(travel_agent)

if __name__ == "__main__":
    streamlit_app()
