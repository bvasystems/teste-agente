from agentle.agents.agent import Agent

from agentle.generations.providers.google import (
    GoogleGenerationProvider,
)

streamlit_app = Agent(
    generation_provider=GoogleGenerationProvider(),
    instructions="Você é um especialista que analisa currículos "
    + "e retorna as informações mais relevantes sobre eles.",
).to_streamlit()

if __name__ == "__main__":
    streamlit_app()
