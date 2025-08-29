from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.parsing.parsers.file_parser import FileParser 


agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    static_knowledge=["examples/curriculum.pdf"],
    document_parser=FileParser(),
    instructions="""Você é uma assistente de IA responsável por responder perguntas e conversar, de maneira educada, sobre o Arthur.""",
)

print(agent.run("Boa noite. quem é o arthur").pretty_formatted())
