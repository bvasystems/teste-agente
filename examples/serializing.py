import pprint
from typing import Any

import dill.detect
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentle.agents.a2a.models.agent_skill import AgentSkill
from agentle.agents.a2a.models.authentication import Authentication
from agentle.agents.a2a.models.capabilities import Capabilities
from agentle.agents.agent import Agent
from agentle.agents.agent_config import AgentConfig
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.http_method import HTTPMethod
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.params.integer_param import integer_param
from agentle.agents.conversations.local_conversation_store import LocalConversationStore
from agentle.agents.knowledge.static_knowledge import StaticKnowledge
from agentle.agents.suspension_manager import InMemorySuspensionStore, SuspensionManager
from agentle.embeddings.providers.google.google_embedding_provider import (
    GoogleEmbeddingProvider,
)
from agentle.generations.providers.cerebras.cerebras_generation_provider import (
    CerebrasGenerationProvider,
)
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.generations.providers.ollama.ollama_generation_provider import (
    OllamaGenerationProvider,
)
from agentle.generations.tools.tool import Tool
from agentle.mcp.servers.stdio_mcp_server import StdioMCPServer
from agentle.mcp.servers.streamable_http_mcp_server import StreamableHTTPMCPServer
from agentle.parsing.cache.in_memory_document_cache_store import (
    InMemoryDocumentCacheStore,
)
from agentle.parsing.parsers.pdf import PDFFileParser
from agentle.stt.providers.google.google_speech_to_text_provider import (
    GoogleSpeechToTextProvider,
)
from agentle.vector_stores.qdrant_vector_store import QdrantVectorStore

load_dotenv()


class ExampleResponse(BaseModel):
    response: str | None = Field(default=None)


async def call_me(param: str | float | None = None) -> ExampleResponse:
    print(param)
    return ExampleResponse(response=None)


def sum(a: int, b: int) -> int:
    return a + b


agent = Agent(
    uid="123",
    name="ExampleAgent",
    description="Example agent",
    url="example url",
    static_knowledge=[StaticKnowledge(content="", cache=None, parse_timeout=30)],
    document_parser=PDFFileParser(),
    document_cache_store=InMemoryDocumentCacheStore(),
    generation_provider=GoogleGenerationProvider(),
    file_visual_description_provider=OllamaGenerationProvider(),
    file_audio_description_provider=CerebrasGenerationProvider(),
    version="0.1.0",
    endpoint="localhost",
    documentationUrl="example.com",
    capabilities=Capabilities(
        streaming=False, pushNotifications=False, stateTransitionHistory=False
    ),
    authentication=Authentication(schemes=["GET", "POST"], credentials=None),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[
        AgentSkill(
            id="skill1",
            name="Language Translation",
            description="Translates text between different languages",
            tags=["language", "translation"],
        ),
        AgentSkill(
            id="skill2",
            name="Coconut Translation",
            description="Translates text between different languages",
            tags=["language", "translation"],
        ),
    ],
    model=lambda: "gemini-2.5-flash",
    instructions="Hello, world!",
    response_schema=ExampleResponse,
    mcp_servers=[
        StreamableHTTPMCPServer(
            server_name="Example Server", server_url="example:8923"
        ),
        StdioMCPServer(server_name="Example STDIO", command="npx -y example"),
    ],
    tools=[call_me, Tool.from_callable(sum)],
    config=AgentConfig(maxToolCalls=2, maxIterations=23),
    debug=True,
    suspension_manager=SuspensionManager(store=InMemorySuspensionStore()),
    speech_to_text_provider=GoogleSpeechToTextProvider(
        generation_provider=GoogleGenerationProvider()
    ),
    conversation_store=LocalConversationStore(),
    vector_stores=[
        QdrantVectorStore(
            default_collection_name="example",
            embedding_provider=GoogleEmbeddingProvider(),
        ),
        QdrantVectorStore(
            default_collection_name="example-2",
            embedding_provider=GoogleEmbeddingProvider(),
        ),
    ],
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

# Test serializability
if not dill.pickles(agent):
    # Find problematic objects
    bad_objects = dill.detect.badobjects(agent, depth=0)
    bad_items = dill.detect.baditems(agent)
    errors = dill.detect.errors(agent)
    
    # Save bad objects and items to file
    with open("serialization_issues.txt", "w") as f:
        f.write(f"Bad items: {bad_items}\n")
        f.write(f"Errors: {errors}\n")
    
    print(f"Problematic objects: {bad_objects}")
    print(f"Bad items: {bad_items}")
    print("Serialization issues saved to serialization_issues.txt")

encoded: str = agent.serialize()
print(len(encoded))
decoded_agent: Agent[Any] = Agent.deserialize(encoded)
pprint.pprint(decoded_agent)
