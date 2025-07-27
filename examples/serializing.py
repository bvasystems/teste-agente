import pprint
from typing import Any
from uuid import uuid4
from agentle.agents.a2a.models.agent_skill import AgentSkill
from agentle.agents.a2a.models.authentication import Authentication
from agentle.agents.a2a.models.capabilities import Capabilities
from agentle.agents.agent import Agent
from agentle.agents.agent_config import AgentConfig
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
from agentle.mcp.servers.stdio_mcp_server import StdioMCPServer
from agentle.mcp.servers.streamable_http_mcp_server import StreamableHTTPMCPServer
from agentle.parsing.cache.in_memory_document_cache_store import (
    InMemoryDocumentCacheStore,
)
from agentle.parsing.parsers.pdf import PDFFileParser
from pydantic import BaseModel, Field

from agentle.stt.providers.google.google_speech_to_text_provider import (
    GoogleSpeechToTextProvider,
)
from agentle.vector_stores.qdrant_vector_store import QdrantVectorStore


class ExampleResponse(BaseModel):
    response: str | None = Field(default=None)


async def call_me(param: str | float | None = None) -> ExampleResponse:
    print(param)
    return ExampleResponse(response=None)


agent = Agent(
    uid=str(uuid4()),
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
    tools=[call_me],
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
)

encoded: str = agent.serialize()
print(len(encoded))
decoded_agent: Agent[Any] = Agent.deserialize(encoded)
pprint.pprint(agent)
