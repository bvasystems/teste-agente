# Changelog

## v0.7.6
- BREAKING: support for OpenTelemetry clients. The parameter "tracing_client" was now replaced with "otel_clients", inside the base `GenerationProvider` class. The parameters for Langfuse stays the same, but the class is different. Here is how you can migrate:

### LEGACY WAY:
```python
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create a tracing client
tracing_client = LangfuseObservabilityClient()

# Create an agent with tracing enabled
agent = Agent(
    name="Traceable Agent",
    generation_provider=GoogleGenerationProvider(tracing_client=tracing_client),
    model="gemini-2.5-flash",
    instructions="You are a helpful assistant.",
    # Tracing is automatically enabled
)

# Run the agent - tracing happens automatically
response = agent.run(
    "What's the weather in Tokyo?", 
    trace_params={
        "name": "weather_query",
        "user_id": "user123",
        "metadata": {"source": "mobile_app"}
    }
)
```

### New way
```python
from agentle.generations.tracing.langfuse_otel_client import LangfuseOtelClient
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create a tracing client
langfuse = LangfuseOtelClient()

# Create an agent with tracing enabled
agent = Agent(
    name="Traceable Agent",
    generation_provider=GoogleGenerationProvider(otel_clients=[langfuse]), # more clients later
    #or
    #generation_provider=GoogleGenerationProvider(otel_clients=langfuse),
    model="gemini-2.5-flash",
    instructions="You are a helpful assistant.",
    # Tracing is automatically enabled
)

# Run the agent - tracing happens automatically
response = agent.run(
    "What's the weather in Tokyo?", 
    trace_params={
        "name": "weather_query",
        "user_id": "user123",
        "metadata": {"source": "mobile_app"}
    }
)
```

- feat: Added new `performance_metrics` attribute (an instance of `PerformanceMetrics`) to help users identify bottlenecks in Agent execution. This implementation introduces zero overhead to the final response time.

## v0.7.5
- chore: changing `AgentProtocol` to `Agent` in WhatsAppBot class.

## v0.7.4
- fix: making WhatsAppProvider an ABC, instead of protocol. To allow it to be used whitin BaseModels.

## v0.7.3
- chore: making bot and session manager classes more human readable

## v0.7.2
- refactor: changing Agent id to str to allow more flexibility

## v0.7.1
- feat: adding serialization and deserialization to the Agent class.

## v0.7.0

- feat: new `VectorStore` class that represents a single vector store. It can be easily integrated with an Agent of the `Agent` class.

Here is an example of the integration with the Agent class:
```py
from agentle.agents.agent import Agent
from agentle.embeddings.providers.google.google_embedding_provider import (
    GoogleEmbeddingProvider,
)
from agentle.vector_stores.qdrant_vector_store import QdrantVectorStore

curriculum_store = QdrantVectorStore(
    default_collection_name="test_collection",  # important to store in state because the Agent will not know which collection to search.
    embedding_provider=GoogleEmbeddingProvider(
        vertexai=True, project="unicortex", location="global"
    ),
    detailed_agent_description="Stores curriculum information.",
)

curriculum_agent = Agent(vector_stores=[curriculum_store])

agent_response = curriculum_agent.run(
    "I need to know a person that can Lead my AI team. Anyone that might help us?"
    + "Ther person MUST know how to program in COBOL. That is a DISCLASSIFYING requirement."
)

print(agent_response.pretty_formatted())
```

Here is an example of usage of the `VectorStore` class, specifically:
```py
from pprint import pprint

from agentle.embeddings.providers.google.google_embedding_provider import (
    GoogleEmbeddingProvider,
)
from agentle.parsing.parsers.pdf import PDFFileParser

logging.basicConfig(level=logging.DEBUG)

qdrant = QdrantVectorStore(
    embedding_provider=GoogleEmbeddingProvider(
        vertexai=True, project="unicortex", location="global"
    )
)

qdrant.create_collection(
    "test_collection", config={"size": 3072, "distance": "COSINE"}
)

pprint(qdrant.list_collections())

pdf_parser = PDFFileParser()

parsed_file = pdf_parser.parse("curriculum.pdf")

chunk_ids = qdrant.upsert_file(
    parsed_file, collection_name="test_collection", exists_behavior="ignore"
)

pprint(chunk_ids)
```

- feat: `ConversationStore` to provide an easy way to store, in the server-side, the convesation with the agent.

- feat: `chat_id` parameter in `run` and `run_async` methods to use with `ConversationStore`.

- refactor: changed all references of `gemini-2.0-flash` to `gemini-2.5-flash`

- fix: Connecting to `SSEMCPServers` now does not give disconnection errors anymore.

- fix: Connecting to `StdioMCPServers` now 

- refactor: changed `httpx` to `aiohttp` usage in `StreamableHTTPMCPServer`.
