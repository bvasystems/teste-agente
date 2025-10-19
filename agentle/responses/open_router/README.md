# OpenRouter Responder - Responses API

Implementation of OpenRouter's new **Responses API (Beta)** for the Agentle framework.

## Overview

The `OpenRouterResponder` class provides access to OpenRouter's Responses API, which is an OpenAI-compatible stateless API that offers:

- ✅ **Basic text generation** (streaming and non-streaming)
- ✅ **Reasoning capabilities** with configurable effort levels
- ✅ **Tool/function calling** with parallel execution support
- ✅ **Web search integration** with citation annotations
- ✅ **Structured output** parsing with Pydantic models

## Key Differences from Chat Completions API

The Responses API differs from the traditional `/chat/completions` endpoint:

1. **Stateless**: Each request is independent; no server-side conversation state
2. **Different format**: Uses `input` instead of `messages` (though both are supported)
3. **Enhanced features**: Built-in reasoning, web search, and more structured tool calling
4. **Event-based streaming**: Uses Server-Sent Events (SSE) with detailed event types

## Installation

The responder is part of the `agentle` package. Ensure you have:

```bash
pip install agentle aiohttp pydantic
```

## Basic Usage

```python
from agentle.responses.open_router.open_router_responder import OpenRouterResponder

# Initialize responder
responder = OpenRouterResponder()  # Uses OPENROUTER_API_KEY env var

# Basic text generation
response = await responder.respond_async(
    input="What is the meaning of life?",
    model="openai/o4-mini",
    max_output_tokens=500,
)

# Access the response
print(response.output_text)  # Convenience property
# or iterate through output items
for item in response.output:
    if item.type == "message":
        for content in item.content:
            if content.type == "output_text":
                print(content.text)
```

## Streaming

The responder returns an `AsyncStream` object for streaming responses, which implements both `AsyncIterator` and `AsyncContextManager` protocols.

### Basic Streaming (AsyncIterator)

```python
# Enable streaming
stream = await responder.respond_async(
    input="Write a short story",
    model="openai/o4-mini",
    stream=True,
)

# Iterate through events
async for event in stream:
    if event.type == "ResponseTextDeltaEvent":
        print(event.delta, end="", flush=True)
    elif event.type == "ResponseCompletedEvent":
        print(f"\nTotal tokens: {event.response.usage.total_tokens}")
```

### Streaming with Context Manager (Recommended)

For automatic cleanup and resource management:

```python
# AsyncStream can be used as a context manager
async with await responder.respond_async(
    input="Write a short story",
    model="openai/o4-mini",
    stream=True,
) as stream:
    async for event in stream:
        if event.type == "ResponseTextDeltaEvent":
            print(event.delta, end="", flush=True)
# Stream is automatically closed when exiting the context
```

## Reasoning

```python
from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.definitions.effort import Effort

response = await responder.respond_async(
    input="Was 1995 30 years ago? Show your reasoning.",
    model="openai/o4-mini",
    reasoning=Reasoning(effort=Effort.high),
)

# Check for reasoning output
for item in response.output:
    if item.type == "reasoning":
        print(f"Reasoning: {item.summary}")
```

## Web Search

```python
response = await responder.respond_async(
    input="What is OpenRouter?",
    model="openai/o4-mini",
    plugins=[{"id": "web", "max_results": 3}],
)

# Access citations
for item in response.output:
    if item.type == "message":
        for content in item.content:
            if content.type == "output_text" and content.annotations:
                for annotation in content.annotations:
                    if annotation.type == "url_citation":
                        print(f"Source: {annotation.url}")
```

## Tool Calling

```python
from agentle.responses.definitions.tool import Tool
from agentle.responses.definitions.function import Function

# Define a tool
weather_tool = Tool(
    type="function",
    function=Function(
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    )
)

response = await responder.respond_async(
    input="What's the weather in Tokyo?",
    model="openai/o4-mini",
    tools=[weather_tool],
    tool_choice="auto",
)

# Check for function calls
for item in response.output:
    if item.type == "function_call":
        print(f"Called: {item.name}({item.arguments})")
```

## Structured Output

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    location: str
    temperature: float
    condition: str

response = await responder.respond_async(
    input="What's the weather in Paris? Return as JSON.",
    model="openai/o4-mini",
    text_format=WeatherInfo,
)

# Access parsed output
weather = response.output_parsed()
if weather:
    print(f"{weather.location}: {weather.temperature}°F, {weather.condition}")
```

## Multi-turn Conversations

Since the API is stateless, you must include full conversation history:

```python
# First turn
response1 = await responder.respond_async(
    input=[
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is 2+2?"}]
        }
    ],
    model="openai/o4-mini",
)

# Second turn - include previous exchange
response2 = await responder.respond_async(
    input=[
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is 2+2?"}]
        },
        response1.output[0],  # Previous assistant message
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What about 3+3?"}]
        }
    ],
    model="openai/o4-mini",
)
```

## Implementation Details

### Architecture

The responder implements the `ResponderMixin` interface and provides:

1. **`_respond_async()`**: Main entry point that handles both streaming and non-streaming
2. **`_handle_non_streaming_response()`**: Parses complete JSON responses
3. **`_stream_events()`**: Async generator for SSE streaming
4. **`_normalize_event_type()`**: Maps OpenRouter event types to internal types

### Event Type Mapping

OpenRouter uses dot-notation event types (e.g., `response.output_text.delta`), which are mapped to our discriminated union types (e.g., `ResponseTextDeltaEvent`).

### Error Handling

- HTTP errors raise `ValueError` with status code and error text
- Malformed JSON events are logged and skipped
- Validation errors are logged but don't crash the stream

## API Reference

See the [OpenRouter Responses API documentation](https://openrouter.ai/docs/api-reference/responses) for:
- Complete parameter reference
- Supported models
- Rate limits
- Error codes

## Examples

See `examples/openrouter_responses_example.py` for comprehensive usage examples.

## Comparison with OpenRouterGenerationProvider

| Feature | OpenRouterGenerationProvider | OpenRouterResponder |
|---------|------------------------------|---------------------|
| API Endpoint | `/chat/completions` | `/responses` |
| State Management | Stateless | Stateless |
| Reasoning | Via model-specific params | Built-in with effort levels |
| Web Search | Plugin-based | Plugin-based |
| Tool Calling | Standard function calling | Enhanced with parallel support |
| Streaming Format | SSE with choices | SSE with detailed events |
| Structured Output | Via response_format | Via text_format |

## Notes

- **Beta API**: This API is in beta and may have breaking changes
- **Stateless**: Always include full conversation history in each request
- **Model Support**: Not all models support all features (reasoning, tools, etc.)
- **Rate Limits**: Standard OpenRouter rate limits apply

## Contributing

When extending this responder:
1. Follow the existing pattern for event handling
2. Add new event types to `_normalize_event_type()` mapping
3. Update examples and documentation
4. Test with both streaming and non-streaming modes
