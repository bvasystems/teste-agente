import os
from typing import Literal, Type, override

from pydantic import Field

import aiohttp
from agentle.responses._streaming.async_stream import AsyncStream
from agentle.responses.definitions.create_response import CreateResponse
from agentle.responses.definitions.response import Response
from agentle.responses.definitions.response_stream_event import ResponseStreamEvent
from agentle.responses.responder_mixin import ResponderMixin

# To use with streaming + structured outputs
from agentle.utils.describe_model_for_llm import describe_model_for_llm
from agentle.utils.make_fields_optional import make_fields_optional
from agentle.utils.parse_streaming_json import parse_streaming_json


class OpenRouterResponder(ResponderMixin):
    type: Literal["openrouter"] = Field("openrouter")
    api_key: str | None = Field(default=None)

    @override
    async def _respond_async[TextFormatT](
        self,
        create_response: CreateResponse,
        text_format: Type[TextFormatT] | None = None,
    ) -> Response[TextFormatT] | AsyncStream[ResponseStreamEvent]:
        _api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not _api_key:
            raise ValueError("No API key provided")


# Example regular response
# {
#   "id": "resp_67ccd7eca01881908ff0b5146584e408072912b2993db808",
#   "object": "response",
#   "created_at": 1741477868,
#   "status": "completed",
#   "error": null,
#   "incomplete_details": null,
#   "instructions": null,
#   "max_output_tokens": null,
#   "model": "o1-2024-12-17",
#   "output": [
#     {
#       "type": "message",
#       "id": "msg_67ccd7f7b5848190a6f3e95d809f6b44072912b2993db808",
#       "status": "completed",
#       "role": "assistant",
#       "content": [
#         {
#           "type": "output_text",
#           "text": "The classic tongue twister...",
#           "annotations": []
#         }
#       ]
#     }
#   ],
#   "parallel_tool_calls": true,
#   "previous_response_id": null,
#   "reasoning": {
#     "effort": "high",
#     "summary": null
#   },
#   "store": true,
#   "temperature": 1.0,
#   "text": {
#     "format": {
#       "type": "text"
#     }
#   },
#   "tool_choice": "auto",
#   "tools": [],
#   "top_p": 1.0,
#   "truncation": "disabled",
#   "usage": {
#     "input_tokens": 81,
#     "input_tokens_details": {
#       "cached_tokens": 0
#     },
#     "output_tokens": 1035,
#     "output_tokens_details": {
#       "reasoning_tokens": 832
#     },
#     "total_tokens": 1116
#   },
#   "user": null,
#   "metadata": {}
# }


# Example streaming response:
# event: response.created
# data: {"type":"response.created","response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"in_progress","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}

# event: response.in_progress
# data: {"type":"response.in_progress","response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"in_progress","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}

# event: response.output_item.added
# data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","type":"message","status":"in_progress","role":"assistant","content":[]}}

# event: response.content_part.added
# data: {"type":"response.content_part.added","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

# event: response.output_text.delta
# data: {"type":"response.output_text.delta","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"delta":"Hi"}

# ...

# event: response.output_text.done
# data: {"type":"response.output_text.done","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"text":"Hi there! How can I assist you today?"}

# event: response.content_part.done
# data: {"type":"response.content_part.done","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}}

# event: response.output_item.done
# data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}}

# event: response.completed
# data: {"type":"response.completed","response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"completed","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[{"id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":{"input_tokens":37,"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":48},"user":null,"metadata":{}}}
