import os
from typing import Literal, override

from pydantic import Field

# import aiohttp
from agentle.responses._streaming.async_stream import AsyncStream
from agentle.responses.definitions.create_response import CreateResponse
from agentle.responses.definitions.response import Response
from agentle.responses.definitions.response_stream_event import ResponseStreamEvent
from agentle.responses.responder_mixin import ResponderMixin


class OpenRouterResponder(ResponderMixin):
    type: Literal["openrouter"] = Field("openrouter")
    api_key: str | None = Field(default=None)

    @override
    async def _respond_async[TextFormatT](
        self, create_response: CreateResponse[TextFormatT]
    ) -> Response[TextFormatT] | AsyncStream[ResponseStreamEvent]:
        _api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not _api_key:
            raise ValueError("No API key provided")
