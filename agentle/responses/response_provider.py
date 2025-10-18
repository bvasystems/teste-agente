from typing import List, Optional, Union
from agentle.responses.definitions.include_enum import IncludeEnum
from agentle.responses.definitions.input_item import InputItem
from agentle.responses.definitions.response import Response
from agentle.responses.definitions.response_stream_options import ResponseStreamOptions
from agentle.responses.definitions.conversation_param import ConversationParam
import abc


class ResponseProvider(abc.ABC):
    async def respond_async[TextFormatT](
        self,
        input: Optional[Union[str, List[InputItem]]] = None,
        include: Optional[List[IncludeEnum]] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: Optional[bool] = None,
        instructions: Optional[str] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[ResponseStreamOptions] = None,
        conversation: Optional[Union[str, ConversationParam]] = None,
        text_format: type[TextFormatT] | None = None,
    ) -> Response[TextFormatT]:
        raise NotImplementedError
