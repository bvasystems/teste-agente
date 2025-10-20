from collections.abc import Callable, Sequence
import time
import uuid
from typing import Any, Literal, Optional

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.assistants.assistant_input_type import AssistantInputType
from agentle.prompts.models.prompt import Prompt
from agentle.responses.definitions.conversation_param import ConversationParam
from agentle.responses.definitions.include_enum import IncludeEnum
from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.definitions.response import Response
from agentle.responses.definitions.response_stream_options import ResponseStreamOptions
from agentle.responses.definitions.service_tier import ServiceTier
from agentle.responses.definitions.text import Text
from agentle.responses.definitions.tool import Tool
from agentle.responses.definitions.tool_choice_allowed import ToolChoiceAllowed
from agentle.responses.definitions.tool_choice_custom import ToolChoiceCustom
from agentle.responses.definitions.tool_choice_function import ToolChoiceFunction
from agentle.responses.definitions.tool_choice_mcp import ToolChoiceMCP
from agentle.responses.definitions.tool_choice_options import ToolChoiceOptions

from agentle.responses.definitions.tool_choice_types import ToolChoiceTypes
from agentle.responses.definitions.truncation import Truncation
from agentle.responses.responder import Responder


class Assistant[ResponseSchema = None](BaseModel):
    created_at: int = Field(
        default=int(time.time()),
        description="""The time the assistant was created.""",
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="""The ID of the assistant.""",
    )

    model: str = Field(
        ...,
        description="""ID of the model to use. You can use the List models API
    to see all of your available models, or see our Model
    overview for descriptions of them.""",
    )

    responder: Responder = Field(
        ...,
        description="""The responder to use for the assistant.""",
    )

    description: Optional[str] = Field(
        default=None,
        description="""The description of the assistant. The maximum
        length is 512 characters.""",
    )

    metadata: Optional[dict[str, str]] = Field(
        default=None,
        description="""Set of 16 key-value pairs that can be attached to an object. 
        This can be useful for storing additional information about the object in a 
        structured format, and querying for objects via API or the dashboard. 
        Keys are strings with a maximum length of 64 characters. 
        Values are strings with a maximum length of 512 characters.""",
    )

    name: Optional[str] = Field(
        default=None,
        description="""The name of the assistant. The maximum length is 256 characters.""",
    )

    temperature: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="""What sampling temperature to use, between 0 and 2. 
        Higher values like 0.8 will make the output more random, while lower values 
        like 0.2 will make it more focused and deterministic.""",
    )

    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="""An alternative to sampling with temperature, called nucleus sampling, 
        where the model considers the results of the tokens with top_p probability mass. 
        So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        
        We generally recommend altering this or temperature but not both.""",
    )

    include: Optional[list[IncludeEnum]] = None
    parallel_tool_calls: Optional[bool] = None
    store: Optional[bool] = None
    instructions: Optional[str | Prompt] = None

    stream_options: Optional[ResponseStreamOptions] = None
    conversation: Optional[str | ConversationParam] = None
    text_format: type[ResponseSchema] | None = None
    # ResponseProperties parameters
    previous_response_id: Optional[str] = None
    reasoning: Optional[Reasoning] = None
    background: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    text: Optional[Text] = None
    tools: Optional[Sequence[Tool | Callable[..., Any]]] = None
    tool_choice: Optional[
        ToolChoiceOptions
        | ToolChoiceAllowed
        | ToolChoiceTypes
        | ToolChoiceFunction
        | ToolChoiceMCP
        | ToolChoiceCustom
    ] = None
    prompt: Optional[Prompt] = None
    truncation: Optional[Truncation] = None
    # ModelResponseProperties parameters
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    safety_identifier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    service_tier: Optional[ServiceTier] = None

    async def execute_async(
        self,
        input: AssistantInputType,
        stream: Optional[Literal[False] | Literal[True]] = None,
    ) -> Response[ResponseSchema]: ...
