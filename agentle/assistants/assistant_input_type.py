from typing import Optional

from agentle.prompts.models.prompt import Prompt
from agentle.responses.definitions.input_item import InputItem


AssistantInputType = Optional[str | list[InputItem] | Prompt]
