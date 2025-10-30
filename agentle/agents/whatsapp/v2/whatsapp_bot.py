from collections.abc import Callable
from typing import Any
from pydantic import ConfigDict
from rsb.models.base_model import BaseModel


class WhatsAppBot(BaseModel):
    webhook_handlers: list[Callable[..., Any]]
    batch_processor_manager: BatchProcessorManager
    model_config = ConfigDict(arbitrary_types_allowed=True)
