# Em: agentle/agents/whatsapp/models/whatsapp_response_base.py

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class WhatsAppResponseBase(BaseModel):
    """
        Base class for WhatsApp bot structured responses.

        This class ensures that all structured outputs from the WhatsApp bot
        contain a 'response' field with the text to be sent to the user.

        Developers can extend this class to add additional structured data
        that they want to extract from the conversation.

        Example:
    ```python
            class CustomerServiceResponse(WhatsAppResponseBase):
                response: str  # Inherited - text to send to user
                sentiment: Literal["happy", "neutral", "frustrated", "angry"]
                urgency: int = Field(ge=1, le=5, description="Urgency level 1-5")
                requires_human: bool = False
                suggested_actions: list[str] = Field(default_factory=list)
    ```
    """

    response: str = Field(
        ...,
        description="The text response that will be sent to the WhatsApp user. This field is required.",
    )
