from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class RequestConfig(BaseModel):
    """Configuration for HTTP requests."""

    timeout: float = Field(description="Request timeout in seconds", default=30.0)

    max_retries: int = Field(
        description="Maximum number of retries for failed requests", default=3
    )

    retry_delay: float = Field(
        description="Delay between retries in seconds", default=1.0
    )

    follow_redirects: bool = Field(
        description="Whether to follow HTTP redirects", default=True
    )
