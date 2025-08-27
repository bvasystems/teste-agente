from __future__ import annotations

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class EvolutionAPIConfig(BaseModel):
    """Configuration for Evolution API."""

    base_url: str = Field(description="Base URL of Evolution API instance")
    instance_name: str = Field(description="Instance name in Evolution API")
    api_key: str = Field(description="API key for authentication")
    webhook_url: str | None = Field(
        default=None, description="Webhook URL for receiving messages"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")

    def clone(
        self,
        new_base_url: str | None = None,
        new_instance_name: str | None = None,
        new_api_key: str | None = None,
        new_webhook_url: str | None = None,
        new_timeout: int | None = None,
    ) -> EvolutionAPIConfig:
        """Clone the configuration with optional new base URL and instance name."""
        return EvolutionAPIConfig(
            base_url=new_base_url or self.base_url,
            instance_name=new_instance_name or self.instance_name,
            api_key=new_api_key or self.api_key,
            webhook_url=new_webhook_url or self.webhook_url,
            timeout=new_timeout or self.timeout,
        )
