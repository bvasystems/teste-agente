"""Type definitions for pricing service configuration."""

from typing import Annotated, Union

from pydantic import Field

from agentle.responses.pricing.default_pricing_service import DefaultPricingService
from agentle.responses.pricing.openrouter_pricing_service import (
    OpenRouterPricingService,
)

# Discriminated union type for all pricing service configurations
PricingService = Annotated[
    Union[DefaultPricingService, OpenRouterPricingService],
    Field(discriminator="type"),
]
