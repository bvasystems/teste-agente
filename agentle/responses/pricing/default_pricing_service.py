"""Default pricing service with static pricing data."""

from typing import Any
from rsb.models.base_model import BaseModel

from agentle.responses.pricing.modality import Modality

# Structure: {model: {modality: {"input": float, "cached_input": float, "output": float}}}
MODEL_PRICING: dict[str, dict[str, dict[str, Any]]] = {
    # OpenAI GPT-5 Models (Text)
    "gpt-5": {
        "text": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    },
    "gpt-5-mini": {
        "text": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    },
    "gpt-5-nano": {
        "text": {"input": 0.05, "cached_input": 0.005, "output": 0.4},
    },
    # OpenAI GPT-4.1 Models
    "gpt-4.1": {
        "text": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
    },
    "gpt-4.1-mini": {
        "text": {"input": 0.4, "cached_input": 0.1, "output": 1.6},
    },
    # OpenAI GPT-4o Models
    "gpt-4o": {
        "text": {"input": 2.5, "cached_input": 1.25, "output": 10.0},
    },
    "gpt-4o-2024-05-13": {
        "text": {"input": 5.0, "output": 15.0},
    },
    "gpt-4o-2024-08-06": {
        "text": {"input": 2.5, "cached_input": 1.25, "output": 10.0},
    },
    "gpt-4o-mini": {
        "text": {"input": 0.15, "cached_input": 0.075, "output": 0.6},
    },
    "gpt-4o-mini-2024-07-18": {
        "text": {"input": 0.15, "output": 0.6},
    },
    # OpenAI Realtime Models (Text and Audio)
    "gpt-realtime": {
        "text": {"input": 4.0, "cached_input": 0.4, "output": 16.0},
        "audio": {"input": 32.0, "cached_input": 0.4, "output": 64.0},
        "image": {"input": 5.0, "cached_input": 0.5},
    },
    "gpt-realtime-mini": {
        "text": {"input": 0.6, "cached_input": 0.06, "output": 2.4},
        "audio": {"input": 10.0, "cached_input": 0.3, "output": 20.0},
        "image": {"input": 0.8, "cached_input": 0.08},
    },
    "gpt-4o-realtime-preview": {
        "text": {"input": 5.0, "cached_input": 2.5, "output": 20.0},
        "audio": {"input": 40.0, "cached_input": 2.5, "output": 80.0},
    },
    "gpt-4o-mini-realtime-preview": {
        "text": {"input": 0.6, "cached_input": 0.3, "output": 2.4},
        "audio": {"input": 10.0, "cached_input": 0.3, "output": 20.0},
    },
    # OpenAI Audio Models
    "gpt-audio": {
        "text": {"input": 2.5, "output": 10.0},
        "audio": {"input": 32.0, "output": 64.0},
    },
    "gpt-audio-mini": {
        "text": {"input": 0.6, "output": 2.4},
        "audio": {"input": 10.0, "output": 20.0},
    },
    "gpt-4o-audio-preview": {
        "text": {"input": 2.5, "output": 10.0},
        "audio": {"input": 40.0, "output": 80.0},
    },
    "gpt-4o-mini-audio-preview": {
        "text": {"input": 0.15, "output": 0.6},
        "audio": {"input": 10.0, "output": 20.0},
    },
    # OpenAI Image Models
    "gpt-image-1": {
        "text": {"input": 5.0, "cached_input": 1.25},
        "image": {"input": 10.0, "cached_input": 2.5, "output": 40.0},
    },
    "gpt-image-1-mini": {
        "text": {"input": 2.0, "cached_input": 0.2},
        "image": {"input": 2.5, "cached_input": 0.25, "output": 8.0},
    },
    # OpenAI O-series Models
    "o1": {
        "text": {"input": 15.0, "cached_input": 7.5, "output": 60.0},
    },
    "o1-mini": {
        "text": {"input": 1.1, "cached_input": 0.55, "output": 4.4},
    },
    "o1-preview": {
        "text": {"input": 15.0, "output": 60.0},
    },
    "o1-preview-2024-09-12": {
        "text": {"input": 15.0, "output": 60.0},
    },
    "o3": {
        "text": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
    },
    "o3-mini": {
        "text": {"input": 1.1, "cached_input": 0.55, "output": 4.4},
    },
    "o4-mini": {
        "text": {"input": 1.1, "cached_input": 0.275, "output": 4.4},
    },
    # OpenAI GPT-4 Legacy Models
    "gpt-4": {
        "text": {"input": 30.0, "output": 60.0},
    },
    "gpt-4-turbo": {
        "text": {"input": 10.0, "output": 30.0},
    },
    "gpt-4-turbo-preview": {
        "text": {"input": 10.0, "output": 30.0},
    },
    # OpenAI GPT-3.5 Models
    "gpt-3.5-turbo": {
        "text": {"input": 0.5, "output": 1.5},
    },
    "gpt-3.5-turbo-0125": {
        "text": {"input": 0.5, "output": 1.5},
    },
    # Anthropic Claude Models
    "claude-3-opus": {
        "text": {"input": 15.0, "output": 75.0},
    },
    "claude-3-opus-20240229": {
        "text": {"input": 15.0, "output": 75.0},
    },
    "claude-3-sonnet": {
        "text": {"input": 3.0, "output": 15.0},
    },
    "claude-3-sonnet-20240229": {
        "text": {"input": 3.0, "output": 15.0},
    },
    "claude-3-5-sonnet": {
        "text": {"input": 3.0, "output": 15.0},
    },
    "claude-3-5-sonnet-20240620": {
        "text": {"input": 3.0, "output": 15.0},
    },
    "claude-3-5-sonnet-20241022": {
        "text": {"input": 3.0, "output": 15.0},
    },
    "claude-3-haiku": {
        "text": {"input": 0.25, "output": 1.25},
    },
    "claude-3-haiku-20240307": {
        "text": {"input": 0.25, "output": 1.25},
    },
    "claude-2.1": {
        "text": {"input": 8.0, "output": 24.0},
    },
    "claude-2": {
        "text": {"input": 8.0, "output": 24.0},
    },
    "claude-instant-1.2": {
        "text": {"input": 0.8, "output": 2.4},
    },
    # Google Gemini Models (with multimodal support)
    "gemini-1.5-pro": {
        "text": {"input": 1.25, "output": 5.0},
        "image": {"input": 1.25, "output": 5.0},
        "video": {"input": 1.25, "output": 5.0},
        "audio": {"input": 1.25, "output": 5.0},
    },
    "gemini-1.5-pro-latest": {
        "text": {"input": 1.25, "output": 5.0},
        "image": {"input": 1.25, "output": 5.0},
        "video": {"input": 1.25, "output": 5.0},
        "audio": {"input": 1.25, "output": 5.0},
    },
    "gemini-1.5-flash": {
        "text": {"input": 0.075, "output": 0.3},
        "image": {"input": 0.075, "output": 0.3},
        "video": {"input": 0.075, "output": 0.3},
        "audio": {"input": 0.075, "output": 0.3},
    },
    "gemini-1.5-flash-latest": {
        "text": {"input": 0.075, "output": 0.3},
        "image": {"input": 0.075, "output": 0.3},
        "video": {"input": 0.075, "output": 0.3},
        "audio": {"input": 0.075, "output": 0.3},
    },
    "gemini-1.0-pro": {
        "text": {"input": 0.5, "output": 1.5},
    },
    "gemini-pro": {
        "text": {"input": 0.5, "output": 1.5},
    },
    # Meta Llama Models
    "llama-3.1-405b": {
        "text": {"input": 3.0, "output": 3.0},
    },
    "llama-3.1-70b": {
        "text": {"input": 0.88, "output": 0.88},
    },
    "llama-3.1-8b": {
        "text": {"input": 0.2, "output": 0.2},
    },
    "llama-3-70b": {
        "text": {"input": 0.88, "output": 0.88},
    },
    "llama-3-8b": {
        "text": {"input": 0.2, "output": 0.2},
    },
    # Mistral Models
    "mistral-large": {
        "text": {"input": 4.0, "output": 12.0},
    },
    "mistral-large-2": {
        "text": {"input": 4.0, "output": 12.0},
    },
    "mistral-medium": {
        "text": {"input": 2.7, "output": 8.1},
    },
    "mistral-small": {
        "text": {"input": 1.0, "output": 3.0},
    },
    "mistral-tiny": {
        "text": {"input": 0.25, "output": 0.25},
    },
    "mixtral-8x7b": {
        "text": {"input": 0.7, "output": 0.7},
    },
    "mixtral-8x22b": {
        "text": {"input": 1.2, "output": 1.2},
    },
}


class DefaultPricingService(BaseModel):
    """
    Default implementation of PricingService using static pricing data.

    This service uses a predefined dictionary of model prices that supports
    different modalities (text, image, audio, video) and caching features.
    Prices are approximate and based on publicly available pricing information.
    """

    type: str = "default"

    def __init__(
        self,
        custom_pricing: dict[str, dict[str, dict[str, float]]] | None = None,
    ):
        """
        Initialize the default pricing service.

        Args:
            custom_pricing: Optional dictionary to override or extend default pricing.
                           Format: {"model": {"modality": {"input": float, "cached_input": float, "output": float}}}
        """
        super().__init__()
        self.pricing = MODEL_PRICING.copy()
        if custom_pricing:
            # Deep merge custom pricing
            for model, modalities in custom_pricing.items():
                if model in self.pricing:
                    self.pricing[model].update(modalities)
                else:
                    self.pricing[model] = modalities

    async def get_input_price_per_million(
        self,
        model: str,
        modality: Modality = "text",
        cached: bool = False,
    ) -> float | None:
        """
        Get the input token price per million tokens for a given model.

        Args:
            model: The model identifier (e.g., "gpt-4", "claude-3-opus")
            modality: The type of input ("text", "image", "audio", "video")
            cached: Whether this is cached input (for models that support caching)

        Returns:
            Price per million input tokens in USD, or None if pricing is unknown
        """
        model_pricing = self.pricing.get(model)
        if not model_pricing:
            return None

        modality_pricing = model_pricing.get(modality)
        if not modality_pricing:
            return None

        # Try cached input first if requested
        if cached and "cached_input" in modality_pricing:
            return modality_pricing["cached_input"]

        # Fall back to regular input price
        return modality_pricing.get("input")

    async def get_output_price_per_million(
        self, model: str, modality: Modality = "text"
    ) -> float | None:
        """
        Get the output token price per million tokens for a given model.

        Args:
            model: The model identifier (e.g., "gpt-4", "claude-3-opus")
            modality: The type of output ("text", "image", "audio", "video")

        Returns:
            Price per million output tokens in USD, or None if pricing is unknown
        """
        model_pricing = self.pricing.get(model)
        if not model_pricing:
            return None

        modality_pricing = model_pricing.get(modality)
        if not modality_pricing:
            return None

        return modality_pricing.get("output")
