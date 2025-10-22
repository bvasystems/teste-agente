"""OpenRouter pricing service with dynamic pricing from API."""

import logging
from typing import Any

import httpx
from rsb.models.base_model import BaseModel

from agentle.responses.pricing.modality import Modality

logger = logging.getLogger(__name__)


class OpenRouterPricingService(BaseModel):
    """
    OpenRouter implementation of PricingService using dynamic pricing from API.

    This service fetches pricing information from OpenRouter's /models API endpoint
    and caches it for performance. Pricing is fetched lazily on first request.

    The OpenRouter API returns pricing per token, which is converted to per million tokens
    for consistency with other pricing services.

    Attributes:
        api_key: OpenRouter API key for authentication
        base_url: Base URL for OpenRouter API (defaults to https://openrouter.ai/api/v1)
        http_client: Optional custom HTTP client for requests
        _models_cache: Internal cache of model pricing data
    """

    type: str = "openrouter"
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    http_client: httpx.AsyncClient | None = None
    _models_cache: dict[str, dict[str, Any]] | None = None

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        http_client: httpx.AsyncClient | None = None,
    ):
        """
        Initialize the OpenRouter pricing service.

        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
            base_url: Base URL for OpenRouter API
            http_client: Optional custom HTTP client for requests
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.http_client = http_client
        self._models_cache = None

    async def _fetch_models(self) -> dict[str, dict[str, Any]]:
        """
        Fetch available models from OpenRouter API and cache them.

        Returns:
            Dictionary mapping model IDs to model information including pricing

        Raises:
            Exception: If API request fails
        """
        if self._models_cache is not None:
            return self._models_cache

        # Get API key from instance or environment
        from os import getenv

        _api_key = self.api_key or getenv("OPENROUTER_API_KEY")
        if not _api_key:
            logger.warning(
                "No OpenRouter API key provided, pricing will not be available"
            )
            self._models_cache = {}
            return self._models_cache

        headers = {
            "Authorization": f"Bearer {_api_key}",
            "Content-Type": "application/json",
        }

        client = self.http_client or httpx.AsyncClient()

        try:
            response = await client.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            models_response = response.json()
            self._models_cache = {
                model["id"]: model for model in models_response.get("data", [])
            }

            logger.debug(
                f"Fetched pricing for {len(self._models_cache)} models from OpenRouter"
            )

            return self._models_cache
        except Exception as e:
            logger.warning(f"Failed to fetch models from OpenRouter: {e}")
            # Return empty cache on failure
            self._models_cache = {}
            return self._models_cache
        finally:
            if self.http_client is None:
                await client.aclose()

    async def get_input_price_per_million(
        self,
        model: str,
        modality: Modality = "text",
        cached: bool = False,
    ) -> float | None:
        """
        Get the input token price per million tokens for a given model.

        Fetches pricing from OpenRouter's /models API endpoint and converts
        from per-token to per-million-tokens pricing.

        Args:
            model: The model identifier (e.g., "anthropic/claude-3-opus")
            modality: The type of input ("text", "image", "audio", "video")
                     Note: OpenRouter primarily uses "text" modality for prompt pricing
            cached: Whether this is cached input (for models that support caching)
                   Note: OpenRouter has input_cache_read/write pricing for some models

        Returns:
            Price per million input tokens in USD, or None if pricing is unknown
        """
        try:
            models = await self._fetch_models()

            if model not in models:
                logger.debug(
                    f"OpenRouter model '{model}' not found in models list. Available models: {len(models)}"
                )
                return None

            model_info = models[model]
            pricing = model_info.get("pricing", {})

            # Handle cached input pricing if requested
            if cached:
                # Check for input_cache_read pricing (for prompt caching)
                input_cache_read = pricing.get("input_cache_read")
                if input_cache_read is not None:
                    # Convert string to float if needed
                    if isinstance(input_cache_read, str):
                        try:
                            input_cache_read = float(input_cache_read)
                        except ValueError:
                            logger.warning(
                                f"Could not parse input_cache_read price '{input_cache_read}' for model {model}"
                            )
                            return None

                    # OpenRouter returns price per token, convert to per million
                    return float(input_cache_read) * 1_000_000

            # Get standard prompt pricing
            prompt_price = pricing.get("prompt", 0.0)

            # Convert string prices to float if needed
            if isinstance(prompt_price, str):
                try:
                    prompt_price = float(prompt_price)
                except ValueError:
                    logger.warning(
                        f"Could not parse prompt price '{prompt_price}' for model {model}"
                    )
                    return None

            # OpenRouter returns price per token, convert to price per million tokens
            return float(prompt_price) * 1_000_000

        except Exception as e:
            logger.error(
                f"Error fetching input pricing for model {model}: {e}. Returning None"
            )
            return None

    async def get_output_price_per_million(
        self, model: str, modality: Modality = "text"
    ) -> float | None:
        """
        Get the output token price per million tokens for a given model.

        Fetches pricing from OpenRouter's /models API endpoint and converts
        from per-token to per-million-tokens pricing.

        Args:
            model: The model identifier (e.g., "anthropic/claude-3-opus")
            modality: The type of output ("text", "image", "audio", "video")
                     Note: OpenRouter primarily uses "text" modality for completion pricing

        Returns:
            Price per million output tokens in USD, or None if pricing is unknown
        """
        try:
            models = await self._fetch_models()

            if model not in models:
                logger.debug(
                    f"OpenRouter model '{model}' not found in models list. Available models: {len(models)}"
                )
                return None

            model_info = models[model]
            pricing = model_info.get("pricing", {})
            completion_price = pricing.get("completion", 0.0)

            # Convert string prices to float if needed
            if isinstance(completion_price, str):
                try:
                    completion_price = float(completion_price)
                except ValueError:
                    logger.warning(
                        f"Could not parse completion price '{completion_price}' for model {model}"
                    )
                    return None

            # OpenRouter returns price per token, convert to price per million tokens
            return float(completion_price) * 1_000_000

        except Exception as e:
            logger.error(
                f"Error fetching output pricing for model {model}: {e}. Returning None"
            )
            return None

    def clear_cache(self) -> None:
        """
        Clear the cached model pricing data.

        Useful for forcing a refresh of pricing information from the API.
        """
        self._models_cache = None
        logger.debug("Cleared OpenRouter pricing cache")
