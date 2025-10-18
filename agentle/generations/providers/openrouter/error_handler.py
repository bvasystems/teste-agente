"""
Error handling utilities for OpenRouter API responses.

This module provides functions to parse OpenRouter error responses and raise
appropriate custom exceptions with detailed, actionable error messages.
"""

import re
from typing import Any

from agentle.generations.providers.openrouter.exceptions import (
    ContextLengthExceededError,
    DataPolicyMismatchError,
    DailyRateLimitExceededError,
    InsufficientCreditsError,
    InternalServerError,
    InvalidCredentialsError,
    InvalidPromptError,
    ModelNotFoundError,
    ModerationError,
    NoAllowedProvidersError,
    NoProvidersAvailableError,
    OpenRouterError,
    PayloadTooLargeError,
    ProviderError,
    ProviderInvalidRequestError,
    RequestTimeoutError,
    UpstreamRateLimitError,
)


def parse_and_raise_openrouter_error(
    status_code: int,
    response_body: dict[str, Any] | None = None,
    response_text: str | None = None,
) -> None:
    """
    Parse an OpenRouter error response and raise the appropriate custom exception.

    Args:
        status_code: HTTP status code from the response
        response_body: Parsed JSON response body (if available)
        response_text: Raw response text (if JSON parsing failed)

    Raises:
        Appropriate OpenRouterError subclass based on the error details
    """
    # Extract error information from response body
    error_message = ""
    error_code: str | None = None

    if response_body:
        # OpenRouter typically returns errors in this format:
        # {"error": {"message": "...", "code": "..."}}
        # or {"error": "..."}
        error_data = response_body.get("error", {})

        if isinstance(error_data, dict):
            message_value = error_data.get("message", "")
            error_message = str(message_value) if message_value else ""  # type: ignore[arg-type]
            code_value = error_data.get("code")
            error_code = str(code_value) if code_value else None  # type: ignore[arg-type]
        elif isinstance(error_data, str):
            error_message = error_data
    elif response_text:
        error_message = response_text

    # Normalize for easier matching
    error_message_lower = error_message.lower()

    # ==================== 400 Bad Request ====================
    if status_code == 400:
        # Context length exceeded
        if (
            "context length" in error_message_lower
            or "maximum context" in error_message_lower
        ):
            # Try to extract token numbers from message
            max_tokens = None
            requested_tokens = None

            # Pattern: "maximum context length is X tokens. However, you requested Y tokens"
            match = re.search(
                r"(\d+)\s+tokens.*?requested\s+(\d+)\s+tokens",
                error_message,
                re.IGNORECASE,
            )
            if match:
                max_tokens = int(match.group(1))
                requested_tokens = int(match.group(2))

            raise ContextLengthExceededError(
                max_tokens, requested_tokens, response_body
            )

        # Provider returned error (invalid request)
        if (
            "provider returned error" in error_message_lower
            and "invalid_request" in error_message_lower
        ):
            raise ProviderInvalidRequestError(error_message, response_body)

        # Invalid prompt / malformed request
        if (
            (error_code and "invalid_prompt" in error_code)
            or "invalid" in error_message_lower
            or "malformed" in error_message_lower
        ):
            raise InvalidPromptError(response_body)

        # Generic 400 error
        raise InvalidPromptError(response_body)

    # ==================== 401 Unauthorized ====================
    elif status_code == 401:
        raise InvalidCredentialsError(response_body)

    # ==================== 402 Payment Required ====================
    elif status_code == 402:
        # Try to extract required credits
        required_credits = None
        match = re.search(r"\$?([\d.]+)", error_message)
        if match:
            try:
                required_credits = float(match.group(1))
            except ValueError:
                pass

        raise InsufficientCreditsError(required_credits, response_body)

    # ==================== 403 Forbidden ====================
    elif status_code == 403:
        # Extract moderation reason if available
        moderation_reason = None
        if response_body and "metadata" in response_body:
            moderation_reason = str(response_body["metadata"])

        raise ModerationError(moderation_reason, response_body)

    # ==================== 404 Not Found ====================
    elif status_code == 404:
        # No endpoints found for model
        if "no endpoints found for model" in error_message_lower:
            # Try to extract model name
            model_id = None
            match = re.search(r"model[:\s]+([^\s]+)", error_message, re.IGNORECASE)
            if match:
                model_id = match.group(1)

            raise ModelNotFoundError(model_id, response_body)

        # Data policy mismatch
        if (
            "data policy" in error_message_lower
            or "privacy settings" in error_message_lower
        ):
            raise DataPolicyMismatchError(response_body)

        # No allowed providers
        if (
            "no allowed providers" in error_message_lower
            or "allowed providers" in error_message_lower
        ):
            raise NoAllowedProvidersError(response_body)

        # Generic 404 - assume model not found
        raise ModelNotFoundError(None, response_body)

    # ==================== 408 Request Timeout ====================
    elif status_code == 408:
        raise RequestTimeoutError(response_body)

    # ==================== 413 Payload Too Large ====================
    elif status_code == 413:
        raise PayloadTooLargeError(response_body)

    # ==================== 429 Rate Limit ====================
    elif status_code == 429:
        # Daily rate limit for free models
        if (
            "free-models-per-day" in error_message_lower
            or "daily" in error_message_lower
        ):
            # Try to extract reset time
            reset_time = None
            if response_body and "reset_at" in response_body:
                reset_time = str(response_body["reset_at"])

            raise DailyRateLimitExceededError(reset_time, response_body)

        # Upstream rate limit
        if (
            "upstream" in error_message_lower
            or "temporarily rate-limited" in error_message_lower
        ):
            # Try to extract model name
            model_id = None
            match = re.search(r"\[([^\]]+)\]", error_message)
            if match:
                model_id = match.group(1)

            raise UpstreamRateLimitError(model_id, response_body)

        # Generic rate limit
        raise DailyRateLimitExceededError(None, response_body)

    # ==================== 500 Internal Server Error ====================
    elif status_code == 500:
        raise InternalServerError(response_body)

    # ==================== 502 Bad Gateway ====================
    elif status_code == 502:
        # Provider error
        provider_message = error_message if error_message else None
        raise ProviderError(provider_message, response_body)

    # ==================== 503 Service Unavailable ====================
    elif status_code == 503:
        # Try to extract model name
        model_id = None
        if response_body and "model" in response_body:
            model_id = response_body["model"]

        raise NoProvidersAvailableError(model_id, response_body)

    # ==================== Unknown Error ====================
    else:
        # Fallback to generic error
        message = f"OpenRouter API Error (HTTP {status_code}): {error_message or 'Unknown error'}"
        raise OpenRouterError(message, status_code, error_code, response_body)
