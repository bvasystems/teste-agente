"""
OpenRouter-specific exceptions with detailed error messages and troubleshooting guidance.

This module provides a comprehensive exception hierarchy for all documented OpenRouter API errors,
including clear descriptions of what went wrong, possible causes, and actionable solutions.
"""

from typing import Any


class OpenRouterError(Exception):
    """Base exception for all OpenRouter API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.response_body = response_body
        super().__init__(message)


# ==================== 400 Bad Request Errors ====================


class OpenRouterBadRequestError(OpenRouterError):
    """Base class for 400 Bad Request errors."""

    pass


class InvalidPromptError(OpenRouterBadRequestError):
    """The request body is malformed, missing required parameters, or contains incorrect data types."""

    def __init__(self, response_body: dict[str, Any] | None = None):
        message = (
            "❌ Invalid Prompt Error\n\n"
            "The request body is malformed, missing required parameters, or contains fields with incorrect data types.\n\n"
            "🔍 Possible Causes:\n"
            "  • Missing 'messages' array in the request\n"
            "  • Incorrect role values (must be 'user', 'assistant', 'system', or 'developer')\n"
            "  • Invalid JSON syntax in the request body\n"
            "  • Incorrect data types for fields (e.g., string instead of number)\n\n"
            "💡 Solutions:\n"
            "  • Validate your request body against the OpenRouter API specification\n"
            "  • Ensure all required fields are present\n"
            "  • Check for JSON syntax errors\n"
            "  • Verify that all field types match the API requirements"
        )
        super().__init__(
            message,
            status_code=400,
            error_code="invalid_prompt",
            response_body=response_body,
        )


class ContextLengthExceededError(OpenRouterBadRequestError):
    """The input exceeds the model's maximum context window."""

    def __init__(
        self,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        if max_tokens and requested_tokens:
            token_info = f"Model supports {max_tokens:,} tokens, but you requested {requested_tokens:,} tokens."
        else:
            token_info = "The input exceeds the model's maximum context length."

        message = (
            f"❌ Context Length Exceeded\n\n"
            f"{token_info}\n\n"
            "🔍 Possible Causes:\n"
            "  • Long conversation histories accumulating over multiple turns\n"
            "  • Large initial prompts or system messages\n"
            "  • Extensive tool outputs or file contents in messages\n"
            "  • Base64-encoded images taking up significant token space\n\n"
            "💡 Solutions:\n"
            "  • Implement token counting on the client side before sending requests\n"
            "  • Truncate older conversation history (keep only recent messages)\n"
            "  • Summarize earlier parts of the conversation\n"
            "  • Use a model with a larger context window\n"
            "  • Compress or reduce the size of file contents and images"
        )
        super().__init__(
            message,
            status_code=400,
            error_code="context_length_exceeded",
            response_body=response_body,
        )


class ProviderInvalidRequestError(OpenRouterBadRequestError):
    """The upstream provider rejected the request as malformed."""

    def __init__(
        self,
        provider_message: str | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        provider_info = (
            f"\n\nProvider Error: {provider_message}" if provider_message else ""
        )

        message = (
            f"❌ Provider Rejected Request{provider_info}\n\n"
            "OpenRouter forwarded your request, but the upstream provider rejected it as malformed.\n\n"
            "🔍 Possible Causes:\n"
            "  • Tool call state management issues (e.g., tool_result without corresponding tool_use)\n"
            "  • Invalid tool_use_id values or mismatched IDs\n"
            "  • Provider-specific parameter constraints violated\n"
            "  • Incorrect message format for the specific provider\n\n"
            "💡 Solutions:\n"
            "  • Review your conversation history and tool call state management\n"
            "  • Ensure tool_use_id values are correctly tracked between messages\n"
            "  • Reset tool call state between independent requests\n"
            "  • Check provider-specific documentation for parameter requirements\n"
            "  • Try a different model/provider to isolate the issue"
        )
        super().__init__(
            message,
            status_code=400,
            error_code="provider_invalid_request",
            response_body=response_body,
        )


# ==================== 401 Unauthorized ====================


class InvalidCredentialsError(OpenRouterError):
    """The API key is missing, malformed, disabled, or expired."""

    def __init__(self, response_body: dict[str, Any] | None = None):
        message = (
            "❌ Invalid Credentials\n\n"
            "The request lacks a valid API key. The key may be missing, malformed, disabled, or expired.\n\n"
            "🔍 Possible Causes:\n"
            "  • The Authorization header is missing\n"
            "  • The API key doesn't start with 'sk-or-v1-'\n"
            "  • Extra spaces or characters in the API key\n"
            "  • The API key has been revoked or expired\n"
            "  • Using the wrong environment variable name\n\n"
            "💡 Solutions:\n"
            "  • Verify your API key in OpenRouter account settings\n"
            "  • Ensure the key is passed correctly: 'Authorization: Bearer YOUR_KEY'\n"
            "  • Check for whitespace or hidden characters in the key\n"
            "  • Generate a new API key if the current one is invalid\n"
            "  • Verify the OPENROUTER_API_KEY environment variable is set correctly"
        )
        super().__init__(
            message,
            status_code=401,
            error_code="invalid_credentials",
            response_body=response_body,
        )


# ==================== 402 Payment Required ====================


class InsufficientCreditsError(OpenRouterError):
    """The account has insufficient credits to process the request."""

    def __init__(
        self,
        required_credits: float | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        credit_info = (
            f"\n\nRequired credits: ${required_credits:.6f}" if required_credits else ""
        )

        message = (
            f"❌ Insufficient Credits{credit_info}\n\n"
            "The cost of processing this request exceeds the available credits in your account.\n\n"
            "🔍 Possible Causes:\n"
            "  • Account has zero or low balance\n"
            "  • Request to a paid model without sufficient funds\n"
            "  • Large request that would exceed available credits\n\n"
            "💡 Solutions:\n"
            "  • Add credits to your OpenRouter account at https://openrouter.ai/credits\n"
            "  • Switch to a free model (model IDs ending with ':free')\n"
            "  • Implement a pre-flight balance check before making requests\n"
            "  • Reduce the size of your requests to lower costs\n"
            "  • Use models with lower per-token pricing"
        )
        super().__init__(
            message,
            status_code=402,
            error_code="insufficient_credits",
            response_body=response_body,
        )


# ==================== 403 Forbidden ====================


class ModerationError(OpenRouterError):
    """The input was flagged by moderation as violating safety policies."""

    def __init__(
        self,
        moderation_reason: str | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        reason_info = (
            f"\n\nModeration Reason: {moderation_reason}" if moderation_reason else ""
        )

        message = (
            f"❌ Content Moderation Violation{reason_info}\n\n"
            "Your input was flagged by the moderation service as potentially violating safety policies.\n\n"
            "🔍 Possible Causes:\n"
            "  • Prompt contains sensitive, harmful, or restricted content\n"
            "  • Content violates the model provider's usage policies\n"
            "  • Input triggers safety filters (violence, hate speech, etc.)\n\n"
            "💡 Solutions:\n"
            "  • DO NOT retry the request with the same content\n"
            "  • Review the moderation metadata to understand the specific issue\n"
            "  • Rephrase your prompt to comply with content policies\n"
            "  • Implement user-facing messaging explaining content restrictions\n"
            "  • Consider using a different model with different moderation policies"
        )
        super().__init__(
            message,
            status_code=403,
            error_code="moderation_violation",
            response_body=response_body,
        )


# ==================== 404 Not Found Errors ====================


class OpenRouterNotFoundError(OpenRouterError):
    """Base class for 404 Not Found errors."""

    pass


class ModelNotFoundError(OpenRouterNotFoundError):
    """The specified model ID does not exist, is deprecated, or is misspelled."""

    def __init__(
        self, model_id: str | None = None, response_body: dict[str, Any] | None = None
    ):
        model_info = f" '{model_id}'" if model_id else ""

        message = (
            f"❌ Model Not Found{model_info}\n\n"
            "The specified model ID does not exist, has been deprecated, or is misspelled.\n\n"
            "🔍 Possible Causes:\n"
            "  • Typo in the model name\n"
            "  • Using an old/deprecated model version\n"
            "  • Model has been removed from OpenRouter\n"
            "  • Incorrect provider prefix (e.g., 'openai/' vs 'anthropic/')\n\n"
            "💡 Solutions:\n"
            "  • Verify the exact model ID at https://openrouter.ai/models\n"
            "  • Check for the latest version of the model\n"
            "  • Use the model search API to find available models\n"
            "  • Review OpenRouter's model deprecation announcements"
        )
        super().__init__(
            message,
            status_code=404,
            error_code="model_not_found",
            response_body=response_body,
        )


class DataPolicyMismatchError(OpenRouterNotFoundError):
    """The user's privacy settings are incompatible with the model's requirements."""

    def __init__(self, response_body: dict[str, Any] | None = None):
        message = (
            "❌ Data Policy Mismatch\n\n"
            "Your privacy settings are incompatible with the requirements of the selected model.\n\n"
            "🔍 Possible Causes:\n"
            "  • Model requires prompt logging for training, but you've disabled it\n"
            "  • Your data retention settings conflict with model requirements\n"
            "  • Privacy settings prevent data sharing with the model provider\n\n"
            "💡 Solutions:\n"
            "  • Go to OpenRouter account settings → Privacy\n"
            "  • Enable 'Allow model training' if required by the model\n"
            "  • Adjust data retention and sharing settings\n"
            "  • Choose a different model with compatible privacy requirements\n"
            "  • Review the model's data policy on its details page"
        )
        super().__init__(
            message,
            status_code=404,
            error_code="data_policy_mismatch",
            response_body=response_body,
        )


class NoAllowedProvidersError(OpenRouterNotFoundError):
    """The routing configuration prevents finding a valid provider for the model."""

    def __init__(self, response_body: dict[str, Any] | None = None):
        message = (
            "❌ No Allowed Providers Available\n\n"
            "Your routing configuration prevents OpenRouter from finding a valid provider for this model.\n\n"
            "🔍 Possible Causes:\n"
            "  • 'Allowed Providers' whitelist doesn't include providers for this model\n"
            "  • All providers for this model are in your 'Ignored Providers' list\n"
            "  • Provider routing preferences are too restrictive\n\n"
            "💡 Solutions:\n"
            "  • Go to OpenRouter account settings → Routing\n"
            "  • Clear the 'Allowed Providers' list to restore default routing\n"
            "  • Remove relevant entries from 'Ignored Providers'\n"
            "  • Check which providers serve your desired model on its details page\n"
            "  • Use a different model that's available through your allowed providers"
        )
        super().__init__(
            message,
            status_code=404,
            error_code="no_allowed_providers",
            response_body=response_body,
        )


# ==================== 408 Request Timeout ====================


class RequestTimeoutError(OpenRouterError):
    """The request took too long to process and was terminated."""

    def __init__(self, response_body: dict[str, Any] | None = None):
        message = (
            "❌ Request Timeout\n\n"
            "Your request took too long to process on the server side and was terminated.\n\n"
            "🔍 Possible Causes:\n"
            "  • Server or model provider is under heavy load\n"
            "  • Long queue times due to high demand\n"
            "  • Network connectivity issues\n"
            "  • Very large or complex request taking too long to process\n\n"
            "💡 Solutions:\n"
            "  • This is a transient error - implement retry with exponential backoff\n"
            "  • DO NOT retry immediately\n"
            "  • Wait 1-5 seconds before first retry, then increase wait time\n"
            "  • Consider using a faster model\n"
            "  • Reduce request complexity or size\n"
            "  • Check OpenRouter status page for ongoing issues"
        )
        super().__init__(
            message,
            status_code=408,
            error_code="request_timeout",
            response_body=response_body,
        )


# ==================== 413 Payload Too Large ====================


class PayloadTooLargeError(OpenRouterError):
    """The request payload exceeds the size limit."""

    def __init__(self, response_body: dict[str, Any] | None = None):
        message = (
            "❌ Payload Too Large\n\n"
            "The request payload exceeds the maximum allowed size.\n\n"
            "🔍 Possible Causes:\n"
            "  • Extremely long conversation history\n"
            "  • Large tool outputs or file contents\n"
            "  • Multiple large base64-encoded images\n"
            "  • Very long system messages or prompts\n\n"
            "💡 Solutions:\n"
            "  • Truncate conversation history (keep only recent messages)\n"
            "  • Summarize older messages instead of including full text\n"
            "  • Reduce image sizes or quality before encoding\n"
            "  • Split large requests into smaller chunks\n"
            "  • Remove unnecessary metadata or verbose tool outputs"
        )
        super().__init__(
            message,
            status_code=413,
            error_code="payload_too_large",
            response_body=response_body,
        )


# ==================== 429 Rate Limit Errors ====================


class RateLimitError(OpenRouterError):
    """Base class for rate limit errors."""

    pass


class DailyRateLimitExceededError(RateLimitError):
    """The account has exceeded its daily request limit for free models."""

    def __init__(
        self, reset_time: str | None = None, response_body: dict[str, Any] | None = None
    ):
        reset_info = (
            f"\n\nRate limit resets at: {reset_time}"
            if reset_time
            else "\n\nRate limit typically resets at 12:00 AM UTC"
        )

        message = (
            f"❌ Daily Rate Limit Exceeded{reset_info}\n\n"
            "Your account has exceeded its daily request limit for free models.\n\n"
            "🔍 Possible Causes:\n"
            "  • Made too many requests to free models today\n"
            "  • Free tier daily quota exhausted\n"
            "  • Failed requests also count toward the limit\n\n"
            "💡 Solutions:\n"
            "  • Wait for the rate limit to reset (typically 12:00 AM UTC)\n"
            "  • Add credits to your account to increase limits\n"
            "  • Switch to paid models (no daily limits with credits)\n"
            "  • Implement request queuing to stay within limits\n"
            "  • Cache responses to reduce redundant requests"
        )
        super().__init__(
            message,
            status_code=429,
            error_code="rate_limit_exceeded",
            response_body=response_body,
        )


class UpstreamRateLimitError(RateLimitError):
    """The model is experiencing high demand at the source provider."""

    def __init__(
        self, model_id: str | None = None, response_body: dict[str, Any] | None = None
    ):
        model_info = f" for {model_id}" if model_id else ""

        message = (
            f"❌ Upstream Rate Limit{model_info}\n\n"
            "The model is experiencing high demand at the source provider.\n\n"
            "🔍 Possible Causes:\n"
            "  • Popular model under heavy global load\n"
            "  • Provider prioritizing direct customers over API aggregators\n"
            "  • Temporary capacity constraints at the provider\n\n"
            "💡 Solutions:\n"
            "  • Switch to a different model temporarily\n"
            "  • Retry after a few minutes with exponential backoff\n"
            "  • Use OpenRouter's fallback models feature\n"
            "  • Check model uptime statistics on OpenRouter\n"
            "  • Consider adding a dedicated provider API key via OpenRouter integrations"
        )
        super().__init__(
            message,
            status_code=429,
            error_code="upstream_rate_limit",
            response_body=response_body,
        )


# ==================== 500 Internal Server Error ====================


class InternalServerError(OpenRouterError):
    """An unexpected error occurred on the OpenRouter servers."""

    def __init__(self, response_body: dict[str, Any] | None = None):
        message = (
            "❌ Internal Server Error\n\n"
            "An unexpected error occurred on the OpenRouter servers.\n\n"
            "🔍 Possible Causes:\n"
            "  • Bug or internal system failure in OpenRouter infrastructure\n"
            "  • Temporary service disruption\n"
            "  • Database or cache issues\n\n"
            "💡 Solutions:\n"
            "  • This is NOT a client-side issue\n"
            "  • Wait and retry with exponential backoff\n"
            "  • Check OpenRouter status page: https://status.openrouter.ai\n"
            "  • Join OpenRouter Discord for real-time updates\n"
            "  • If persistent, report the issue to OpenRouter support"
        )
        super().__init__(
            message,
            status_code=500,
            error_code="server_error",
            response_body=response_body,
        )


# ==================== 502 Bad Gateway ====================


class ProviderError(OpenRouterError):
    """The upstream provider responded with an error or invalid response."""

    def __init__(
        self,
        provider_message: str | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        provider_info = (
            f"\n\nProvider Error: {provider_message}" if provider_message else ""
        )

        message = (
            f"❌ Provider Error{provider_info}\n\n"
            "OpenRouter sent the request to the provider, but the provider responded with an error.\n\n"
            "🔍 Possible Causes:\n"
            "  • Provider's service is down or experiencing an outage\n"
            "  • Temporary network issue between OpenRouter and provider\n"
            "  • Provider returned an invalid or malformed response\n"
            "  • This is a VERY COMMON transient error\n\n"
            "💡 Solutions:\n"
            "  • Check model uptime statistics on OpenRouter website\n"
            "  • Retry the request - OpenRouter may route to a different provider\n"
            "  • Use fallback models to automatically try alternatives\n"
            "  • Switch to a different model temporarily\n"
            "  • Implement exponential backoff retry strategy\n"
            "  • Check provider status pages for known outages"
        )
        super().__init__(
            message,
            status_code=502,
            error_code="provider_error",
            response_body=response_body,
        )


# ==================== 503 Service Unavailable ====================


class NoProvidersAvailableError(OpenRouterError):
    """All providers for the requested model are currently unavailable."""

    def __init__(
        self, model_id: str | None = None, response_body: dict[str, Any] | None = None
    ):
        model_info = f" for {model_id}" if model_id else ""

        message = (
            f"❌ No Providers Available{model_info}\n\n"
            "All potential providers for this model are currently down or unavailable.\n\n"
            "🔍 Possible Causes:\n"
            "  • Widespread outage affecting all providers for this model\n"
            "  • All providers are at capacity\n"
            "  • Maintenance window for the model\n\n"
            "💡 Solutions:\n"
            "  • This indicates a severe availability issue\n"
            "  • Wait 5-15 minutes before retrying\n"
            "  • Check OpenRouter status page for updates\n"
            "  • Switch to a completely different model\n"
            "  • Use OpenRouter's fallback models feature\n"
            "  • Monitor OpenRouter Discord for service updates"
        )
        super().__init__(
            message,
            status_code=503,
            error_code="no_providers_available",
            response_body=response_body,
        )
