"""
Simplified, high-performance @observe decorator for Agentle generation providers using Langfuse V3.

This decorator provides automatic tracing for AI generation methods using the new
Langfuse V3 SDK with OpenTelemetry context managers.
"""

import functools
import inspect
import logging
from datetime import datetime
from typing import Any, Callable, TypeVar, cast, get_args

from langfuse import get_client, Langfuse

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.types.model_kind import ModelKind

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def observe(func: F) -> F:
    """
    High-performance decorator that adds observability to generation methods using Langfuse V3.

    This decorator wraps generation methods to automatically handle tracing using
    Langfuse V3's OpenTelemetry-based context managers.

    Usage:
        ```python
        class MyProvider(GenerationProvider):
            @observe
            async def generate_async(self, ...) -> Generation[T]:
                # Your generation logic here
                return generation
        ```

    Args:
        func: The generation method to decorate

    Returns:
        The wrapped function with automatic tracing
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Generation[Any]:
        # Get the provider instance (self)
        provider = args[0]
        if not isinstance(provider, GenerationProvider) or not getattr(
            provider, "tracing_client", None
        ):
            # Not a provider method, execute without tracing
            return await func(*args, **kwargs)

        # Get Langfuse client
        langfuse = get_client()

        # Extract parameters for tracing
        trace_data = _extract_trace_data(func, provider, args, kwargs)

        # Create trace name
        trace_name = f"{provider.organization}_{trace_data['model']}_generation"

        # Use V3 context manager for automatic tracing
        with langfuse.start_as_current_span(
            name=trace_name,
            input={
                "messages": trace_data["messages"],
                "config": trace_data["config"],
                "model": trace_data["model"],
                "message_count": trace_data["message_count"],
                "has_tools": trace_data["has_tools"],
                "has_schema": trace_data["has_schema"],
            },
            metadata={
                "provider": trace_data["provider"],
                "model": trace_data["model"],
            },
        ) as root_span:
            # Set trace attributes
            root_span.update_trace(
                user_id=trace_data.get("user_id"),
                session_id=trace_data.get("session_id"),
                tags=[trace_data["provider"], trace_data["model"]],
            )

            start_time = datetime.now()

            try:
                # Create nested generation for the actual LLM call
                with langfuse.start_as_current_generation(
                    name=f"{trace_data['provider']}_generation",
                    model=trace_data["model"],
                    input={"message_count": trace_data["message_count"]},
                    metadata=trace_data["config"],
                ) as generation:
                    # Execute the actual generation method
                    response = await func(*args, **kwargs)

                    # Calculate metrics
                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                    # Update generation with results
                    usage_details = None
                    cost_details = None

                    if hasattr(response, "usage") and response.usage:
                        usage = response.usage
                        usage_details = {
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }

                        # Calculate costs
                        try:
                            input_cost = provider.price_per_million_tokens_input(
                                trace_data["model"]
                            ) * (usage.prompt_tokens / 1_000_000)
                            output_cost = provider.price_per_million_tokens_output(
                                trace_data["model"]
                            ) * (usage.completion_tokens / 1_000_000)
                            cost_details = {
                                "input_cost": input_cost,
                                "output_cost": output_cost,
                                "total_cost": input_cost + output_cost,
                            }
                        except Exception:
                            pass  # Cost calculation is optional

                    generation.update(
                        output={
                            "completion": getattr(response, "text", str(response)),
                            "choices_count": len(getattr(response, "choices", [])),
                            "has_tool_calls": len(getattr(response, "tool_calls", []))
                            > 0,
                        },
                        usage_details=usage_details,
                        cost_details=cost_details,
                        metadata={"duration_ms": duration_ms},
                    )

                    # Add scores using V3 methods
                    _add_success_scores(langfuse, trace_data, response, duration_ms)

                    # Update root span with final output
                    root_span.update(
                        output={
                            "completion": getattr(response, "text", str(response)),
                            "success": True,
                            "duration_ms": duration_ms,
                            "total_tokens": usage_details["total_tokens"]
                            if usage_details
                            else 0,
                            "total_cost": cost_details["total_cost"]
                            if cost_details
                            else 0,
                        }
                    )

                    return response

            except Exception as error:
                # Calculate duration for error case
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                error_str = str(error)
                error_type = type(error).__name__

                # Update generation with error
                with langfuse.start_as_current_generation(
                    name=f"{trace_data['provider']}_generation_error",
                    model=trace_data["model"],
                    input={"message_count": trace_data["message_count"]},
                    metadata={
                        **trace_data["config"],
                        "error": True,
                        "error_type": error_type,
                    },
                ) as error_generation:
                    error_generation.update(
                        output={
                            "error": error_str,
                            "error_type": error_type,
                        },
                        metadata={
                            "duration_ms": duration_ms,
                            "error": True,
                            "error_type": error_type,
                        },
                    )

                # Add error scores
                _add_error_scores(langfuse, error, duration_ms)

                # Update root span with error
                root_span.update(
                    output={
                        "error": error_str,
                        "error_type": error_type,
                        "success": False,
                        "duration_ms": duration_ms,
                    }
                )

                # Re-raise the original exception
                raise

    return cast(F, wrapper)


def _extract_trace_data(
    func: Callable[..., Any],
    provider: GenerationProvider,
    args: tuple[Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Extract minimal data needed for tracing."""
    try:
        # Get function signature and bind arguments
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Extract key parameters
        model = bound_args.arguments.get("model") or provider.default_model
        messages = bound_args.arguments.get("messages", [])
        response_schema = bound_args.arguments.get("response_schema")
        generation_config = (
            bound_args.arguments.get("generation_config") or GenerationConfig()
        )
        tools = bound_args.arguments.get("tools")

        # Handle ModelKind mapping
        model_kind_values = get_args(ModelKind)
        if model in model_kind_values:
            model = provider.map_model_kind_to_provider_model(cast(ModelKind, model))

        # Extract user context from trace_params if available
        trace_params = getattr(generation_config, "trace_params", {})
        user_id = trace_params.get("user_id")
        session_id = trace_params.get("session_id")

        return {
            "model": model,
            "provider": provider.organization,
            "message_count": len(messages),
            "has_tools": tools is not None and len(tools) > 0,
            "has_schema": response_schema is not None,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content_length": len("".join(str(part) for part in msg.parts)),
                }
                for msg in messages
            ],
            "config": {
                k: v
                for k, v in generation_config.__dict__.items()
                if not k.startswith("_") and not callable(v) and v is not None
            },
        }

    except Exception as e:
        logger.warning(f"Error extracting trace data: {e}")
        return {
            "model": getattr(provider, "default_model", "unknown"),
            "provider": getattr(provider, "organization", "unknown"),
            "message_count": 0,
            "has_tools": False,
            "has_schema": False,
            "messages": [],
            "config": {},
        }


def _add_success_scores(
    langfuse: Langfuse,
    trace_data: dict[str, Any],
    response: Generation[Any],
    duration_ms: float,
) -> None:
    """Add various success scores to the current trace."""
    try:
        # Success score
        langfuse.score_current_trace(
            name="trace_success",
            value=1.0,
            comment="Generation completed successfully",
        )

        # Latency score
        latency_score = (
            1.0 if duration_ms < 1000 else (0.8 if duration_ms < 3000 else 0.6)
        )
        langfuse.score_current_trace(
            name="latency_score",
            value=latency_score,
            comment=f"Response time: {duration_ms:.2f}ms",
        )

        # Model tier score
        model_name = trace_data["model"].lower()
        if any(
            premium in model_name
            for premium in ["gpt-4", "claude-3-opus", "gemini-1.5-pro"]
        ):
            model_tier = 1.0
        elif any(mid in model_name for mid in ["gemini-1.5-flash", "claude-3-haiku"]):
            model_tier = 0.7
        else:
            model_tier = 0.5

        langfuse.score_current_trace(
            name="model_tier",
            value=model_tier,
            comment=f"Model capability tier: {trace_data['model']}",
        )

    except Exception as e:
        logger.error(f"Error adding success scores: {e}")


def _add_error_scores(
    langfuse: Langfuse,
    error: Exception,
    duration_ms: float,
) -> None:
    """Add error-related scores to the current trace."""
    try:
        # Error score
        langfuse.score_current_trace(
            name="trace_success",
            value=0.0,
            comment=f"Error: {type(error).__name__} - {str(error)[:100]}",
        )

        # Error category
        error_str = str(error).lower()
        error_category = "other"

        if "timeout" in error_str:
            error_category = "timeout"
        elif "connection" in error_str or "network" in error_str:
            error_category = "network"
        elif "auth" in error_str or "key" in error_str:
            error_category = "authentication"
        elif "limit" in error_str or "quota" in error_str:
            error_category = "rate_limit"

        langfuse.score_current_trace(
            name="error_category",
            value=error_category,
            comment=f"Error classified as: {error_category}",
        )

    except Exception as e:
        logger.error(f"Error adding error scores: {e}")
