"""Typed models for tracing context management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field, computed_field

from agentle.generations.tracing.otel_client import GenerationContext, TraceContext
from agentle.generations.tracing.otel_client_type import OtelClientType


class TraceInputData(BaseModel):
    """Structured input data for trace context."""

    input: str | list[dict[str, Any]] | None
    model: str | None
    has_tools: bool
    tools_count: int
    has_structured_output: bool
    reasoning_enabled: bool
    reasoning_effort: str | None
    temperature: float | None
    top_p: float | None
    max_output_tokens: int | None
    stream: bool


class TraceMetadata(BaseModel):
    """Metadata for trace context."""

    model: str
    provider: str
    base_url: str
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls, merging custom_metadata."""
        result = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
        }
        result.update(self.custom_metadata)
        return result


class UsageDetails(BaseModel):
    """Token usage details from API response."""

    input: int
    output: int
    total: int
    unit: str
    reasoning_tokens: int | None = None


class CostDetails(BaseModel):
    """Cost calculation details."""

    input: float
    output: float
    total: float
    currency: str
    input_tokens: int
    output_tokens: int


class TracingContext(BaseModel):
    """Container for a single client's tracing contexts."""

    model_config = {"arbitrary_types_allowed": True}

    client: OtelClientType
    trace_gen: AsyncGenerator[TraceContext | None, None]
    trace_ctx: TraceContext | None
    generation_gen: AsyncGenerator[GenerationContext | None, None]
    generation_ctx: GenerationContext | None

    @computed_field
    @property
    def client_name(self) -> str:
        """Get the client class name for logging."""
        return type(self.client).__name__
