"""API metrics tracking."""

from __future__ import annotations

from rsb.models.base_model import BaseModel


class APIMetrics(BaseModel):
    """Metrics for API usage."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    requests_by_endpoint: dict[str, int] = {}
