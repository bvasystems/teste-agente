from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable, Sequence

from agentle.generations.providers.base.generation_provider import GenerationProvider


@runtime_checkable
class LoadBalancerProtocol(Protocol):
    """
    Abstract protocol for provider-aware load balancers.

    A LoadBalancer decides ordering and admission of requests across multiple
    providers using policies such as weighted round-robin, least-loaded, or
    quota/rate limits (e.g., X req/min free tiers).

    Contract:
    - rank_providers: provide a prioritized ordering (can also filter temporarily
      unavailable providers).
    - acquire: atomically check/admit a request for a provider; returns False when
      the policy prohibits sending right now (e.g., minute quota exceeded).
    - record_result: update internal metrics after a request completes (success/failure,
      usage). Implementations may ignore fields they don't track.
    """

    @abc.abstractmethod
    async def rank_providers(
        self,
        providers: Sequence[GenerationProvider],
        *,
        model: str | None = None,
    ) -> Sequence[GenerationProvider]:
        """Return providers reordered by preference under current policy."""
        ...

    @abc.abstractmethod
    async def acquire(self, provider_id: str, *, model: str | None = None) -> bool:
        """Attempt to admit a request for the given provider. Return True if allowed."""
        ...

    @abc.abstractmethod
    async def record_result(
        self,
        provider_id: str,
        *,
        model: str | None = None,
        success: bool = True,
        # Optional usage metrics; implementations can ignore
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Record the outcome of a request for accounting/metrics."""
        ...
