from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import MutableMapping, Sequence, override, Tuple, Dict, List

from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.resilience.load_balancer.load_balancer_protocol import (
    LoadBalancerProtocol,
)


@dataclass
class ProviderQuota:
    req_per_min: int | None = None  # None means unlimited
    prompt_tokens_per_min: int | None = None
    completion_tokens_per_min: int | None = None
    weight: int = 1  # for weighted round-robin ranking


class InMemoryLoadBalancer(LoadBalancerProtocol):
    """
    Simple in-memory load balancer with per-provider minute quotas and
    weighted ranking. Not suitable for multi-process or multi-machine scenarios.
    """

    def __init__(
        self,
        *,
        quotas: MutableMapping[str, ProviderQuota] | None = None,
        model_quotas: MutableMapping[Tuple[str, str], ProviderQuota] | None = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._quotas: MutableMapping[str, ProviderQuota] = quotas or {}
        # Per (provider_id, model) quotas override provider-level quotas when present
        self._model_quotas: MutableMapping[Tuple[str, str], ProviderQuota] = (
            model_quotas or {}
        )
        # Sliding window per (provider, model): timestamps of accepted calls
        self._accepted_ts: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        # Token windows per (provider, model)
        self._prompt_token_events: Dict[Tuple[str, str], List[Tuple[float, int]]] = (
            defaultdict(list)
        )
        self._completion_token_events: Dict[
            Tuple[str, str], List[Tuple[float, int]]
        ] = defaultdict(list)
        # last rank index per (model) to implement weighted rotation (best-effort)
        self._rr_index: dict[str, int] = defaultdict(int)

    def _purge_old(self, key: Tuple[str, str], now: float) -> None:
        window_start = now - 60.0
        bucket = self._accepted_ts[key]
        # drop older than 60s
        i = 0
        while i < len(bucket) and bucket[i] < window_start:
            i += 1
        if i:
            del bucket[:i]
        # Purge tokens
        pt = self._prompt_token_events[key]
        i = 0
        while i < len(pt) and pt[i][0] < window_start:
            i += 1
        if i:
            del pt[:i]
        ct = self._completion_token_events[key]
        i = 0
        while i < len(ct) and ct[i][0] < window_start:
            i += 1
        if i:
            del ct[:i]

    def _effective_quota(self, provider_id: str, model: str | None) -> ProviderQuota:
        if model is not None and (provider_id, model) in self._model_quotas:
            return self._model_quotas[(provider_id, model)]
        return self._quotas.get(provider_id, ProviderQuota())

    async def rank_providers(
        self,
        providers: Sequence[GenerationProvider],
        *,
        model: str | None = None,
    ) -> Sequence[GenerationProvider]:
        # Sort by: under all applicable quotas, min headroom across quotas, then weight
        now = time.time()
        async with self._lock:
            scored: list[tuple[int, float, int, GenerationProvider]] = []
            for p in providers:
                pid = p.circuit_identity
                key = (pid, model or "__any__")
                self._purge_old(key, now)
                quota = self._effective_quota(pid, model)
                # Compute headrooms
                headrooms: list[float] = []
                # req/min
                if quota.req_per_min is not None and quota.req_per_min > 0:
                    used = len(self._accepted_ts[key])
                    remaining = max(0, quota.req_per_min - used)
                    headrooms.append(remaining / float(quota.req_per_min))
                # prompt tokens/min
                if (
                    quota.prompt_tokens_per_min is not None
                    and quota.prompt_tokens_per_min > 0
                ):
                    used_pt = sum(v for _, v in self._prompt_token_events[key])
                    remaining_pt = max(0, quota.prompt_tokens_per_min - used_pt)
                    headrooms.append(remaining_pt / float(quota.prompt_tokens_per_min))
                # completion tokens/min
                if (
                    quota.completion_tokens_per_min is not None
                    and quota.completion_tokens_per_min > 0
                ):
                    used_ct = sum(v for _, v in self._completion_token_events[key])
                    remaining_ct = max(0, quota.completion_tokens_per_min - used_ct)
                    headrooms.append(
                        remaining_ct / float(quota.completion_tokens_per_min)
                    )

                if not headrooms:  # unlimited
                    under_quota = 1
                    min_headroom = 1.0
                else:
                    min_headroom = min(headrooms)
                    under_quota = 1 if min_headroom > 0.0 else 0
                weight = quota.weight
                scored.append((under_quota, min_headroom, weight, p))

            # prioritize under_quota first, then higher weight
            scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            ordered = [p for _, _, _, p in scored]

            # lightweight rotation among top by moving rr_index steps
            key = model or "__default__"
            if ordered:
                idx = self._rr_index[key] % len(ordered)
                ordered = ordered[idx:] + ordered[:idx]
                self._rr_index[key] = (self._rr_index[key] + 1) % (10**9)

            return ordered

    async def acquire(self, provider_id: str, *, model: str | None = None) -> bool:
        now = time.time()
        async with self._lock:
            key = (provider_id, model or "__any__")
            self._purge_old(key, now)
            quota = self._effective_quota(provider_id, model)
            # req/min gate
            if (
                quota.req_per_min is not None
                and quota.req_per_min > 0
                and len(self._accepted_ts[key]) >= quota.req_per_min
            ):
                return False
            # token gates are enforced post-result via ranking bias; we can't know tokens yet
            # For safety, we still admit; callers should prefer providers with token headroom.
            self._accepted_ts[key].append(now)
            return True

    @override
    async def record_result(
        self,
        provider_id: str,
        *,
        model: str | None = None,
        success: bool = True,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        now = time.time()
        async with self._lock:
            key = (provider_id, model or "__any__")
            # Ensure windows exist and purge
            self._purge_old(key, now)
            if prompt_tokens and prompt_tokens > 0:
                self._prompt_token_events[key].append((now, int(prompt_tokens)))
            if completion_tokens and completion_tokens > 0:
                self._completion_token_events[key].append((now, int(completion_tokens)))
        return None
