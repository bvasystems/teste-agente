from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from typing import Any, override

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.messages.message import Message
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.failover.failover_generation_provider import (
    FailoverGenerationProvider,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.resilience.load_balancer import (
    InMemoryLoadBalancer,
    ProviderQuota,
)


class MockProvider(GenerationProvider):
    """
    Minimal provider used to demonstrate the LoadBalancer.

    - Distinguishes instances via provider_id (e.g., different API keys)
    - Returns a mock Generation and configurable token usage per model
    """

    def __init__(self, *, provider_label: str, organization: str = "google") -> None:
        super().__init__(provider_id=provider_label)
        self._organization = organization
        self._default_model = "gemini-1.5-pro"

    @property
    @override
    def default_model(self) -> str:
        return self._default_model

    @property
    @override
    def organization(self) -> str:
        return self._organization

    @override
    async def generate_async[T = None](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: Any | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        # Simulate token usage per model to reflect quotas
        model_str = model if isinstance(model, str) and model else self.default_model
        if model_str == "gemini-1.5-pro":
            pt, ct = 1300, 700
        elif model_str == "gemini-1.5-flash":
            pt, ct = 500, 300
        else:
            pt, ct = 300, 200

        # Produce a simple mock generation with a model string that includes provider identity
        g = Generation.mock()
        g = g.clone(
            new_created=datetime.now(),
            new_model=f"{self.organization}:{self.provider_id}:{model_str}",
            new_usage=Usage(prompt_tokens=pt, completion_tokens=ct),
        )
        return g  # type: ignore[return-value]

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    def map_model_kind_to_provider_model(self, model_kind: ModelKind) -> str:
        # Not used in this demo; map to default
        return self.default_model


async def main() -> None:
    # Two providers simulating different API keys for the same vendor
    p_a = MockProvider(provider_label="gcp-key-A")
    p_b = MockProvider(provider_label="gcp-key-B")

    # Provider-level default quotas
    provider_defaults = {
        p_a.circuit_identity: ProviderQuota(
            req_per_min=10,
            prompt_tokens_per_min=50_000,
            completion_tokens_per_min=50_000,
            weight=1,
        ),
        p_b.circuit_identity: ProviderQuota(
            req_per_min=10,
            prompt_tokens_per_min=50_000,
            completion_tokens_per_min=50_000,
            weight=1,
        ),
    }

    # Per-model fine-tuning (example: flash allows more RPM and tokens than pro on B)
    model_overrides = {
        (p_a.circuit_identity, "gemini-1.5-pro"): ProviderQuota(
            req_per_min=3,
            prompt_tokens_per_min=5_000,
            completion_tokens_per_min=5_000,
            weight=1,
        ),
        (p_b.circuit_identity, "gemini-1.5-pro"): ProviderQuota(
            req_per_min=2,
            prompt_tokens_per_min=5_000,
            completion_tokens_per_min=5_000,
            weight=2,
        ),
        (p_a.circuit_identity, "gemini-1.5-flash"): ProviderQuota(
            req_per_min=6,
            prompt_tokens_per_min=20_000,
            completion_tokens_per_min=20_000,
            weight=2,
        ),
        (p_b.circuit_identity, "gemini-1.5-flash"): ProviderQuota(
            req_per_min=8,
            prompt_tokens_per_min=25_000,
            completion_tokens_per_min=25_000,
            weight=3,
        ),
    }

    lb = InMemoryLoadBalancer(quotas=provider_defaults, model_quotas=model_overrides)

    failover = FailoverGenerationProvider(
        generation_providers=[p_a, p_b],
        load_balancer=lb,
    )

    # Perform a series of calls; the LB should prioritize providers with per-model headroom
    models = [
        "gemini-1.5-pro",
        "gemini-1.5-pro",
        "gemini-1.5-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash",
    ]

    print("Starting demo: making requests with per-model quotas")
    for idx, mdl in enumerate(models, start=1):
        gen = await failover.generate_async(
            model=mdl,
            messages=[],
            response_schema=None,
        )
        print(
            f"{idx:02d}. model={mdl:16s} -> served_by={gen.model} usage={gen.usage.prompt_tokens}/{gen.usage.completion_tokens}"
        )

    print(
        "\nTip: Run this script again within a minute to see quotas tighten and decisions shift."
    )


if __name__ == "__main__":
    asyncio.run(main())
