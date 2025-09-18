import asyncio
import os
import tempfile
from typing import Sequence, cast
from datetime import datetime
import uuid

import pytest
import pytest_asyncio

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.failover.failover_generation_provider import (
    FailoverGenerationProvider,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.resilience.circuit_breaker.duckdb_circuit_breaker import (
    DuckDBCircuitBreaker,
)


# Note: tests use pytest-asyncio; avoid nested asyncio.run in teardowns.


# --- Test doubles -----------------------------------------------------------


class _BaseFakeProvider(GenerationProvider):
    def __init__(self, name: str, *, provider_id: str | None = None) -> None:
        super().__init__(otel_clients=None, provider_id=provider_id or name)
        self._name = name

    @property
    def default_model(self) -> str:  # pragma: no cover - trivial
        return "fake-model"

    @property
    def organization(self) -> str:  # pragma: no cover - trivial
        return "fake"

    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:  # pragma: no cover - trivial
        return 0.0

    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:  # pragma: no cover - trivial
        return 0.0

    def map_model_kind_to_provider_model(
        self, model_kind: ModelKind
    ) -> str:  # pragma: no cover - not used
        return "fake-model"


class SuccessProvider(_BaseFakeProvider):
    async def generate_async(
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[DeveloperMessage | UserMessage],
        response_schema=None,
        generation_config=None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[None]:
        msg = GeneratedAssistantMessage(
            parts=[TextPart(text=f"{self._name}:ok")], parsed=None
        )
        return Generation(
            id=uuid.uuid4(),
            object="chat.generation",
            created=datetime.now(),
            model=cast(str, model) if isinstance(model, str) else self.default_model,
            choices=[Choice(index=0, message=msg)],
            usage=Usage.zero(),
        )


class FailProvider(_BaseFakeProvider):
    def __init__(
        self, name: str, *, transient: bool, provider_id: str | None = None
    ) -> None:
        super().__init__(name, provider_id=provider_id)
        self._transient = transient

    async def generate_async(
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[DeveloperMessage | UserMessage],
        response_schema=None,
        generation_config=None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[None]:  # pragma: no cover - raises
        if self._transient:
            # Message should be detected by _should_trip_circuit
            raise RuntimeError("Timeout connecting to upstream service")
        else:
            raise ValueError("Bad request 400 - user input invalid")


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture()
def cb_path() -> str:
    with tempfile.TemporaryDirectory(prefix="cb-") as tmp:
        yield os.path.join(tmp, "circuit_breaker.duckdb")


@pytest_asyncio.fixture()
async def breaker(cb_path: str) -> DuckDBCircuitBreaker:
    b = DuckDBCircuitBreaker(
        db_path=cb_path,
        failure_threshold=2,
        recovery_timeout=0.05,
        half_open_max_calls=2,
        half_open_success_threshold=2,
        exponential_backoff_multiplier=2.0,
        max_recovery_timeout=1.0,
    )
    try:
        yield b
    finally:
        await b.close()


@pytest_asyncio.fixture()
async def breaker_close_on_first_success(cb_path: str) -> DuckDBCircuitBreaker:
    b = DuckDBCircuitBreaker(
        db_path=cb_path,
        failure_threshold=2,
        recovery_timeout=0.05,
        half_open_max_calls=2,
        half_open_success_threshold=1,  # close circuit after one success in half-open
        exponential_backoff_multiplier=2.0,
        max_recovery_timeout=1.0,
    )
    try:
        yield b
    finally:
        await b.close()


def _messages():
    return [
        DeveloperMessage(parts=[TextPart(text="sys")]),
        UserMessage(parts=[TextPart(text="hi")]),
    ]


# --- Tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failover_on_transient_failure_records_failure(
    breaker: DuckDBCircuitBreaker,
):
    p1 = FailProvider("p1", transient=True, provider_id="p1")
    p2 = SuccessProvider("p2", provider_id="p2")
    f = FailoverGenerationProvider(
        generation_providers=[p1, p2], circuit_breaker=breaker
    )

    gen = await f.generate_async(model="m", messages=_messages())
    assert "p2:ok" in gen.message.text
    await asyncio.sleep(0.05)
    cid1 = f._get_provider_circuit_id(p1, "m")
    assert await breaker.get_failure_count(cid1) == 1


@pytest.mark.asyncio
async def test_non_transient_failure_does_not_trip_circuit(
    breaker: DuckDBCircuitBreaker,
):
    p1 = FailProvider("p1", transient=False, provider_id="p1")
    p2 = SuccessProvider("p2", provider_id="p2")
    f = FailoverGenerationProvider(
        generation_providers=[p1, p2], circuit_breaker=breaker
    )

    gen = await f.generate_async(model="m", messages=_messages())
    assert "p2:ok" in gen.message.text
    await asyncio.sleep(0.05)
    cid1 = f._get_provider_circuit_id(p1, "m")
    assert await breaker.get_failure_count(cid1) == 0


@pytest.mark.asyncio
async def test_skip_open_circuit_and_use_next_provider(breaker: DuckDBCircuitBreaker):
    p1 = SuccessProvider("p1", provider_id="p1")  # would succeed, but should be skipped
    p2 = SuccessProvider("p2", provider_id="p2")
    f = FailoverGenerationProvider(
        generation_providers=[p1, p2], circuit_breaker=breaker
    )

    cid1 = f._get_provider_circuit_id(p1, "m")
    await breaker.record_failure(cid1)
    await breaker.record_failure(cid1)
    assert await breaker.is_open(cid1) is True
    gen = await f.generate_async(model="m", messages=_messages())
    assert "p2:ok" in gen.message.text


@pytest.mark.asyncio
async def test_last_resort_tries_skipped_provider_if_all_others_fail(
    breaker: DuckDBCircuitBreaker,
):
    p1 = SuccessProvider("p1", provider_id="p1")
    p2 = FailProvider("p2", transient=True, provider_id="p2")
    f = FailoverGenerationProvider(
        generation_providers=[p1, p2], circuit_breaker=breaker
    )

    cid1 = f._get_provider_circuit_id(p1, "m")
    await breaker.record_failure(cid1)
    await breaker.record_failure(cid1)
    assert await breaker.is_open(cid1) is True
    gen = await f.generate_async(model="m", messages=_messages())
    assert "p1:ok" in gen.message.text


@pytest.mark.asyncio
async def test_half_open_allows_attempt_and_closes_on_success(
    breaker_close_on_first_success: DuckDBCircuitBreaker,
):
    breaker = breaker_close_on_first_success
    p1 = SuccessProvider("p1", provider_id="p1")
    f = FailoverGenerationProvider(generation_providers=[p1], circuit_breaker=breaker)
    cid1 = f._get_provider_circuit_id(p1, "m")

    await breaker.record_failure(cid1)
    await breaker.record_failure(cid1)
    assert await breaker.is_open(cid1) is True
    await asyncio.sleep(0.06)
    assert await breaker.is_open(cid1) is False
    gen = await f.generate_async(model="m", messages=_messages())
    assert "p1:ok" in gen.message.text
    await asyncio.sleep(0.05)
    assert await breaker.is_open(cid1) is False
