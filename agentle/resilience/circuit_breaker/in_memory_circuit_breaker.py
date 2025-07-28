import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, override

from agentle.resilience.circuit_breaker.circuit_breaker_protocol import (
    CircuitBreakerProtocol,
)
from agentle.resilience.circuit_breaker.circuit_state import CircuitState

logger = logging.getLogger(__name__)


@dataclass
class InMemoryCircuitBreaker(CircuitBreakerProtocol):
    """
    In-memory circuit breaker implementation with enhanced features.

    This implementation provides circuit breaking functionality with improved
    error handling, metrics collection, and configurable behavior.

    WARNING: This implementation stores state in memory and is NOT suitable
    for distributed systems with multiple processes/workers. Use RedisCircuitBreaker
    or DatabaseCircuitBreaker for production distributed scenarios.

    Features:
    - Configurable failure thresholds and recovery timeouts
    - Half-open state for testing recovery
    - Exponential backoff for recovery attempts
    - Comprehensive metrics and monitoring
    - Thread-safe operations with minimal locking
    - Automatic cleanup of stale circuits
    """

    failure_threshold: int = 5
    recovery_timeout: float = 300.0  # 5 minutes
    half_open_max_calls: int = 3  # Max calls to allow in half-open state
    half_open_success_threshold: int = 2  # Successes needed to close circuit
    exponential_backoff_multiplier: float = 1.5
    max_recovery_timeout: float = 1800.0  # 30 minutes max
    circuit_cleanup_interval: int = 3600  # 1 hour
    enable_metrics: bool = True

    _circuits: dict[str, CircuitState] = field(
        default_factory=lambda: defaultdict(CircuitState)
    )
    _half_open_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _half_open_successes: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _recovery_attempts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _metrics: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _last_cleanup: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        if not hasattr(self, "_cleanup_task"):
            self._cleanup_task = None
            self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            try:
                loop = asyncio.get_event_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
                logger.debug("Started circuit breaker cleanup task")
            except RuntimeError:
                # No event loop running, cleanup will be done manually
                pass

    async def _cleanup_loop(self):
        """Background task to clean up stale circuits."""
        while True:
            try:
                await asyncio.sleep(self.circuit_cleanup_interval)
                await self._cleanup_stale_circuits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in circuit breaker cleanup loop: {e}")

    async def _cleanup_stale_circuits(self):
        """Remove circuits that haven't been used recently."""
        async with self._lock:
            current_time = time.time()
            stale_threshold = current_time - (self.circuit_cleanup_interval * 2)

            stale_circuits: list[str] = []
            for circuit_id, circuit in self._circuits.items():
                if circuit.last_failure_time < stale_threshold and not circuit.is_open:
                    stale_circuits.append(circuit_id)

            for circuit_id in stale_circuits:
                del self._circuits[circuit_id]
                if circuit_id in self._half_open_calls:
                    del self._half_open_calls[circuit_id]
                if circuit_id in self._half_open_successes:
                    del self._half_open_successes[circuit_id]
                if circuit_id in self._recovery_attempts:
                    del self._recovery_attempts[circuit_id]

            if stale_circuits:
                logger.debug(f"Cleaned up {len(stale_circuits)} stale circuits")
                if self.enable_metrics:
                    self._metrics["circuits_cleaned"] += len(stale_circuits)

            self._last_cleanup = current_time

    @override
    async def is_open(self, circuit_id: str) -> bool:
        """Check if the circuit is open (blocking operations)."""
        async with self._lock:
            circuit = self._circuits[circuit_id]

            if not circuit.is_open:
                return False

            current_time = time.time()
            recovery_timeout = self._calculate_recovery_timeout(circuit_id)

            # Check if recovery timeout has passed
            if current_time - circuit.last_failure_time > recovery_timeout:
                # Transition to half-open state
                circuit.is_open = False
                self._half_open_calls[circuit_id] = 0
                self._half_open_successes[circuit_id] = 0
                logger.info(f"Circuit {circuit_id} transitioned to half-open state")

                if self.enable_metrics:
                    self._metrics["circuits_half_opened"] += 1

                return False

            if self.enable_metrics:
                self._metrics["calls_blocked"] += 1

            return True

    def _calculate_recovery_timeout(self, circuit_id: str) -> float:
        """Calculate recovery timeout with exponential backoff."""
        attempts = self._recovery_attempts[circuit_id]
        timeout = self.recovery_timeout * (
            self.exponential_backoff_multiplier**attempts
        )
        return min(timeout, self.max_recovery_timeout)

    @override
    async def record_success(self, circuit_id: str) -> None:
        """Record a successful operation."""
        async with self._lock:
            circuit = self._circuits[circuit_id]

            if circuit.is_open:
                # Circuit is open, this shouldn't happen
                logger.warning(f"Success recorded for open circuit {circuit_id}")
                return

            # Check if we're in half-open state
            if circuit_id in self._half_open_calls:
                self._half_open_calls[circuit_id] += 1
                self._half_open_successes[circuit_id] += 1

                # Check if we should close the circuit
                if (
                    self._half_open_successes[circuit_id]
                    >= self.half_open_success_threshold
                    or self._half_open_calls[circuit_id] >= self.half_open_max_calls
                ):
                    if (
                        self._half_open_successes[circuit_id]
                        >= self.half_open_success_threshold
                    ):
                        # Enough successes, close the circuit
                        circuit.failure_count = 0
                        circuit.last_failure_time = 0.0
                        self._recovery_attempts[circuit_id] = 0

                        # Clean up half-open state
                        del self._half_open_calls[circuit_id]
                        del self._half_open_successes[circuit_id]

                        logger.info(
                            f"Circuit {circuit_id} closed after successful recovery"
                        )

                        if self.enable_metrics:
                            self._metrics["circuits_closed"] += 1
                    else:
                        # Not enough successes, reopen the circuit
                        circuit.is_open = True
                        circuit.last_failure_time = time.time()
                        self._recovery_attempts[circuit_id] += 1

                        # Clean up half-open state
                        del self._half_open_calls[circuit_id]
                        del self._half_open_successes[circuit_id]

                        logger.warning(
                            f"Circuit {circuit_id} reopened after failed recovery attempt"
                        )

                        if self.enable_metrics:
                            self._metrics["circuits_reopened"] += 1
            else:
                # Normal operation, reset failure count
                circuit.failure_count = 0
                circuit.last_failure_time = 0.0
                self._recovery_attempts[circuit_id] = 0

            if self.enable_metrics:
                self._metrics["successes_recorded"] += 1

    @override
    async def record_failure(self, circuit_id: str) -> None:
        """Record a failed operation."""
        async with self._lock:
            circuit = self._circuits[circuit_id]
            current_time = time.time()

            # Check if we're in half-open state
            if circuit_id in self._half_open_calls:
                self._half_open_calls[circuit_id] += 1

                # Failure in half-open state, reopen the circuit
                circuit.is_open = True
                circuit.failure_count += 1
                circuit.last_failure_time = current_time
                self._recovery_attempts[circuit_id] += 1

                # Clean up half-open state
                del self._half_open_calls[circuit_id]
                if circuit_id in self._half_open_successes:
                    del self._half_open_successes[circuit_id]

                logger.warning(
                    f"Circuit {circuit_id} reopened due to failure in half-open state"
                )

                if self.enable_metrics:
                    self._metrics["circuits_reopened"] += 1
            else:
                # Normal failure handling
                circuit.failure_count += 1
                circuit.last_failure_time = current_time

                if circuit.failure_count >= self.failure_threshold:
                    circuit.is_open = True
                    logger.warning(
                        f"Circuit {circuit_id} opened due to {circuit.failure_count} failures"
                    )

                    if self.enable_metrics:
                        self._metrics["circuits_opened"] += 1

            if self.enable_metrics:
                self._metrics["failures_recorded"] += 1

    @override
    async def get_failure_count(self, circuit_id: str) -> int:
        """Get the current failure count for the circuit."""
        async with self._lock:
            return self._circuits[circuit_id].failure_count

    @override
    async def reset_circuit(self, circuit_id: str) -> None:
        """Manually reset the circuit to closed state."""
        async with self._lock:
            circuit = self._circuits[circuit_id]
            circuit.failure_count = 0
            circuit.is_open = False
            circuit.last_failure_time = 0.0
            self._recovery_attempts[circuit_id] = 0

            # Clean up half-open state if exists
            if circuit_id in self._half_open_calls:
                del self._half_open_calls[circuit_id]
            if circuit_id in self._half_open_successes:
                del self._half_open_successes[circuit_id]

            logger.info(f"Circuit {circuit_id} manually reset")

            if self.enable_metrics:
                self._metrics["circuits_reset"] += 1

    async def get_circuit_state(self, circuit_id: str) -> dict[str, Any]:
        """Get detailed state information for a circuit."""
        async with self._lock:
            circuit = self._circuits[circuit_id]
            current_time = time.time()

            state = {
                "circuit_id": circuit_id,
                "is_open": circuit.is_open,
                "failure_count": circuit.failure_count,
                "last_failure_time": circuit.last_failure_time,
                "recovery_attempts": self._recovery_attempts[circuit_id],
                "is_half_open": circuit_id in self._half_open_calls,
                "next_recovery_attempt_in_seconds": 0,
            }

            if circuit.is_open:
                recovery_timeout = self._calculate_recovery_timeout(circuit_id)
                time_since_failure = current_time - circuit.last_failure_time
                state["next_recovery_attempt_in_seconds"] = max(
                    0, recovery_timeout - time_since_failure
                )

            if circuit_id in self._half_open_calls:
                state.update(
                    {
                        "half_open_calls": self._half_open_calls[circuit_id],
                        "half_open_successes": self._half_open_successes[circuit_id],
                        "remaining_half_open_calls": max(
                            0,
                            self.half_open_max_calls
                            - self._half_open_calls[circuit_id],
                        ),
                    }
                )

            return state

    async def get_all_circuits(self) -> list[dict[str, Any]]:
        """Get state information for all circuits."""
        async with self._lock:
            circuits = []
            for circuit_id in self._circuits.keys():
                state = await self.get_circuit_state(circuit_id)
                circuits.append(state)
            return circuits

    async def get_metrics(self) -> dict[str, int]:
        """Get circuit breaker metrics."""
        if not self.enable_metrics:
            return {}

        async with self._lock:
            metrics = dict(self._metrics)
            metrics.update(
                {
                    "total_circuits": len(self._circuits),
                    "open_circuits": sum(
                        1 for c in self._circuits.values() if c.is_open
                    ),
                    "half_open_circuits": len(self._half_open_calls),
                    "closed_circuits": sum(
                        1 for c in self._circuits.values() if not c.is_open
                    )
                    - len(self._half_open_calls),
                    "last_cleanup_seconds_ago": int(time.time() - self._last_cleanup),
                }
            )
            return metrics

    async def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        if self.enable_metrics:
            async with self._lock:
                self._metrics.clear()
                logger.debug("Circuit breaker metrics reset")

    async def bulk_reset_circuits(self, circuit_ids: list[str]) -> dict[str, bool]:
        """Reset multiple circuits in a single operation."""
        results = {}
        async with self._lock:
            for circuit_id in circuit_ids:
                try:
                    circuit = self._circuits[circuit_id]
                    circuit.failure_count = 0
                    circuit.is_open = False
                    circuit.last_failure_time = 0.0
                    self._recovery_attempts[circuit_id] = 0

                    # Clean up half-open state if exists
                    if circuit_id in self._half_open_calls:
                        del self._half_open_calls[circuit_id]
                    if circuit_id in self._half_open_successes:
                        del self._half_open_successes[circuit_id]

                    results[circuit_id] = True

                    if self.enable_metrics:
                        self._metrics["circuits_reset"] += 1

                except Exception as e:
                    logger.error(f"Failed to reset circuit {circuit_id}: {e}")
                    results[circuit_id] = False

        reset_count = sum(1 for success in results.values() if success)
        logger.info(
            f"Bulk reset completed: {reset_count}/{len(circuit_ids)} circuits reset"
        )
        return results

    async def close(self) -> None:
        """Clean up resources and stop background tasks."""
        if hasattr(self, "_cleanup_task") and self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Clear all state
        async with self._lock:
            self._circuits.clear()
            self._half_open_calls.clear()
            self._half_open_successes.clear()
            self._recovery_attempts.clear()
            self._metrics.clear()

        logger.info("Circuit breaker closed and resources cleaned up")
