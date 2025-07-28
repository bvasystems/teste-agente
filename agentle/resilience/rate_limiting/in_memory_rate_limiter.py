"""
In-memory rate limiter implementation with sliding window algorithm.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, override, Optional, Type, cast

from agentle.resilience.rate_limiting.rate_limit_config import RateLimitConfig
from agentle.resilience.rate_limiting.rate_limiter_protocol import RateLimiterProtocol

logger = logging.getLogger(__name__)


class RWLock:
    """Simple read-write lock implementation for asyncio."""

    def __init__(self):
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        self._lock = asyncio.Lock()
        self._read_ready = asyncio.Condition(self._lock)
        self._write_ready = asyncio.Condition(self._lock)

    async def __aenter__(self):
        """Acquire a read lock."""
        await self.reader_lock()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Release a read lock."""
        await self.reader_unlock()

    async def reader_lock(self):
        """Acquire a read lock."""
        async with self._lock:
            while self._writers_waiting > 0 or self._writer_active:
                await self._read_ready.wait()
            self._readers += 1

    async def reader_unlock(self):
        """Release a read lock."""
        async with self._lock:
            self._readers -= 1
            if self._readers == 0 and self._writers_waiting > 0:
                self._write_ready.notify()

    async def writer_lock(self):
        """Acquire a write lock."""
        async with self._lock:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                await self._write_ready.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    async def writer_unlock(self):
        """Release a write lock."""
        async with self._lock:
            self._writer_active = False
            if self._writers_waiting > 0:
                self._write_ready.notify()
            else:
                self._read_ready.notify_all()


class ReaderLock:
    """Context manager for read locks."""

    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.reader_lock()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        await self._rwlock.reader_unlock()


class WriterLock:
    """Context manager for write locks."""

    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.writer_lock()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        await self._rwlock.writer_unlock()


@dataclass
class TimeWindow:
    """Represents a time window for rate limiting."""

    requests: deque[float] = field(default_factory=deque)
    window_size_seconds: int = 60
    max_requests: int = 100

    def add_request(self, timestamp: float) -> None:
        """Add a request timestamp to the window."""
        self.requests.append(timestamp)
        self._clean_old_requests(timestamp)

    def _clean_old_requests(self, current_time: float) -> None:
        """Remove requests outside the current window."""
        cutoff_time = current_time - self.window_size_seconds
        while self.requests and self.requests[0] <= cutoff_time:
            self.requests.popleft()

    def get_current_count(self, current_time: float) -> int:
        """Get the current number of requests in the window."""
        self._clean_old_requests(current_time)
        return len(self.requests)

    def can_accept_request(self, current_time: float) -> bool:
        """Check if a new request can be accepted."""
        return self.get_current_count(current_time) < self.max_requests


@dataclass
class InMemoryRateLimiter(RateLimiterProtocol):
    """
    In-memory rate limiter implementation using sliding window algorithm.

    This implementation provides efficient rate limiting with multiple time windows
    (per minute, per hour, per day) and automatic cleanup of expired data.

    Features:
    - Sliding window algorithm for accurate rate limiting
    - Multiple time windows (minute, hour, day)
    - Automatic cleanup of expired request data
    - Thread-safe operations with minimal locking
    - Comprehensive metrics and monitoring
    - Memory-efficient storage with configurable limits
    - Bulk operations for better performance

    WARNING: This implementation stores state in memory and is NOT suitable
    for distributed systems. Use RedisRateLimiter for distributed scenarios.
    """

    default_config: RateLimitConfig = field(
        default_factory=lambda: {
            "max_requests_per_minute": 60,
            "max_requests_per_hour": 1000,
            "max_requests_per_day": 10000,
        }
    )
    cleanup_interval_seconds: int = 300  # 5 minutes
    max_identifiers: int = 10000  # Maximum number of identifiers to track
    enable_metrics: bool = True

    _minute_windows: MutableMapping[str, TimeWindow] = field(
        default_factory=lambda: defaultdict(lambda: TimeWindow(window_size_seconds=60))
    )
    _hour_windows: MutableMapping[str, TimeWindow] = field(
        default_factory=lambda: defaultdict(
            lambda: TimeWindow(window_size_seconds=3600)
        )
    )
    _day_windows: MutableMapping[str, TimeWindow] = field(
        default_factory=lambda: defaultdict(
            lambda: TimeWindow(window_size_seconds=86400)
        )
    )
    _lock: RWLock = field(default_factory=RWLock)
    _metrics: MutableMapping[str, int] = field(default_factory=lambda: defaultdict(int))
    _last_cleanup: float = field(default_factory=time.time)
    _cleanup_task: asyncio.Task[Any] | None = field(default=None)

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            try:
                loop = asyncio.get_event_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
                logger.debug("Started rate limiter cleanup task")
            except RuntimeError:
                # No event loop running, cleanup will be done manually
                pass

    async def _cleanup_loop(self):
        """Background task to clean up expired rate limit data."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                await self._cleanup_expired_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup loop: {e}")

    async def _cleanup_expired_data(self):
        """Remove identifiers with expired time windows."""
        async with WriterLock(self._lock):
            current_time = time.time()

            # Clean up minute windows
            minute_cutoff = current_time - 60
            expired_minute_ids = [
                identifier
                for identifier, window in self._minute_windows.items()
                if not window.requests or window.requests[-1] < minute_cutoff
            ]

            for identifier in expired_minute_ids:
                del self._minute_windows[identifier]

            # Clean up hour windows
            hour_cutoff = current_time - 3600
            expired_hour_ids = [
                identifier
                for identifier, window in self._hour_windows.items()
                if not window.requests or window.requests[-1] < hour_cutoff
            ]

            for identifier in expired_hour_ids:
                del self._hour_windows[identifier]

            # Clean up day windows
            day_cutoff = current_time - 86400
            expired_day_ids = [
                identifier
                for identifier, window in self._day_windows.items()
                if not window.requests or window.requests[-1] < day_cutoff
            ]

            for identifier in expired_day_ids:
                del self._day_windows[identifier]

            total_cleaned = (
                len(expired_minute_ids) + len(expired_hour_ids) + len(expired_day_ids)
            )
            if total_cleaned > 0:
                logger.debug(f"Cleaned up {total_cleaned} expired rate limit entries")
                if self.enable_metrics:
                    self._metrics["entries_cleaned"] += total_cleaned

            # Enforce max identifiers limit by removing oldest
            await self._enforce_identifier_limit()

            self._last_cleanup = current_time

    async def _enforce_identifier_limit(self):
        """Enforce maximum number of tracked identifiers."""
        total_identifiers = len(
            set(
                list(self._minute_windows.keys())
                + list(self._hour_windows.keys())
                + list(self._day_windows.keys())
            )
        )

        if total_identifiers > self.max_identifiers:
            # Find oldest identifiers across all windows
            identifier_last_activity: dict[str, float] = {}

            for identifier, window in self._minute_windows.items():
                if window.requests:
                    identifier_last_activity[identifier] = max(
                        identifier_last_activity.get(identifier, 0.0),
                        window.requests[-1],
                    )

            for identifier, window in self._hour_windows.items():
                if window.requests:
                    identifier_last_activity[identifier] = max(
                        identifier_last_activity.get(identifier, 0.0),
                        window.requests[-1],
                    )

            for identifier, window in self._day_windows.items():
                if window.requests:
                    identifier_last_activity[identifier] = max(
                        identifier_last_activity.get(identifier, 0.0),
                        window.requests[-1],
                    )

            # Sort by last activity and remove oldest
            sorted_identifiers = sorted(
                identifier_last_activity.items(), key=lambda x: x[1]
            )

            to_remove = total_identifiers - int(
                self.max_identifiers * 0.8
            )  # Remove to 80% of max
            for identifier, _ in sorted_identifiers[:to_remove]:
                if identifier in self._minute_windows:
                    del self._minute_windows[identifier]
                if identifier in self._hour_windows:
                    del self._hour_windows[identifier]
                if identifier in self._day_windows:
                    del self._day_windows[identifier]

            if to_remove > 0:
                logger.info(f"Evicted {to_remove} oldest identifiers to enforce limit")
                if self.enable_metrics:
                    self._metrics["identifiers_evicted"] += to_remove

    def _get_effective_config(self, config: RateLimitConfig | None) -> RateLimitConfig:
        """Get effective configuration by merging with defaults."""
        if config is None:
            return self.default_config

        effective_config = dict(self.default_config)
        effective_config.update(config)
        return cast(RateLimitConfig, effective_config)

    @override
    async def can_proceed(
        self, identifier: str, config: RateLimitConfig | None = None
    ) -> bool:
        """Check if the operation can proceed within rate limits."""
        effective_config = self._get_effective_config(config)
        current_time = time.time()

        async with ReaderLock(self._lock):
            # Check minute limit
            if "max_requests_per_minute" in effective_config:
                minute_window = self._minute_windows[identifier]
                minute_window.max_requests = effective_config["max_requests_per_minute"]

                if not minute_window.can_accept_request(current_time):
                    if self.enable_metrics:
                        self._metrics["requests_blocked_minute"] += 1
                    return False

            # Check hour limit
            if "max_requests_per_hour" in effective_config:
                hour_window = self._hour_windows[identifier]
                hour_window.max_requests = effective_config["max_requests_per_hour"]

                if not hour_window.can_accept_request(current_time):
                    if self.enable_metrics:
                        self._metrics["requests_blocked_hour"] += 1
                    return False

            # Check day limit
            if "max_requests_per_day" in effective_config:
                day_window = self._day_windows[identifier]
                day_window.max_requests = effective_config["max_requests_per_day"]

                if not day_window.can_accept_request(current_time):
                    if self.enable_metrics:
                        self._metrics["requests_blocked_day"] += 1
                    return False

        if self.enable_metrics:
            self._metrics["requests_allowed"] += 1

        return True

    @override
    async def record_request(
        self, identifier: str, config: RateLimitConfig | None = None
    ) -> None:
        """Record that a request was made."""
        effective_config = self._get_effective_config(config)
        current_time = time.time()

        async with WriterLock(self._lock):
            # Record in minute window
            if "max_requests_per_minute" in effective_config:
                minute_window = self._minute_windows[identifier]
                minute_window.max_requests = effective_config["max_requests_per_minute"]
                minute_window.add_request(current_time)

            # Record in hour window
            if "max_requests_per_hour" in effective_config:
                hour_window = self._hour_windows[identifier]
                hour_window.max_requests = effective_config["max_requests_per_hour"]
                hour_window.add_request(current_time)

            # Record in day window
            if "max_requests_per_day" in effective_config:
                day_window = self._day_windows[identifier]
                day_window.max_requests = effective_config["max_requests_per_day"]
                day_window.add_request(current_time)

        if self.enable_metrics:
            self._metrics["requests_recorded"] += 1

    @override
    async def get_current_usage(self, identifier: str) -> Mapping[str, int]:
        """Get current usage statistics for an identifier."""
        current_time = time.time()

        async with ReaderLock(self._lock):
            usage: dict[str, int] = {}

            if identifier in self._minute_windows:
                minute_window = self._minute_windows[identifier]
                usage["requests_per_minute"] = minute_window.get_current_count(
                    current_time
                )
                usage["max_requests_per_minute"] = minute_window.max_requests

            if identifier in self._hour_windows:
                hour_window = self._hour_windows[identifier]
                usage["requests_per_hour"] = hour_window.get_current_count(current_time)
                usage["max_requests_per_hour"] = hour_window.max_requests

            if identifier in self._day_windows:
                day_window = self._day_windows[identifier]
                usage["requests_per_day"] = day_window.get_current_count(current_time)
                usage["max_requests_per_day"] = day_window.max_requests

            return usage

    @override
    async def reset_limits(self, identifier: str) -> None:
        """Reset rate limits for an identifier."""
        async with WriterLock(self._lock):
            if identifier in self._minute_windows:
                del self._minute_windows[identifier]
            if identifier in self._hour_windows:
                del self._hour_windows[identifier]
            if identifier in self._day_windows:
                del self._day_windows[identifier]

        logger.debug(f"Reset rate limits for identifier: {identifier}")

        if self.enable_metrics:
            self._metrics["limits_reset"] += 1

    async def bulk_reset_limits(self, identifiers: list[str]) -> dict[str, bool]:
        """Reset rate limits for multiple identifiers in a single operation."""
        results: dict[str, bool] = {}

        async with WriterLock(self._lock):
            for identifier in identifiers:
                try:
                    if identifier in self._minute_windows:
                        del self._minute_windows[identifier]
                    if identifier in self._hour_windows:
                        del self._hour_windows[identifier]
                    if identifier in self._day_windows:
                        del self._day_windows[identifier]

                    results[identifier] = True

                    if self.enable_metrics:
                        self._metrics["limits_reset"] += 1

                except Exception as e:
                    logger.error(f"Failed to reset limits for {identifier}: {e}")
                    results[identifier] = False

        reset_count = sum(1 for success in results.values() if success)
        logger.info(
            f"Bulk reset completed: {reset_count}/{len(identifiers)} identifiers reset"
        )
        return results

    async def get_all_identifiers(self) -> list[str]:
        """Get all currently tracked identifiers."""
        async with ReaderLock(self._lock):
            all_identifiers: set[str] = set()
            all_identifiers.update(self._minute_windows.keys())
            all_identifiers.update(self._hour_windows.keys())
            all_identifiers.update(self._day_windows.keys())
            return list(all_identifiers)

    async def get_identifier_details(self, identifier: str) -> dict[str, Any]:
        """Get detailed information about an identifier's rate limiting state."""
        current_time = time.time()

        async with ReaderLock(self._lock):
            details: dict[str, Any] = {
                "identifier": identifier,
                "current_usage": await self.get_current_usage(identifier),
                "windows": {},
            }

            if identifier in self._minute_windows:
                window = self._minute_windows[identifier]
                details["windows"]["minute"] = {
                    "current_requests": len(window.requests),
                    "max_requests": window.max_requests,
                    "oldest_request_age_seconds": current_time - window.requests[0]
                    if window.requests
                    else 0,
                    "newest_request_age_seconds": current_time - window.requests[-1]
                    if window.requests
                    else 0,
                }

            if identifier in self._hour_windows:
                window = self._hour_windows[identifier]
                details["windows"]["hour"] = {
                    "current_requests": len(window.requests),
                    "max_requests": window.max_requests,
                    "oldest_request_age_seconds": current_time - window.requests[0]
                    if window.requests
                    else 0,
                    "newest_request_age_seconds": current_time - window.requests[-1]
                    if window.requests
                    else 0,
                }

            if identifier in self._day_windows:
                window = self._day_windows[identifier]
                details["windows"]["day"] = {
                    "current_requests": len(window.requests),
                    "max_requests": window.max_requests,
                    "oldest_request_age_seconds": current_time - window.requests[0]
                    if window.requests
                    else 0,
                    "newest_request_age_seconds": current_time - window.requests[-1]
                    if window.requests
                    else 0,
                }

            return details

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive rate limiter metrics."""
        if not self.enable_metrics:
            return {}

        async with ReaderLock(self._lock):
            metrics: dict[str, Any] = dict(self._metrics)
            metrics.update(
                {
                    "total_identifiers": len(
                        set(
                            list(self._minute_windows.keys())
                            + list(self._hour_windows.keys())
                            + list(self._day_windows.keys())
                        )
                    ),
                    "minute_windows": len(self._minute_windows),
                    "hour_windows": len(self._hour_windows),
                    "day_windows": len(self._day_windows),
                    "last_cleanup_seconds_ago": int(time.time() - self._last_cleanup),
                    "memory_usage": {
                        "minute_requests_tracked": sum(
                            len(w.requests) for w in self._minute_windows.values()
                        ),
                        "hour_requests_tracked": sum(
                            len(w.requests) for w in self._hour_windows.values()
                        ),
                        "day_requests_tracked": sum(
                            len(w.requests) for w in self._day_windows.values()
                        ),
                    },
                }
            )
            return metrics

    async def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        if self.enable_metrics:
            async with WriterLock(self._lock):
                self._metrics.clear()
                logger.debug("Rate limiter metrics reset")

    async def force_cleanup(self) -> dict[str, int | float]:
        """Force immediate cleanup and return detailed results."""
        start_time = time.time()

        await self._cleanup_expired_data()

        duration_ms = (time.time() - start_time) * 1000

        async with ReaderLock(self._lock):
            result: dict[str, int | float] = {
                "minute_windows_remaining": len(self._minute_windows),
                "hour_windows_remaining": len(self._hour_windows),
                "day_windows_remaining": len(self._day_windows),
                "total_identifiers": len(
                    set(
                        list(self._minute_windows.keys())
                        + list(self._hour_windows.keys())
                        + list(self._day_windows.keys())
                    )
                ),
                "duration_ms": duration_ms,
            }

        logger.info(f"Force cleanup completed: {result}")
        return result

    async def close(self) -> None:
        """Clean up resources and stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Clear all state
        async with WriterLock(self._lock):
            self._minute_windows.clear()
            self._hour_windows.clear()
            self._day_windows.clear()
            self._metrics.clear()

        logger.info("Rate limiter closed and resources cleaned up")
