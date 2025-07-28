"""
Session manager that coordinates session operations with enhanced resilience.
"""

import logging
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.sessions.session_store import SessionStore

logger = logging.getLogger(__name__)


class SessionManager[T_Session: BaseModel](BaseModel):
    """
    Session manager that provides a high-level interface for session operations.

    This class coordinates between different session store implementations
    and provides additional features like session validation, event handling,
    metadata management, and enhanced error handling.
    """

    session_store: SessionStore[T_Session]
    default_ttl_seconds: int | None = Field(default=3600)
    enable_events: bool = Field(default=False)
    enable_metrics: bool = Field(default=True)
    auto_cleanup_interval_seconds: int = Field(default=300)  # 5 minutes
    max_retry_attempts: int = Field(default=3)
    retry_delay_seconds: float = Field(default=1.0)

    _event_handlers: MutableMapping[str, list[Callable[..., Any]]] = PrivateAttr(
        default_factory=lambda: {
            "session_created": [],
            "session_updated": [],
            "session_deleted": [],
            "session_expired": [],
            "session_error": [],
        }
    )
    _metrics: MutableMapping[str, Any] = PrivateAttr(
        default_factory=lambda: {
            "operations_total": 0,
            "operations_successful": 0,
            "operations_failed": 0,
            "sessions_created": 0,
            "sessions_updated": 0,
            "sessions_deleted": 0,
            "sessions_expired": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_cleanup_at": None,
            "average_operation_time_ms": 0.0,
        }
    )
    _operation_times: list[float] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def get_session(
        self,
        session_id: str,
        refresh_ttl: bool = False,
        additional_ttl_seconds: int | None = None,
    ) -> T_Session | None:
        """
        Get a session by ID with optional TTL refresh and enhanced error handling.

        Args:
            session_id: Session ID to retrieve
            refresh_ttl: Whether to refresh the TTL when accessing the session
            additional_ttl_seconds: Additional seconds to add to TTL if refreshing

        Returns:
            The session if found, None otherwise
        """
        start_time = time.time()

        try:
            session = await self._retry_operation(
                self.session_store.get_session, session_id
            )

            if session is not None:
                self._record_metric("cache_hits")

                if refresh_ttl:
                    # Refresh TTL on access
                    ttl = additional_ttl_seconds or self.default_ttl_seconds
                    if ttl:
                        await self._retry_operation(
                            self.session_store.set_session, session_id, session, ttl
                        )
                        logger.debug(f"Refreshed TTL for session {session_id}")
            else:
                self._record_metric("cache_misses")

            self._record_successful_operation(start_time)
            return session

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to get session {session_id}: {e}")
            await self._fire_event("session_error", session_id, None, error=str(e))
            return None

    async def create_session(
        self,
        session_id: str,
        session: T_Session,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Create a new session with enhanced validation and error handling.

        Args:
            session_id: Unique identifier for the session
            session: The session object to store
            ttl_seconds: TTL for the session (uses default if not provided)
            overwrite: Whether to overwrite if session already exists

        Returns:
            True if session was created, False if it already exists (and overwrite=False)
        """
        start_time = time.time()

        try:
            # Validate session_id
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")

            # Check if session already exists
            if not overwrite:
                existing = await self._retry_operation(
                    self.session_store.exists, session_id
                )
                if existing:
                    logger.debug(
                        f"Session {session_id} already exists, not overwriting"
                    )
                    return False

            # Set TTL
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

            # Store the session
            await self._retry_operation(
                self.session_store.set_session, session_id, session, ttl
            )

            # Fire event
            if self.enable_events:
                await self._fire_event("session_created", session_id, session)

            self._record_metric("sessions_created")
            self._record_successful_operation(start_time)
            logger.debug(f"Created session {session_id} with TTL {ttl}")
            return True

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to create session {session_id}: {e}")
            await self._fire_event("session_error", session_id, session, error=str(e))
            raise

    async def update_session(
        self,
        session_id: str,
        session: T_Session,
        ttl_seconds: int | None = None,
        create_if_missing: bool = False,
    ) -> bool:
        """
        Update an existing session with enhanced validation.

        Args:
            session_id: Session ID to update
            session: Updated session object
            ttl_seconds: New TTL for the session
            create_if_missing: Whether to create the session if it doesn't exist

        Returns:
            True if session was updated, False if it doesn't exist (and create_if_missing=False)
        """
        start_time = time.time()

        try:
            # Validate session_id
            if not session_id or not session_id.strip():
                raise ValueError("Session ID cannot be empty")

            # Check if session exists
            exists = await self._retry_operation(self.session_store.exists, session_id)

            if not exists and not create_if_missing:
                logger.debug(f"Session {session_id} does not exist, not creating")
                return False

            # Set TTL
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

            # Store the session
            await self._retry_operation(
                self.session_store.set_session, session_id, session, ttl
            )

            # Fire event
            if self.enable_events:
                event_type = "session_created" if not exists else "session_updated"
                await self._fire_event(event_type, session_id, session)

            metric_key = "sessions_created" if not exists else "sessions_updated"
            self._record_metric(metric_key)
            self._record_successful_operation(start_time)
            logger.debug(f"Updated session {session_id}")
            return True

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to update session {session_id}: {e}")
            await self._fire_event("session_error", session_id, session, error=str(e))
            raise

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session with enhanced error handling.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if it didn't exist
        """
        start_time = time.time()

        try:
            # Get session data before deletion for event
            session_data = None
            if self.enable_events:
                session_data = await self._retry_operation(
                    self.session_store.get_session, session_id
                )

            # Delete the session
            deleted = await self._retry_operation(
                self.session_store.delete_session, session_id
            )

            # Fire event
            if deleted and self.enable_events:
                await self._fire_event("session_deleted", session_id, session_data)

            if deleted:
                self._record_metric("sessions_deleted")
                logger.debug(f"Deleted session {session_id}")

            self._record_successful_operation(start_time)
            return deleted

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to delete session {session_id}: {e}")
            await self._fire_event("session_error", session_id, None, error=str(e))
            return False

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists with error handling."""
        try:
            return await self._retry_operation(self.session_store.exists, session_id)
        except Exception as e:
            logger.error(f"Failed to check if session {session_id} exists: {e}")
            return False

    async def list_sessions(
        self, pattern: str | None = None, include_metadata: bool = False
    ) -> Sequence[str] | list[Mapping[str, Any]]:
        """
        List session IDs or session metadata with enhanced error handling.

        Args:
            pattern: Optional pattern to filter session IDs
            include_metadata: Whether to include session metadata

        Returns:
            List of session IDs or session metadata dictionaries
        """
        start_time = time.time()

        try:
            session_ids = await self._retry_operation(
                self.session_store.list_sessions, pattern
            )

            if not include_metadata:
                self._record_successful_operation(start_time)
                return session_ids

            # Build metadata for each session
            sessions_with_metadata = []
            for session_id in session_ids:
                metadata: MutableMapping[str, Any] = {
                    "session_id": session_id,
                    "exists": True,  # We know it exists since we just listed it
                }

                # Add TTL information if available
                if hasattr(self.session_store, "get_session_ttl"):
                    try:
                        ttl_method = getattr(self.session_store, "get_session_ttl")
                        ttl = await ttl_method(session_id)
                        metadata["ttl_seconds"] = ttl
                    except Exception as e:
                        logger.warning(
                            f"Failed to get TTL for session {session_id}: {e}"
                        )
                        metadata["ttl_seconds"] = None

                sessions_with_metadata.append(metadata)

            self._record_successful_operation(start_time)
            return sessions_with_metadata

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to list sessions: {e}")
            return [] if not include_metadata else []

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions with enhanced logging.

        Returns:
            Number of sessions cleaned up
        """
        start_time = time.time()

        try:
            cleaned_count = await self._retry_operation(
                self.session_store.cleanup_expired
            )

            if cleaned_count > 0:
                self._record_metric("sessions_expired", cleaned_count)
                logger.info(f"Cleaned up {cleaned_count} expired sessions")

            self._metrics["last_cleanup_at"] = time.time()
            self._record_successful_operation(start_time)
            return cleaned_count

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0

    async def get_session_count(self) -> int:
        """Get the total number of active sessions with error handling."""
        try:
            return await self._retry_operation(self.session_store.get_session_count)
        except Exception as e:
            logger.error(f"Failed to get session count: {e}")
            return 0

    async def extend_session_ttl(
        self, session_id: str, additional_seconds: int
    ) -> bool:
        """
        Extend the TTL of a session with enhanced error handling.

        Args:
            session_id: Session ID to extend
            additional_seconds: Additional seconds to add to TTL

        Returns:
            True if TTL was extended, False if session doesn't exist
        """
        start_time = time.time()

        try:
            if hasattr(self.session_store, "extend_session_ttl"):
                extend_method = getattr(self.session_store, "extend_session_ttl")
                result = await self._retry_operation(
                    extend_method, session_id, additional_seconds
                )
                self._record_successful_operation(start_time)
                return result
            else:
                # Fallback: get session and re-store with extended TTL
                session = await self._retry_operation(
                    self.session_store.get_session, session_id
                )
                if session is None:
                    return False

                # Calculate new TTL (this is approximate since we don't know the current TTL)
                new_ttl = additional_seconds
                await self._retry_operation(
                    self.session_store.set_session, session_id, session, new_ttl
                )
                self._record_successful_operation(start_time)
                return True

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to extend TTL for session {session_id}: {e}")
            return False

    async def batch_get_sessions(
        self, session_ids: Sequence[str]
    ) -> Mapping[str, T_Session | None]:
        """
        Get multiple sessions in a batch operation for better performance.

        Args:
            session_ids: List of session IDs to retrieve

        Returns:
            Dictionary mapping session IDs to sessions (or None if not found)
        """
        start_time = time.time()
        results: MutableMapping[str, T_Session | None] = {}

        try:
            # TODO: If the session store supports batch operations, use them
            # For now, do individual gets with some concurrency control
            import asyncio

            async def get_single_session(
                session_id: str,
            ) -> tuple[str, T_Session | None]:
                session = await self.get_session(session_id)
                return session_id, session

            # Limit concurrency to avoid overwhelming the store
            semaphore = asyncio.Semaphore(10)

            async def bounded_get(session_id: str) -> tuple[str, T_Session | None]:
                async with semaphore:
                    return await get_single_session(session_id)

            tasks = [bounded_get(session_id) for session_id in session_ids]
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed_results:
                if isinstance(result, BaseException):
                    logger.warning(f"Failed to get session in batch: {result}")
                    continue
                # result is a tuple (session_id, session) at this point
                session_id, session = result
                results[session_id] = session

            self._record_successful_operation(start_time)
            logger.debug(f"Batch retrieved {len(results)} sessions")
            return results

        except Exception as e:
            self._record_failed_operation(start_time)
            logger.error(f"Failed to batch get sessions: {e}")
            return results

    def add_event_handler(self, event_type: str, handler: Callable[..., Any]) -> None:
        """
        Add an event handler for session events.

        Args:
            event_type: Type of event (session_created, session_updated, session_deleted, session_expired, session_error)
            handler: Async callable to handle the event
        """
        if event_type not in self._event_handlers:
            raise ValueError(f"Unknown event type: {event_type}")

        self._event_handlers[event_type].append(handler)
        logger.debug(f"Added event handler for {event_type}")

    def remove_event_handler(
        self, event_type: str, handler: Callable[..., Any]
    ) -> bool:
        """
        Remove an event handler.

        Args:
            event_type: Type of event
            handler: Handler to remove

        Returns:
            True if handler was removed, False if not found
        """
        if event_type not in self._event_handlers:
            return False

        try:
            self._event_handlers[event_type].remove(handler)
            logger.debug(f"Removed event handler for {event_type}")
            return True
        except ValueError:
            return False

    async def _retry_operation(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Retry an operation with exponential backoff.

        Args:
            operation: The operation to retry
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            Result of the operation

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retry_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retry_attempts - 1:
                    delay = self.retry_delay_seconds * (
                        2**attempt
                    )  # Exponential backoff
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retry_attempts}), "
                        + f"retrying in {delay}s: {e}"
                    )
                    import asyncio

                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Operation failed after {self.max_retry_attempts} attempts: {e}"
                    )

        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("Operation failed with no exception recorded")

    async def _fire_event(
        self,
        event_type: str,
        session_id: str,
        session_data: T_Session | None,
        **extra_data: Any,
    ) -> None:
        """Fire an event to all registered handlers."""
        if not self.enable_events or event_type not in self._event_handlers:
            return

        handlers = self._event_handlers[event_type]
        for handler in handlers:
            try:
                # Try calling as async first, then sync
                try:
                    await handler(session_id, session_data, **extra_data)
                except TypeError:
                    # Might be a sync function
                    handler(session_id, session_data, **extra_data)
            except Exception as e:
                logger.warning(f"Event handler error for {event_type}: {e}")
                # Don't let event handler errors break session operations

    def _record_metric(self, metric_name: str, value: int = 1) -> None:
        """Record a metric if metrics are enabled."""
        if not self.enable_metrics:
            return

        if metric_name in self._metrics:
            if isinstance(self._metrics[metric_name], int):
                self._metrics[metric_name] += value
            else:
                self._metrics[metric_name] = value
        else:
            self._metrics[metric_name] = value

    def _record_successful_operation(self, start_time: float) -> None:
        """Record a successful operation with timing."""
        if not self.enable_metrics:
            return

        duration_ms = (time.time() - start_time) * 1000
        self._operation_times.append(duration_ms)

        # Keep only last 1000 operations for average calculation
        if len(self._operation_times) > 1000:
            self._operation_times = self._operation_times[-1000:]

        self._metrics["operations_total"] += 1
        self._metrics["operations_successful"] += 1
        self._metrics["average_operation_time_ms"] = sum(self._operation_times) / len(
            self._operation_times
        )

    def _record_failed_operation(self, start_time: float) -> None:
        """Record a failed operation."""
        if not self.enable_metrics:
            return

        self._metrics["operations_total"] += 1
        self._metrics["operations_failed"] += 1

    async def close(self) -> None:
        """Close the session manager and underlying store."""
        try:
            await self.session_store.close()
            logger.info("Session manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing session manager: {e}")

    def get_stats(self) -> Mapping[str, Any]:
        """
        Get comprehensive statistics about the session manager.

        Returns:
            Dictionary with statistics
        """
        handler_counts = {}
        for event_type, handlers in self._event_handlers.items():
            handler_counts[event_type] = len(handlers)

        base_stats = {
            "default_ttl_seconds": self.default_ttl_seconds,
            "events_enabled": self.enable_events,
            "metrics_enabled": self.enable_metrics,
            "event_handlers": handler_counts,
            "retry_config": {
                "max_attempts": self.max_retry_attempts,
                "delay_seconds": self.retry_delay_seconds,
            },
        }

        # Add metrics if enabled
        if self.enable_metrics:
            base_stats["metrics"] = dict(self._metrics)

        # Add store-specific stats if available
        if hasattr(self.session_store, "get_stats"):
            stats_method = getattr(self.session_store, "get_stats")
            base_stats["store_stats"] = stats_method()

        return base_stats

    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        if self.enable_metrics:
            for key in self._metrics:
                if isinstance(self._metrics[key], int):
                    self._metrics[key] = 0
                elif key == "average_operation_time_ms":
                    self._metrics[key] = 0.0
                elif key == "last_cleanup_at":
                    self._metrics[key] = None

            self._operation_times.clear()
            logger.debug("Session manager metrics reset")
