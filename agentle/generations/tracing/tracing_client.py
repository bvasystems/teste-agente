"""
Simplified tracing client interface for the Agentle framework using Langfuse V3.

This module provides a minimal interface since Langfuse V3 handles most
tracing operations automatically through OpenTelemetry context managers.
"""

from abc import ABC, abstractmethod


class TracingClient(ABC):
    """
    Simplified base class for tracing clients compatible with Langfuse V3.

    Since Langfuse V3 uses OpenTelemetry context managers, most tracing
    operations are handled automatically by the @observe decorator.
    """

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if tracing is enabled and functional.

        Returns:
            bool: True if tracing is enabled and working
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        Force flush all pending events to the backend.

        This ensures all events are sent to the observability platform.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the client and clean up resources.

        This should flush any pending data and terminate background threads.
        """
        pass

    def auth_check(self) -> bool:
        """
        Check if the provided credentials are valid.

        Returns:
            bool: True if authentication is successful
        """
        return True  # Default implementation
