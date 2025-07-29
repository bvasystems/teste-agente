"""
Simplified Langfuse V3 client wrapper for the Agentle framework.

This implementation properly uses the Langfuse V3 SDK instead of trying to
recreate the V2 API patterns.
"""

import logging
from typing import Optional

from langfuse import Langfuse

from agentle.generations.tracing.tracing_client import TracingClient

logger = logging.getLogger(__name__)


class LangfuseTracingClient(TracingClient):
    """
    Simplified Langfuse V3 client wrapper.

    This client simply initializes and manages the Langfuse V3 client,
    letting the @observe decorator handle all tracing operations through
    the V3 context managers.
    """

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the Langfuse V3 client wrapper.

        Args:
            public_key: Langfuse public key (or set LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or set LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (or set LANGFUSE_HOST env var)
            debug: Enable debug logging
        """
        try:
            # Initialize Langfuse V3 client - this automatically registers
            # it as the global client accessible via get_client()
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=debug,
            )

            self._enabled = True
            self._logger = logging.getLogger(__name__)

            if debug:
                self._logger.setLevel(logging.DEBUG)

            self._logger.info("Langfuse V3 client initialized successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize Langfuse V3 client: {e}")
            self._enabled = False
            raise RuntimeError(f"Failed to initialize Langfuse client: {e}")

    def is_enabled(self) -> bool:
        """Check if tracing is enabled and functional."""
        return self._enabled

    def flush(self) -> None:
        """Flush all pending events to Langfuse."""
        if not self.is_enabled():
            return

        try:
            self._client.flush()
            self._logger.debug("Flushed all events to Langfuse")
        except Exception as e:
            self._logger.error(f"Error flushing events: {e}")

    def shutdown(self) -> None:
        """Shutdown the client and clean up resources."""
        if not self.is_enabled():
            return

        try:
            self._client.shutdown()
            self._logger.debug("Langfuse client shutdown complete")
        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")

    def auth_check(self) -> bool:
        """Check if the provided credentials are valid."""
        if not self.is_enabled():
            return False

        try:
            return self._client.auth_check()
        except Exception as e:
            self._logger.error(f"Auth check failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[BaseException],
    ) -> None:
        """Context manager exit with cleanup."""
        self.shutdown()
