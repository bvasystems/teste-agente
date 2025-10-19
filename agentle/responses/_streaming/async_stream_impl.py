"""
Concrete implementation of AsyncStream for streaming responses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TypeVar

from agentle.responses._streaming.async_stream import AsyncStream

_T = TypeVar("_T")


class AsyncStreamImpl(AsyncStream[_T]):
    """
    Concrete implementation of AsyncStream protocol.
    
    Wraps an async generator to provide both AsyncIterator and AsyncContextManager
    interfaces for streaming responses.
    
    Usage:
        ```python
        async def generate_events():
            for i in range(10):
                yield i
        
        stream = AsyncStreamImpl(generate_events())
        
        # Use as async iterator
        async for event in stream:
            print(event)
        
        # Or use as context manager
        async with stream:
            async for event in stream:
                print(event)
        ```
    """
    
    def __init__(self, generator: AsyncIterator[_T]):
        """
        Initialize the stream with an async generator.
        
        Args:
            generator: The async generator that produces stream items
        """
        self._generator = generator
        self._closed = False
    
    async def __anext__(self) -> _T:
        """
        Get the next item from the stream.
        
        Returns:
            The next item from the underlying generator
            
        Raises:
            StopAsyncIteration: When the stream is exhausted
        """
        if self._closed:
            raise StopAsyncIteration
        
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise
    
    def __aiter__(self) -> AsyncStream[_T]:
        """
        Return self as the async iterator.
        
        Returns:
            Self
        """
        return self
    
    async def __aenter__(self) -> AsyncStream[_T]:
        """
        Enter the async context manager.
        
        Returns:
            Self
        """
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """
        Exit the async context manager.
        
        Performs cleanup by closing the underlying generator if it supports it.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self._closed = True
        
        # Try to close the generator if it has aclose method
        if hasattr(self._generator, 'aclose'):
            await self._generator.aclose()  # type: ignore
