from typing import Any

from rsb.coroutines.run_sync import run_sync
from rsb.models import BaseModel, Field

from agentle.responses.definitions.function_tool import FunctionTool
from agentle.responses.definitions.function_tool_call import FunctionToolCall


class ToolPair(BaseModel):
    """
    A container class that pairs a function tool with its associated function calls.

    This class maintains the relationship between a FunctionTool and the sequence
    of FunctionToolCall instances that have been made to that tool. It provides
    methods to update both the tool definition and track call history.

    Attributes:
        function_tool: The function tool definition, can be None if only calls exist.
        function_calls: List of function calls made to this tool in chronological order.

    Example:
        >>> tool = FunctionTool.from_callable(my_function)
        >>> pair = ToolPair(function_tool=tool, function_calls=[])
        >>> call = FunctionToolCall(name="my_function", arguments={"x": 1})
        >>> pair.change_function_call(call)
    """

    function_tool: FunctionTool | None = Field(
        default=None, description="The function tool."
    )
    function_calls: list[FunctionToolCall] | None = Field(
        default=None,
        description="The list of function calls in sequence they were called.",
    )

    def change_function_tool(self, function_tool: FunctionTool) -> None:
        """
        Update the function tool definition for this pair.

        Args:
            function_tool: The new function tool to associate with this pair.

        Example:
            >>> old_tool = FunctionTool.from_callable(old_function)
            >>> new_tool = FunctionTool.from_callable(new_function)
            >>> pair = ToolPair(function_tool=old_tool)
            >>> pair.change_function_tool(new_tool)
            >>> assert pair.function_tool == new_tool
        """
        self.function_tool = function_tool

    def change_function_call(self, function_call: FunctionToolCall) -> None:
        """
        Add a new function call to the call history for this tool.

        If no function calls list exists, it will be initialized as an empty list
        before adding the new call.

        Args:
            function_call: The function call to add to the history.

        Example:
            >>> pair = ToolPair(function_tool=tool)
            >>> call1 = FunctionToolCall(name="test", arguments={"x": 1})
            >>> call2 = FunctionToolCall(name="test", arguments={"x": 2})
            >>> pair.change_function_call(call1)
            >>> pair.change_function_call(call2)
            >>> assert len(pair.function_calls) == 2
        """
        if self.function_calls is None:
            self.function_calls = []
        self.function_calls.append(function_call)


class FunctionCallStore(BaseModel):
    """
    A centralized store for managing function tools and their call history.

    This class provides a comprehensive interface for managing function tools,
    tracking their usage, and executing them. It maintains a dictionary mapping
    tool names to ToolPair objects, which contain both the tool definition and
    the history of calls made to that tool.

    The store supports both synchronous and asynchronous tool execution, call
    tracking, tool management, and various query operations for analytics and
    debugging purposes.

    Attributes:
        store: Dictionary mapping tool names to ToolPair objects containing
               the tool definition and call history.

    Example:
        >>> store = FunctionCallStore()
        >>> tool = FunctionTool.from_callable(my_function)
        >>> store.add_function_tool(tool)
        >>> result = store.call_function_tool("my_function", arg1=1, arg2=2)
        >>> calls = store.retrieve_function_calls("my_function")
        >>> print(f"Tool called {len(calls)} times")
    """

    store: dict[str, ToolPair] = Field(
        default_factory=dict, description="The store of function tools and calls."
    )

    def call_function_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Synchronously call a function tool with the given name and arguments.

        This is a convenience method that wraps the async version using run_sync.
        It should be used when you need to call tools from synchronous code.

        Args:
            name: The name of the function tool to call.
            *args: Positional arguments to pass to the function tool.
            **kwargs: Keyword arguments to pass to the function tool.

        Returns:
            The result of calling the function tool.

        Raises:
            ValueError: If the function tool with the given name is not found.
            Exception: Any exception raised by the underlying function tool.

        Example:
            >>> store = FunctionCallStore()
            >>> tool = FunctionTool.from_callable(lambda x, y: x + y)
            >>> store.add_function_tool(tool)
            >>> result = store.call_function_tool("lambda", 5, 3)
            >>> print(result)  # Output: 8
        """
        return run_sync(self.call_function_tool_async(name, *args, **kwargs))

    async def call_function_tool_async(
        self, name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Asynchronously call a function tool with the given name and arguments.

        This is the core method for executing function tools. It looks up the tool
        by name, validates its existence, and calls it with the provided arguments.
        The call is not automatically tracked - use add_function_call to record it.

        Args:
            name: The name of the function tool to call.
            *args: Positional arguments to pass to the function tool.
            **kwargs: Keyword arguments to pass to the function tool.

        Returns:
            The result of calling the function tool.

        Raises:
            ValueError: If the function tool with the given name is not found
                       or if the tool definition is None.
            Exception: Any exception raised by the underlying function tool.

        Example:
            >>> async def example():
            ...     store = FunctionCallStore()
            ...     tool = FunctionTool.from_callable(async def multiply(x, y): return x * y)
            ...     store.add_function_tool(tool)
            ...     result = await store.call_function_tool_async("multiply", 4, 5)
            ...     print(result)  # Output: 20
        """

        if name not in self.store:
            raise ValueError(f"Function tool {name} not found")

        function_tool = self.store[name].function_tool

        if function_tool is None:
            raise ValueError(f"Function tool {name} not found")

        return await function_tool.call_async(*args, **kwargs)

    def add_function_tool(self, function_tool: FunctionTool) -> None:
        """
        Add or update a function tool in the store.

        If a tool with the same name already exists, it will be updated with
        the new tool definition. If it's a new tool, a new ToolPair will be
        created with the tool and an empty call history.

        Args:
            function_tool: The function tool to add or update.

        Example:
            >>> store = FunctionCallStore()
            >>> tool1 = FunctionTool.from_callable(lambda x: x * 2)
            >>> tool2 = FunctionTool.from_callable(lambda x: x * 3)
            >>> store.add_function_tool(tool1)
            >>> store.add_function_tool(tool2)  # Updates existing tool
            >>> assert store.retrieve_function_tool("lambda") == tool2
        """
        if function_tool.name not in self.store:
            self.store[function_tool.name] = ToolPair(
                function_tool=function_tool, function_calls=None
            )
        else:
            self.store[function_tool.name].change_function_tool(function_tool)

    def add_function_call(self, function_call: FunctionToolCall) -> None:
        """
        Add a function call to the call history for a tool.

        If no tool with the given name exists, a new ToolPair will be created
        with only the function call (no tool definition). If the tool exists,
        the call will be appended to its call history.

        Args:
            function_call: The function call to add to the history.

        Example:
            >>> store = FunctionCallStore()
            >>> call1 = FunctionToolCall(name="test_tool", arguments={"x": 1})
            >>> call2 = FunctionToolCall(name="test_tool", arguments={"x": 2})
            >>> store.add_function_call(call1)
            >>> store.add_function_call(call2)
            >>> calls = store.retrieve_function_calls("test_tool")
            >>> assert len(calls) == 2
        """
        if function_call.name not in self.store:
            self.store[function_call.name] = ToolPair(
                function_tool=None, function_calls=[function_call]
            )
        else:
            self.store[function_call.name].change_function_call(function_call)

    def retrieve_function_tool(self, name: str) -> FunctionTool | None:
        """
        Retrieve a function tool by its name.

        Args:
            name: The name of the function tool to retrieve.

        Returns:
            The function tool if found, None otherwise.

        Example:
            >>> store = FunctionCallStore()
            >>> tool = FunctionTool.from_callable(lambda x: x + 1)
            >>> store.add_function_tool(tool)
            >>> retrieved = store.retrieve_function_tool("lambda")
            >>> assert retrieved == tool
            >>> missing = store.retrieve_function_tool("nonexistent")
            >>> assert missing is None
        """
        if name not in self.store:
            return None

        return self.store[name].function_tool

    def retrieve_function_calls(self, name: str) -> list[FunctionToolCall] | None:
        """
        Retrieve all function calls for a given tool name.

        Args:
            name: The name of the function tool.

        Returns:
            List of function calls in chronological order, or None if not found.

        Example:
            >>> store = FunctionCallStore()
            >>> call1 = FunctionToolCall(name="test", arguments={"x": 1})
            >>> call2 = FunctionToolCall(name="test", arguments={"x": 2})
            >>> store.add_function_call(call1)
            >>> store.add_function_call(call2)
            >>> calls = store.retrieve_function_calls("test")
            >>> assert len(calls) == 2
            >>> assert calls[0] == call1
            >>> assert calls[1] == call2
        """
        if name not in self.store:
            return None

        return self.store[name].function_calls

    def retrieve_function_call(
        self, name: str, index: int = -1
    ) -> FunctionToolCall | None:
        """
        Retrieve a specific function call by name and index.

        Args:
            name: The name of the function tool.
            index: The index of the function call to retrieve. Defaults to -1 (last call).
                   Use negative indices to count from the end.

        Returns:
            The function call at the specified index, or None if not found or invalid index.

        Example:
            >>> store = FunctionCallStore()
            >>> call1 = FunctionToolCall(name="test", arguments={"x": 1})
            >>> call2 = FunctionToolCall(name="test", arguments={"x": 2})
            >>> call3 = FunctionToolCall(name="test", arguments={"x": 3})
            >>> store.add_function_call(call1)
            >>> store.add_function_call(call2)
            >>> store.add_function_call(call3)
            >>> last_call = store.retrieve_function_call("test")  # Gets call3
            >>> first_call = store.retrieve_function_call("test", 0)  # Gets call1
            >>> second_call = store.retrieve_function_call("test", 1)  # Gets call2
        """
        function_calls = self.retrieve_function_calls(name)
        if function_calls is None or not function_calls:
            return None

        try:
            return function_calls[index]
        except IndexError:
            return None

    def get_all_tool_names(self) -> list[str]:
        """
        Get all registered tool names in the store.

        Returns:
            List of all tool names currently in the store.

        Example:
            >>> store = FunctionCallStore()
            >>> tool1 = FunctionTool.from_callable(lambda x: x + 1)
            >>> tool2 = FunctionTool.from_callable(lambda x: x * 2)
            >>> store.add_function_tool(tool1)
            >>> store.add_function_tool(tool2)
            >>> names = store.get_all_tool_names()
            >>> assert "lambda" in names
            >>> assert len(names) == 2
        """
        return list(self.store.keys())

    def get_call_count(self, name: str) -> int:
        """
        Get the number of times a tool was called.

        Args:
            name: The name of the function tool.

        Returns:
            Number of calls for the tool, 0 if not found or no calls.

        Example:
            >>> store = FunctionCallStore()
            >>> call1 = FunctionToolCall(name="test", arguments={"x": 1})
            >>> call2 = FunctionToolCall(name="test", arguments={"x": 2})
            >>> store.add_function_call(call1)
            >>> store.add_function_call(call2)
            >>> count = store.get_call_count("test")
            >>> assert count == 2
            >>> missing_count = store.get_call_count("nonexistent")
            >>> assert missing_count == 0
        """
        if name not in self.store:
            return 0

        function_calls = self.store[name].function_calls
        return len(function_calls) if function_calls else 0

    def remove_function_tool(self, name: str) -> bool:
        """
        Remove a function tool and all its calls from the store.

        Args:
            name: The name of the function tool to remove.

        Returns:
            True if the tool was removed, False if it wasn't found.

        Example:
            >>> store = FunctionCallStore()
            >>> tool = FunctionTool.from_callable(lambda x: x + 1)
            >>> store.add_function_tool(tool)
            >>> store.add_function_call(FunctionToolCall(name="lambda", arguments={"x": 1}))
            >>> removed = store.remove_function_tool("lambda")
            >>> assert removed is True
            >>> assert store.retrieve_function_tool("lambda") is None
            >>> not_found = store.remove_function_tool("nonexistent")
            >>> assert not_found is False
        """
        if name not in self.store:
            return False

        del self.store[name]
        return True

    def remove_call(self, name: str, index: int) -> bool:
        """
        Remove a specific function call by name and index.

        Args:
            name: The name of the function tool.
            index: The index of the call to remove. Use negative indices to count from the end.

        Returns:
            True if the call was removed, False if not found or invalid index.

        Example:
            >>> store = FunctionCallStore()
            >>> call1 = FunctionToolCall(name="test", arguments={"x": 1})
            >>> call2 = FunctionToolCall(name="test", arguments={"x": 2})
            >>> call3 = FunctionToolCall(name="test", arguments={"x": 3})
            >>> store.add_function_call(call1)
            >>> store.add_function_call(call2)
            >>> store.add_function_call(call3)
            >>> removed = store.remove_call("test", 1)  # Remove call2
            >>> assert removed is True
            >>> calls = store.retrieve_function_calls("test")
            >>> assert len(calls) == 2
            >>> assert calls[0] == call1
            >>> assert calls[1] == call3
        """
        if name not in self.store:
            return False

        function_calls = self.store[name].function_calls
        if function_calls is None or not function_calls:
            return False

        try:
            function_calls.pop(index)
            return True
        except IndexError:
            return False

    def clear_all_calls(self) -> None:
        """
        Clear all function calls from all tools, but keep the tools themselves.
        """
        for tool_pair in self.store.values():
            tool_pair.function_calls = None

    def is_function_tool_present(self, function_tool: FunctionTool) -> bool:
        """
        Check if a function tool is present in the store.

        Args:
            function_tool: The function tool to check for.

        Returns:
            True if the tool is present, False otherwise.
        """
        if function_tool.name not in self.store:
            return False

        stored_tool = self.store[function_tool.name].function_tool
        return stored_tool is not None

    def get_tools_with_calls(self) -> list[str]:
        """
        Get names of tools that have been called at least once.

        Returns:
            List of tool names that have function calls.
        """
        return [
            name
            for name, tool_pair in self.store.items()
            if tool_pair.function_calls and len(tool_pair.function_calls) > 0
        ]

    def get_tools_without_calls(self) -> list[str]:
        """
        Get names of tools that have never been called.

        Returns:
            List of tool names that have no function calls.
        """
        return [
            name
            for name, tool_pair in self.store.items()
            if not tool_pair.function_calls or len(tool_pair.function_calls) == 0
        ]

    def get_total_call_count(self) -> int:
        """
        Get total number of function calls across all tools.

        Returns:
            Total number of function calls in the store.
        """
        return sum(
            len(tool_pair.function_calls) if tool_pair.function_calls else 0
            for tool_pair in self.store.values()
        )

    def get_most_called_tool(self) -> tuple[str, int] | None:
        """
        Get the most frequently called tool and its call count.

        Returns:
            Tuple of (tool_name, call_count) for the most called tool, or None if no calls exist.
        """
        if not self.store:
            return None

        max_calls = 0
        most_called_tool = None

        for name, tool_pair in self.store.items():
            call_count = (
                len(tool_pair.function_calls) if tool_pair.function_calls else 0
            )
            if call_count > max_calls:
                max_calls = call_count
                most_called_tool = name

        return (most_called_tool, max_calls) if most_called_tool is not None else None


if __name__ == "__main__":

    def add(a: int, b: int) -> int:
        return a + b

    example = FunctionTool.from_callable(add)
    function_call_store = FunctionCallStore()
    function_call_store.add_function_tool(example)

    result = function_call_store.call_function_tool("add", 1, 2)
    print(result)
