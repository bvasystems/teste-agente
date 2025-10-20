from typing import Any
from rsb.models import BaseModel, Field

from agentle.responses.definitions.function_tool import FunctionTool
from agentle.responses.definitions.function_tool_call import FunctionToolCall


class ToolPair(BaseModel):
    function_tool: FunctionTool | None = Field(
        default=None, description="The function tool."
    )
    function_calls: list[FunctionToolCall] | None = Field(
        default=None,
        description="The list of function calls in sequence they were called.",
    )

    def change_function_tool(self, function_tool: FunctionTool) -> None:
        self.function_tool = function_tool

    def change_function_call(self, function_call: FunctionToolCall) -> None:
        if self.function_calls is None:
            self.function_calls = []
        self.function_calls.append(function_call)


class FunctionCallStore(BaseModel):
    store: dict[str, ToolPair] = Field(
        ..., description="The store of function tools and calls."
    )

    async def call_function_tool_async(
        self, name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Call a function tool with the given name and arguments.

        Args:
            name: The name of the function tool to call.
            *args: The arguments to pass to the function tool.
            **kwargs: The keyword arguments to pass to the function tool.

        Returns:
            The result of calling the function tool.
        """

        if name not in self.store:
            raise ValueError(f"Function tool {name} not found")

        function_tool = self.store[name].function_tool

        if function_tool is None:
            raise ValueError(f"Function tool {name} not found")

        return await function_tool.call_async(*args, **kwargs)

    def add_function_tool(self, function_tool: FunctionTool) -> None:
        if function_tool.name not in self.store:
            self.store[function_tool.name] = ToolPair(
                function_tool=function_tool, function_calls=None
            )
        else:
            self.store[function_tool.name].change_function_tool(function_tool)

    def add_function_call(self, function_call: FunctionToolCall) -> None:
        if function_call.name not in self.store:
            self.store[function_call.name] = ToolPair(
                function_tool=None, function_calls=[function_call]
            )
        else:
            self.store[function_call.name].change_function_call(function_call)

    def retrieve_function_tool(self, name: str) -> FunctionTool | None:
        if name not in self.store:
            return None

        return self.store[name].function_tool

    def retrieve_function_calls(self, name: str) -> list[FunctionToolCall] | None:
        """
        Retrieve all function calls for a given name.

        Args:
            name: The name of the function tool.

        Returns:
            List of function calls, or None if not found.
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

        Returns:
            The function call at the specified index, or None if not found.
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
            List of all tool names.
        """
        return list(self.store.keys())

    def get_call_count(self, name: str) -> int:
        """
        Get the number of times a tool was called.

        Args:
            name: The name of the function tool.

        Returns:
            Number of calls for the tool, 0 if not found or no calls.
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
            index: The index of the call to remove.

        Returns:
            True if the call was removed, False if not found or invalid index.
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
