from typing import Any
from rsb.models import BaseModel, Field

from agentle.responses.definitions.function_tool import FunctionTool
from agentle.responses.definitions.function_tool_call import FunctionToolCall


class ToolDuo(BaseModel):
    function_tool: FunctionTool | None = Field(
        default=None, description="The function tool."
    )
    function_call: FunctionToolCall | None = Field(
        default=None, description="The function call."
    )

    def change_function_tool(self, function_tool: FunctionTool) -> None:
        self.function_tool = function_tool

    def change_function_call(self, function_call: FunctionToolCall) -> None:
        self.function_call = function_call


class FunctionCallStore(BaseModel):
    store: dict[str, ToolDuo] = Field(
        ..., description="The store of function tools and calls."
    )

    async def call_function_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
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
            self.store[function_tool.name] = ToolDuo(
                function_tool=function_tool, function_call=None
            )
        else:
            self.store[function_tool.name].change_function_tool(function_tool)

    def add_function_call(self, function_call: FunctionToolCall) -> None:
        if function_call.name not in self.store:
            self.store[function_call.name] = ToolDuo(
                function_tool=None, function_call=function_call
            )
        else:
            self.store[function_call.name].change_function_call(function_call)

    def retrieve_function_tool(self, name: str) -> FunctionTool | None:
        if name not in self.store:
            return None

        return self.store[name].function_tool

    def retrieve_function_call(self, name: str) -> FunctionToolCall | None:
        if name not in self.store:
            return None

        return self.store[name].function_call
