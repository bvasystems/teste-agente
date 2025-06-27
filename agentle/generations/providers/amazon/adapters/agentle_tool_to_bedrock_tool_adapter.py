from typing import override
from rsb.adapters.adapter import Adapter

from agentle.generations.providers.amazon.models.tool_input_schema import (
    ToolInputSchema,
)
from agentle.generations.providers.amazon.models.tool_specification import (
    ToolSpecification,
)
from agentle.generations.tools.tool import Tool
from agentle.generations.providers.amazon.models.tool import Tool as BedrockTool


class AgentleToolToBedrockToolAdapter(Adapter[Tool, BedrockTool]):
    @override
    def adapt(self, _f: Tool) -> BedrockTool:
        return BedrockTool(
            toolSpec=ToolSpecification(
                name=_f.name,
                description=_f.description or "",
                inputSchema=ToolInputSchema(json=_f.parameters),
            )
        )
