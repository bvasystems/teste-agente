from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


class AgentleToolToOpenaiToolAdapter(Adapter[Tool, "ChatCompletionToolParam"]):
    def adapt(self, tool: Tool) -> ChatCompletionToolParam:
        from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
        from openai.types.shared_params.function_definition import FunctionDefinition

        # Generate valid JSON schema for OpenAI parameters
        properties = {}
        required = []
        
        # Check if parameters is already a full JSON schema object
        original_parameters = tool.parameters or {}
        if original_parameters.get("type") == "object" and "properties" in original_parameters:
            parameters_schema = original_parameters
        else:
            # Convert raw parameter dictionary to JSON schema format
            for param_name, param_info in original_parameters.items():
                if isinstance(param_info, dict):
                    prop_info = param_info.copy()
                    
                    # Convert Python type names to JSON schema types
                    type_mapping = {
                        "str": "string", 
                        "int": "integer", 
                        "float": "number", 
                        "bool": "boolean",
                        "dict": "object",
                        "list": "array"
                    }
                    if "type" in prop_info and prop_info["type"] in type_mapping:
                        prop_info["type"] = type_mapping[prop_info["type"]]
                    elif "type" not in prop_info or prop_info["type"] == "object":
                        prop_info["type"] = "string" # Default fallback
                        
                    # Extract required flag
                    is_required = prop_info.pop("required", False)
                    if is_required:
                        required.append(param_name)
                    
                    properties[param_name] = prop_info
                else:
                    # Fallback for unexpected formats
                    properties[param_name] = {"type": "string"}
            
            from typing import Any
            parameters_schema: dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                parameters_schema["required"] = required

        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description or "",
                parameters=parameters_schema,
            ),
            type="function",
        )
