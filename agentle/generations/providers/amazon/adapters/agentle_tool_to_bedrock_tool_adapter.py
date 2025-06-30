from collections.abc import Mapping, MutableMapping, MutableSequence
from typing import Any, override
from rsb.adapters.adapter import Adapter

from agentle.generations.providers.amazon.models.tool_input_schema import (
    ToolInputSchema,
)
from agentle.generations.providers.amazon.models.tool_specification import (
    ToolSpecification,
)
from agentle.generations.tools.tool import Tool
from agentle.generations.providers.amazon.models.tool import Tool as BedrockTool


# Type alias for parameter info structure
ParameterInfo = Mapping[str, Any]


class AgentleToolToBedrockToolAdapter(Adapter[Tool, BedrockTool]):
    @override
    def adapt(self, _f: Tool) -> BedrockTool:
        # Extrair propriedades e required da estrutura atual
        properties: Mapping[str, Mapping[str, Any]] = {}
        required: MutableSequence[str] = []

        type_mapping: Mapping[str, str] = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
            "any": "string",
            "null": "null",
            "none": "null",
        }

        # Ensure parameters exist and is a dict
        if hasattr(_f, "parameters"):
            for param_name, param_info in _f.parameters.items():
                # Ensure param_info is a dictionary
                if not isinstance(param_info, dict):
                    # If param_info is not a dict, create a minimal valid parameter
                    properties[param_name] = {
                        "type": "string",
                        "description": f"Parameter {param_name}",
                    }
                    continue

                bedrock_param: MutableMapping[str, Any] = {}

                # Mapear tipo com type safety
                param_type = param_info.get("type", "string")
                if isinstance(param_type, str):
                    # Handle potential type names case-insensitively
                    normalized_type = param_type.lower()
                    bedrock_param["type"] = type_mapping.get(normalized_type, "string")
                else:
                    # If type is not a string, default to string
                    bedrock_param["type"] = "string"

                # Adicionar descrição se disponível
                description = param_info.get("description")
                if isinstance(description, str):
                    bedrock_param["description"] = description
                else:
                    # Bedrock permite descrição vazia ou pode usar o nome do parâmetro
                    bedrock_param["description"] = f"Parameter {param_name}"

                # Adicionar valor padrão se disponível
                if "default" in param_info:
                    default_value = param_info["default"]
                    # Ensure the default value is JSON-serializable
                    if default_value is not None:
                        bedrock_param["default"] = default_value

                properties[param_name] = bedrock_param

                # Adicionar à lista de required se necessário
                # Handle different ways required might be specified
                is_required = param_info.get("required", False)
                if isinstance(is_required, bool) and is_required:
                    required.append(param_name)
                elif "default" not in param_info:
                    # If no default is provided, consider it required
                    required.append(param_name)

        # Garantir que sempre temos um schema válido, mesmo sem parâmetros
        json_schema: Mapping[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        # Handle potential None values for name and description
        tool_name = _f.name if hasattr(_f, "name") and _f.name else "unnamed_tool"
        tool_description = ""
        if hasattr(_f, "description") and _f.description:
            tool_description = _f.description

        return BedrockTool(
            toolSpec=ToolSpecification(
                name=tool_name,
                description=tool_description,
                inputSchema=ToolInputSchema(json=json_schema),
            )
        )
