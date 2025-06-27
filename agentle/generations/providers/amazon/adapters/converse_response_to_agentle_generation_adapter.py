from dataclasses import dataclass, field
from typing import override
from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef
from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.generation import Generation


@dataclass(frozen=True)
class ConverseResponseToAgentleGenerationAdapter[T](
    Adapter[ConverseResponseTypeDef, Generation[T]]
):
    response_schema: type[T] | None = field(default=None)

    @override
    def adapt(self, _f: ConverseResponseTypeDef) -> Generation[T]:
        return Generation.mock()
