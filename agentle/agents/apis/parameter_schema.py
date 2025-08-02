from agentle.agents.apis.array_schema import ArraySchema
from agentle.agents.apis.object_schema import ObjectSchema
from agentle.agents.apis.primitive_schema import PrimitiveSchema


ParameterSchema = ObjectSchema | ArraySchema | PrimitiveSchema
