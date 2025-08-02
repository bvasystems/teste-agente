from typing import Any
from agentle.agents.apis.array_schema import ArraySchema
from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.object_schema import ObjectSchema
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.parameter_schema import ParameterSchema
from agentle.agents.apis.primitive_schema import PrimitiveSchema


def string_param(
    name: str,
    description: str,
    required: bool = False,
    enum: list[str] | None = None,
    default: str | None = None,
    location: ParameterLocation = ParameterLocation.QUERY,
) -> EndpointParameter:
    """Create a string parameter."""
    return EndpointParameter(
        name=name,
        description=description,
        parameter_schema=PrimitiveSchema(type="string", enum=enum),
        location=location,
        required=required,
        default=default,
    )


def integer_param(
    name: str,
    description: str,
    required: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
    default: int | None = None,
    location: ParameterLocation = ParameterLocation.QUERY,
) -> EndpointParameter:
    """Create an integer parameter."""
    return EndpointParameter(
        name=name,
        description=description,
        parameter_schema=PrimitiveSchema(
            type="integer", minimum=minimum, maximum=maximum
        ),
        location=location,
        required=required,
        default=default,
    )


def object_param(
    name: str,
    description: str,
    properties: dict[str, ParameterSchema],
    required_props: list[str] | None = None,
    required: bool = False,
    location: ParameterLocation = ParameterLocation.BODY,
    example: dict[str, Any] | None = None,
) -> EndpointParameter:
    """Create an object parameter with proper schema."""
    return EndpointParameter(
        name=name,
        description=description,
        parameter_schema=ObjectSchema(
            properties=properties, required=required_props or [], example=example
        ),
        location=location,
        required=required,
    )


def array_param(
    name: str,
    description: str,
    item_schema: ParameterSchema,
    required: bool = False,
    min_items: int | None = None,
    max_items: int | None = None,
    location: ParameterLocation = ParameterLocation.QUERY,
    example: list[Any] | None = None,
) -> EndpointParameter:
    """Create an array parameter."""
    return EndpointParameter(
        name=name,
        description=description,
        parameter_schema=ArraySchema(
            items=item_schema, min_items=min_items, max_items=max_items, example=example
        ),
        location=location,
        required=required,
    )
