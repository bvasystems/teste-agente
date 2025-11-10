from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.primitive_schema import PrimitiveSchema


def boolean_param(
    name: str,
    description: str,
    required: bool = False,
    default: bool | None = None,
    location: ParameterLocation = ParameterLocation.QUERY,
) -> EndpointParameter:
    """Create a boolean parameter.

    Args:
        name: Parameter name
        description: Parameter description
        required: Whether the parameter is required
        default: Default value for the parameter
        location: Where the parameter should be placed in the request

    Returns:
        EndpointParameter configured for boolean values

    Example:
        ```python
        from agentle.agents.apis.params.boolean_param import boolean_param

        boolean_param(
            name="enabled",
            description="Enable feature",
            required=False,
            default=True
        )
        ```
    """
    return EndpointParameter(
        name=name,
        description=description,
        parameter_schema=PrimitiveSchema(type="boolean"),
        location=location,
        required=required,
        default=default,
    )
