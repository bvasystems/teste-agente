from agentle.agents.apis.endpoint_parameter import EndpointParameter
from agentle.agents.apis.parameter_location import ParameterLocation
from agentle.agents.apis.primitive_schema import PrimitiveSchema


def number_param(
    name: str,
    description: str,
    required: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    default: float | None = None,
    location: ParameterLocation = ParameterLocation.QUERY,
    format: str | None = None,
) -> EndpointParameter:
    """Create a number (float/decimal) parameter.

    Args:
        name: Parameter name
        description: Parameter description
        required: Whether the parameter is required
        minimum: Minimum allowed value
        maximum: Maximum allowed value
        default: Default value for the parameter
        location: Where the parameter should be placed in the request
        format: Format hint (e.g., 'float', 'double', 'decimal')

    Returns:
        EndpointParameter configured for number values

    Example:
        ```python
        from agentle.agents.apis.params.number_param import number_param

        number_param(
            name="price",
            description="Product price",
            required=True,
            minimum=0.0,
            default=99.99
        )
        ```
    """
    return EndpointParameter(
        name=name,
        description=description,
        parameter_schema=PrimitiveSchema(
            type="number",
            minimum=minimum,
            maximum=maximum,
            format=format,
        ),
        location=location,
        required=required,
        default=default,
    )
