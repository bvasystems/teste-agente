from pydantic.types import StrictInt, StrictStr


AnyVariants = list[StrictStr] | list[StrictInt]
