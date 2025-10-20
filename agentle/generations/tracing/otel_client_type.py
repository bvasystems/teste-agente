from typing import Annotated

from rsb.models.field import Field

from agentle.generations.tracing.langfuse_otel_client import LangfuseOtelClient
from agentle.generations.tracing.no_op_otel_client import NoOpOtelClient

OtelClientType = Annotated[
    LangfuseOtelClient | NoOpOtelClient, Field(discriminator="type")
]
