import logging
from agentle.generations.providers.amazon.bedrock_generation_provider import (
    BedrockGenerationProvider,
)


logging.basicConfig(level=logging.CRITICAL)

provider = BedrockGenerationProvider()

generation = provider.create_generation_by_prompt("Hello!")

print(generation)
