"""
Module defining the DeveloperMessage class representing messages from developers.
"""

from __future__ import annotations

from typing import Literal, Sequence

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import CacheControl, TextPart
from agentle.generations.tools.tool import Tool


@valueobject
class DeveloperMessage(BaseModel):
    """
    Represents a message from a developer in the system.

    This class can contain a sequence of different message parts including
    text, files, and tools.
    """

    role: Literal["developer"] = Field(
        default="developer",
        description="Discriminator field to identify this as a developer message. Always set to 'developer'.",
    )

    parts: Sequence[TextPart | FilePart | Tool] = Field(
        description="The sequence of message parts that make up this developer message.",
    )

    def cache_text_parts(self) -> DeveloperMessage:
        parts: list[TextPart | FilePart | Tool] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                parts.append(
                    TextPart(
                        text=part.text, cache_control=CacheControl(type="ephemeral")
                    )
                )
            else:
                parts.append(part)

        return DeveloperMessage(
            role=self.role,
            parts=parts,
        )

    @property
    def text(self) -> str:
        """
        Returns the concatenated text representation of all parts in this message.

        Returns:
            str: The concatenated text of all message parts.
        """
        return "".join(
            part.text if isinstance(part.text, str) else part.text.text
            for part in self.parts
        )
