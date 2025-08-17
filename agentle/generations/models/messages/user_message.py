"""
Module defining the UserMessage class representing messages from users.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool import Tool
from agentle.generations.tools.tool_execution_result import ToolExecutionResult


class UserMessage(BaseModel):
    """
    Represents a message from a user in the system.

    This class can contain a sequence of different message parts including
    text, files, tools, and tool execution suggestions.
    """

    role: Literal["user"] = Field(
        default="user",
        description="Discriminator field to identify this as a user message. Always set to 'user'.",
    )

    parts: Sequence[
        TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion | ToolExecutionResult
    ] = Field(
        description="The sequence of message parts that make up this user message.",
    )

    def insert_part(
        self,
        parts: TextPart
        | FilePart
        | Tool[Any]
        | ToolExecutionSuggestion
        | ToolExecutionResult
        | list[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ],
        /,
        index: int = 0,
    ) -> None:
        _self_parts = list(self.parts)
        if isinstance(parts, list):
            for part in parts:
                _self_parts.insert(index, part)  # type: ignore[reportArgumentType]
            return

        _self_parts.insert(index, parts)  # type: ignore[reportArgumentType]
        self.parts = _self_parts

    def append_part(
        self,
        parts: TextPart
        | FilePart
        | Tool[Any]
        | ToolExecutionSuggestion
        | ToolExecutionResult
        | Sequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ],
        /,
    ) -> None:
        _self_parts = list(self.parts)
        if isinstance(parts, Sequence):
            _self_parts.extend(parts)  # type: ignore[reportArgumentType]
            return

        _self_parts.append(parts)  # type: ignore[reportArgumentType]
        self.parts = _self_parts

    @property
    def text(self) -> str:
        return "".join([str(p.text) for p in self.parts])

    @classmethod
    def create_named(
        cls,
        parts: Sequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ],
        name: str | None = None,
    ) -> UserMessage:
        """
        Creates a user message with a name identifier.

        Args:
            parts: The sequence of message parts to include in the message.
            name: Optional name to identify the user sending this message.

        Returns:
            A UserMessage instance with the name prepended if provided.
        """
        if name is None:
            return cls(role="user", parts=parts)

        return cls(
            role="user",
            parts=[TextPart(text=f"[{name}]: ")] + list(parts),
        )

    def merge_text_parts(self, other: UserMessage) -> UserMessage:
        """
        Merges text parts from another UserMessage with this one.

        This method takes all TextPart instances from the other UserMessage,
        concatenates their text, and appends it to the last TextPart in this
        message. If this message has no TextParts, a new TextPart is created
        with the concatenated text from the other message.

        Args:
            other: Another UserMessage whose text parts will be merged into this one.

        Returns:
            A new UserMessage with the merged text parts and all other parts preserved.

        Example:
            ```python
            # Original message with text and file parts
            msg1 = UserMessage(parts=[
                TextPart(text="Hello "),
                FilePart(file_path="document.pdf"),
                TextPart(text="world")
            ])

            # Other message with text parts
            msg2 = UserMessage(parts=[
                TextPart(text=" and "),
                TextPart(text="universe!")
            ])

            # Merge text parts
            merged = msg1.merge_text_parts(msg2)
            # Result: [TextPart("Hello "), FilePart("document.pdf"), TextPart("world and universe!")]
            ```
        """
        # Extract text parts from the other message
        other_text_parts = [p for p in other.parts if isinstance(p, TextPart)]

        # If other has no text parts, return a copy of self
        if not other_text_parts:
            return UserMessage(role="user", parts=list(self.parts))

        # Concatenate all text from other message's text parts
        other_text = "".join([str(tp.text) for tp in other_text_parts])

        # Create a copy of self parts to work with
        new_parts = list(self.parts)

        # Find the last text part in self.parts
        last_text_part_index = -1
        for i in range(len(new_parts) - 1, -1, -1):
            if isinstance(new_parts[i], TextPart):
                last_text_part_index = i
                break

        if last_text_part_index >= 0:
            # Merge with the last text part
            last_text_part = new_parts[last_text_part_index]
            merged_text_part = TextPart(text=str(other_text) + str(last_text_part.text))

            # Replace the last text part with the merged one
            new_parts[last_text_part_index] = merged_text_part
        else:
            # No text parts in self, append a new text part with other's text
            new_parts.append(TextPart(text=other_text))

        return UserMessage(role="user", parts=new_parts)

    def __add__(self, other: UserMessage) -> UserMessage:
        """
        Combines two UserMessage instances by concatenating their parts.

        Args:
            other: Another UserMessage instance to combine with this one.

        Returns:
            A new UserMessage with parts from both messages combined.

        Raises:
            TypeError: If other is not a UserMessage instance.
        """
        # Combine parts from both messages
        combined_parts = list(self.parts) + list(other.parts)

        return UserMessage(role="user", parts=combined_parts)


if __name__ == "__main__":
    m = UserMessage(parts=[TextPart(text="hello")])
