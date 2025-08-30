from __future__ import annotations

from typing import Literal, Optional

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.parsing.parsed_file import ParsedFile
from agentle.utils.file_validation import (
    FileValidationError,
    is_file_path,
    is_url,
    validate_content_type,
)

NO_CACHE = None

type NoCache = None


class StaticKnowledge(BaseModel):
    """
    Static knowledge is a collection of knowledge that is provided to the agent at the time of creation.
    """

    content: str = Field(
        description="""The content of the knowledge.
        can be a url, a local file path, or a string of text."""
    )

    cache: int | NoCache | Literal["infinite"] = Field(
        default=NO_CACHE,
        description="The cache time of the knowledge. If None, the knowledge is not cached. If 'infinite', the knowledge is cached indefinitely.",
    )

    parse_timeout: float = Field(default=30)
    """The timeout for the parse operation in seconds."""

    @classmethod
    def from_parsed_file(
        cls,
        parsed_file: ParsedFile,
        cache: int | NoCache | Literal["infinite"] = NO_CACHE,
    ) -> StaticKnowledge:
        return cls(content=parsed_file.md, cache=cache)

    def is_url(self) -> bool:
        """Check if the content is a URL.

        Returns:
            True if content is a valid URL, False otherwise
        """
        return is_url(self.content)

    def is_file_path(self, base_path: Optional[str] = None) -> bool:
        """Check if the content is a valid file path that exists.

        Args:
            base_path: Optional base path for resolving relative paths

        Returns:
            True if content is a valid existing file path, False otherwise
        """
        return is_file_path(self.content, base_path)

    def is_raw_text(self, base_path: Optional[str] = None) -> bool:
        """Check if the content is raw text (not a URL or file path).

        Args:
            base_path: Optional base path for resolving relative paths

        Returns:
            True if content is raw text, False otherwise
        """
        return not self.is_url() and not self.is_file_path(base_path)

    def validate_and_resolve(self, base_path: Optional[str] = None) -> tuple[str, str]:
        """Validate the content and return its type and resolved form.

        Args:
            base_path: Optional base path for resolving relative file paths

        Returns:
            Tuple of (content_type, resolved_content) where:
            - content_type is one of: 'url', 'file_path', 'raw_text'
            - resolved_content is the original content or resolved file path

        Raises:
            FileNotFoundError: If content appears to be a file path but doesn't exist
            InvalidPathError: If content appears to be a file path but is invalid
        """
        try:
            return validate_content_type(self.content, base_path)
        except FileValidationError as e:
            # Re-raise with additional context about the StaticKnowledge instance
            raise type(e)(f"StaticKnowledge validation failed: {str(e)}", e.path) from e

    def __str__(self) -> str:
        return self.content
