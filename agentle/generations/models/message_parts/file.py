"""
Module for file-based message parts.
"""

import mimetypes
from typing import Literal
import base64 as b64
from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class FilePart(BaseModel):
    """
    Represents a file attachment part of a message.

    This class handles binary file data with appropriate MIME type validation.
    """

    type: Literal["file"] = Field(
        default="file",
        description="Discriminator field to identify this as a file message part.",
    )

    data: bytes | str = Field(
        description="The binary content of the file. or the Base64 encoded contents"
    )

    mime_type: str = Field(
        description="The MIME type of the file, must be a valid MIME type from Python's mimetypes module."
    )

    @property
    def text(self) -> str:
        """
        Returns a text representation of the file part.

        Returns:
            str: A text representation containing the MIME type.
        """
        return f"<file>\n{self.mime_type}\n </file>"

    @property
    def base64(self) -> str:
        """
        Returns the base64 encoded representation of the file data.

        Returns:
            str: Base64 encoded string of the file data.
        """
        if isinstance(self.data, bytes):
            # If data is bytes, encode it to base64
            return b64.b64encode(self.data).decode("utf-8")

        # If data is already a string, assume it's already base64 encoded
        # Validate it's valid base64 by trying to decode and re-encode
        try:
            # Test if it's valid base64
            b64.b64decode(self.data, validate=True)
            return self.data
        except Exception:
            # If not valid base64, assume it's a regular string and encode it
            return b64.b64encode(self.data.encode("utf-8")).decode("utf-8")

    def __str__(self) -> str:
        return self.text

    def __post_init__(self) -> None:
        """
        Validates that the provided MIME type is official.

        Raises:
            ValueError: If the MIME type is not in the list of official MIME types.
        """
        allowed_mimes = mimetypes.types_map.values()
        mime_type_unknown = self.mime_type not in allowed_mimes
        if mime_type_unknown:
            raise ValueError(
                f"The provided MIME ({self.mime_type}) is not in the list of official mime types: {allowed_mimes}"
            )
