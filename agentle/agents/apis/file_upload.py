"""File upload support for endpoints."""

from __future__ import annotations

import mimetypes
from rsb.models.base_model import BaseModel


class FileUpload(BaseModel):
    """Represents a file to be uploaded."""

    filename: str
    content: bytes
    mime_type: str | None = None

    def to_form_part(self) -> tuple[str, bytes, str]:
        """Convert to multipart form part."""
        mime = (
            self.mime_type
            or mimetypes.guess_type(self.filename)[0]
            or "application/octet-stream"
        )
        return (self.filename, self.content, mime)
