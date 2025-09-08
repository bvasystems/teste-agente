"""
Module for file-based message parts with enhanced validation.
"""

import base64 as b64
import mimetypes
import io
import zipfile
import tarfile
import json
from typing import Any, Literal, Protocol, TYPE_CHECKING, override

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType


class _ImageModule(Protocol):
    """Protocol for PIL Image module."""

    @staticmethod
    def open(fp: str | bytes | io.BytesIO) -> "PILImageType": ...


class _MagicModule(Protocol):
    """Protocol for python-magic module."""

    @staticmethod
    def from_buffer(buffer: bytes, mime: bool = False) -> str: ...


class _PyPDF2Module(Protocol):
    """Protocol for PyPDF2 module."""

    class PdfReader:
        def __init__(self, stream: str | io.BytesIO) -> None: ...


class _SoundFileModule(Protocol):
    """Protocol for soundfile module."""

    @staticmethod
    def info(file: str | io.BytesIO) -> Any: ...


class _OptionalDependencies:
    """Type-safe handler for optional dependency imports."""

    def __init__(self) -> None:
        self._pil_image: _ImageModule | None = self._try_import_pil()
        self._magic: _MagicModule | None = self._try_import_magic()
        self._pypdf2: _PyPDF2Module | None = self._try_import_pypdf2()
        self._soundfile: _SoundFileModule | None = self._try_import_soundfile()

    def _try_import_pil(self) -> _ImageModule | None:
        try:
            from PIL import Image

            return Image  # type: ignore[return-value]
        except ImportError:
            return None

    def _try_import_magic(self) -> _MagicModule | None:            
        try:
            import magic

            return magic  # type: ignore[return-value]
        except ImportError:
            return None

    def _try_import_pypdf2(self) -> _PyPDF2Module | None:
        try:
            import PyPDF2

            return PyPDF2  # type: ignore[return-value]
        except ImportError:
            return None

    def _try_import_soundfile(self) -> _SoundFileModule | None:
        try:
            import soundfile as sf

            return sf  # type: ignore[return-value]
        except ImportError:
            return None

    @property
    def pil_image(self) -> _ImageModule | None:
        """PIL Image module if available."""
        return self._pil_image

    @property
    def magic(self) -> _MagicModule | None:
        """Python-magic module if available."""
        return self._magic

    @property
    def pypdf2(self) -> _PyPDF2Module | None:
        """PyPDF2 module if available."""
        return self._pypdf2

    @property
    def soundfile(self) -> _SoundFileModule | None:
        """Soundfile module if available."""
        return self._soundfile

    @property
    def has_pil(self) -> bool:
        """Check if PIL is available."""
        return self._pil_image is not None

    @property
    def has_magic(self) -> bool:
        """Check if python-magic is available."""
        return self._magic is not None

    @property
    def has_pypdf2(self) -> bool:
        """Check if PyPDF2 is available."""
        return self._pypdf2 is not None

    @property
    def has_soundfile(self) -> bool:
        """Check if soundfile is available."""
        return self._soundfile is not None


# Global instance for dependency checking
_deps: _OptionalDependencies = _OptionalDependencies()


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
        description="The binary content of the file or the Base64 encoded contents"
    )

    mime_type: str = Field(
        description="The MIME type of the file, must be a valid MIME type from Python's mimetypes module."
    )

    @override
    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._validate_mime_type()
        self._validate_data_integrity()
        self._validate_mime_type_matches_data()

    def _get_bytes_data(self) -> bytes:
        """Convert data to bytes regardless of input format."""
        if isinstance(self.data, bytes):
            return self.data

        # Try to decode as base64 first
        try:
            return b64.b64decode(self.data, validate=True)
        except Exception:
            # If not valid base64, treat as UTF-8 string
            return self.data.encode("utf-8")

    def _validate_mime_type(self) -> None:
        """Validates that the provided MIME type is official."""
        # Get all known MIME types from mimetypes module
        all_mimes: set[str] = set(mimetypes.types_map.values())
        # Also include common types that might not be in types_map
        all_mimes.update(mimetypes.common_types.values())

        if self.mime_type not in all_mimes:
            raise ValueError(
                f"The provided MIME type '{self.mime_type}' is not in the list of official MIME types."
            )

    def _validate_data_integrity(self) -> None:
        """Validates that the data is well-formed for its claimed MIME type."""
        try:
            data_bytes: bytes = self._get_bytes_data()
        except Exception as e:
            raise ValueError(f"Cannot decode file data: {e}") from e

        mime_category: str = self.mime_type.split("/")[0]

        if mime_category == "image":
            self._validate_image_data(data_bytes)
        elif mime_category == "audio":
            self._validate_audio_data(data_bytes)
        elif mime_category == "text":
            self._validate_text_data(data_bytes)
        elif self.mime_type == "application/pdf":
            self._validate_pdf_data(data_bytes)
        elif self.mime_type == "application/json":
            self._validate_json_data(data_bytes)
        elif self.mime_type in ["application/zip", "application/x-zip-compressed"]:
            self._validate_zip_data(data_bytes)
        elif self.mime_type in ["application/x-tar", "application/x-gtar"]:
            self._validate_tar_data(data_bytes)
        # Add more validators as needed

    def _validate_image_data(self, data_bytes: bytes) -> None:
        """Validate image data using PIL."""
        if not _deps.has_pil or _deps.pil_image is None:
            return  # Skip validation if PIL not available

        try:
            with _deps.pil_image.open(io.BytesIO(data_bytes)) as img:
                img.verify()  # Verify the image is valid
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}") from e

    def _validate_audio_data(self, data_bytes: bytes) -> None:
        """Validate audio data using soundfile."""
        if not _deps.has_soundfile or _deps.soundfile is None:
            return  # Skip validation if soundfile not available

        try:
            _deps.soundfile.info(io.BytesIO(data_bytes))
        except Exception as e:
            raise ValueError(f"Invalid audio data: {e}") from e

    def _validate_text_data(self, data_bytes: bytes) -> None:
        """Validate text data can be decoded."""
        try:
            data_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ["latin1", "cp1252", "ascii"]:
                try:
                    data_bytes.decode(encoding)
                    return
                except UnicodeDecodeError:
                    continue
            raise ValueError("Text data cannot be decoded with common encodings")

    def _validate_pdf_data(self, data_bytes: bytes) -> None:
        """Validate PDF data using PyPDF2."""
        if not _deps.has_pypdf2 or _deps.pypdf2 is None:
            return  # Skip validation if PyPDF2 not available

        try:
            _deps.pypdf2.PdfReader(io.BytesIO(data_bytes))
        except Exception as e:
            raise ValueError(f"Invalid PDF data: {e}") from e

    def _validate_json_data(self, data_bytes: bytes) -> None:
        """Validate JSON data."""
        try:
            json.loads(data_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}") from e

    def _validate_zip_data(self, data_bytes: bytes) -> None:
        """Validate ZIP archive data."""
        try:
            with zipfile.ZipFile(io.BytesIO(data_bytes), "r") as zf:
                bad_file: str | None = zf.testzip()
                if bad_file is not None:
                    raise ValueError(f"Corrupted file in ZIP: {bad_file}")
        except Exception as e:
            raise ValueError(f"Invalid ZIP data: {e}") from e

    def _validate_tar_data(self, data_bytes: bytes) -> None:
        """Validate TAR archive data."""
        try:
            with tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r") as tf:
                names: list[str] = tf.getnames()
                if not names:
                    raise ValueError("Empty TAR archive")
        except Exception as e:
            raise ValueError(f"Invalid TAR data: {e}") from e

    def _validate_mime_type_matches_data(self) -> None:
        """Validate that the declared MIME type matches the actual data content."""
        if not _deps.has_magic or _deps.magic is None:
            return  # Skip validation if python-magic not available

        try:
            data_bytes: bytes = self._get_bytes_data()
            detected_mime: str = _deps.magic.from_buffer(data_bytes, mime=True)

            # Some flexibility for common variations
            mime_variations: dict[str, list[str]] = {
                "image/jpeg": ["image/jpeg", "image/jpg"],
                "application/zip": ["application/zip", "application/x-zip-compressed"],
                "text/plain": ["text/plain", "text/x-python", "text/x-script.python"],
            }

            declared_mime: str = self.mime_type
            valid_mimes: list[str] = mime_variations.get(declared_mime, [declared_mime])

            if (
                detected_mime not in valid_mimes
                and declared_mime
                not in mime_variations.get(detected_mime, [declared_mime])
            ):
                raise ValueError(
                    f"MIME type mismatch: declared '{declared_mime}' but detected '{detected_mime}'"
                )

        except Exception:
            # Don't fail hard on detection errors, just pass silently
            pass

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
