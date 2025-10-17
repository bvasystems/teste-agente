"""
Microsoft Word Document Parser Module

This module provides functionality for parsing Microsoft Word documents (.doc, .docx) into
structured representations. It can extract text content, process embedded images, and
organize the document content.
"""

import logging
import os
import tempfile
import shutil
import subprocess
import hashlib
import uuid
import zipfile
from pathlib import Path
from typing import Literal, Any

from rsb.models.field import Field


from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)

from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.parsing.image import Image
from agentle.parsing.parsed_file import ParsedFile
from agentle.parsing.section_content import SectionContent
from agentle.parsing.document_parser import DocumentParser
from agentle.utils.file_validation import (
    FileValidationError,
    resolve_file_path,
    validate_file_exists,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------
class DocxParserError(Exception):
    """Base exception for DOCX parsing."""


class DocxFileValidationError(DocxParserError):
    """Raised when file validation fails."""


class DocxEmptyFileError(DocxParserError):
    """Raised when the DOCX file is empty."""


class DocxReadError(DocxParserError):
    """Raised on I/O read failures."""


class DocxProviderError(DocxParserError):
    """Raised on fatal provider (visual description) failures."""


class DocxFileParser(DocumentParser):
    """
    Parser for processing Microsoft Word documents (.doc, .docx).

    This parser extracts content from Word documents, including text and embedded images.
    With the "high" strategy, embedded images are analyzed using a visual description
    agent to extract text via OCR and generate descriptions. The parser represents the
    entire document as a single section containing all text and image content.

    **Attributes:**

    *   `strategy` (Literal["high", "low"]):
        The parsing strategy to use. Defaults to "high".
        - "high": Performs thorough parsing including OCR and image analysis
        - "low": Performs basic text extraction without analyzing images

        **Example:**
        ```python
        parser = DocxFileParser(strategy="low")  # Use faster, less intensive parsing
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe the image content. If provided and
        strategy is "high", this agent will be used to analyze images embedded
        in the document.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Focus on diagram and chart analysis in technical documents",
            response_schema=VisualMediaDescription
        )

        parser = DocxFileParser(visual_description_agent=custom_agent)
        ```

    *   `multi_modal_provider` (GenerationProvider):
        An alternative to using a visual_description_agent. This is a generation
        provider capable of handling multi-modal content (text and images).
        Defaults to GoogleGenerationProvider().

        Note: You cannot use both visual_description_agent and multi_modal_provider
        at the same time.

    **Usage Examples:**

    Basic parsing of a Word document:
    ```python
    from agentle.parsing.parsers.docx import DocxFileParser

    # Create a parser with default settings
    parser = DocxFileParser()

    # Parse a Word document
    parsed_doc = parser.parse("report.docx")

    # Access the text content
    print(parsed_doc.sections[0].text)

    # Access embedded images
    for image in parsed_doc.sections[0].images:
        print(f"Image: {image.name}")
        if image.ocr_text:
            print(f"  OCR text: {image.ocr_text}")
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse a Word document
    result = parse("document.docx")

    # Access the document content
    print(f"Document: {result.name}")
    print(f"Text content: {result.sections[0].text[:100]}...")
    print(f"Contains {len(result.sections[0].images)} images")
    ```
    """

    type: Literal["docx"] = "docx"

    strategy: Literal["high", "low"] = Field(default="high")

    visual_description_provider: GenerationProvider | None = Field(
        default=None,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    model: str | None = Field(default=None)
    """Model to use for visual description generation. If not provided, uses the provider's default model."""

    use_native_docx_processing: bool = Field(default=False)
    """Enable native DOCX processing by sending the entire DOCX to the AI provider.
    
    When enabled, the parser will send the complete DOCX file directly to the AI provider
    (if it supports native DOCX file processing) and request structured markdown extraction.
    This completely eliminates backend processing and can run efficiently on AWS instances.
    
    Requirements:
    - visual_description_provider must be set
    - The provider must support FilePart with mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    
    Benefits:
    - No python-docx dependency required
    - No LibreOffice/pandoc conversion needed
    - Faster processing as AI handles everything
    - Works well on small AWS instances
    
    Note: When this is enabled, most other configuration options are ignored as the AI handles all processing.
    """

    async def parse_async(
        self,
        document_path: str,
    ) -> ParsedFile:
        """
        Parse a Word document into a single Markdown section and describe visuals without duplicating page OCR.
        """
        # Check if native DOCX processing is enabled and we have a provider
        if self.use_native_docx_processing and self.visual_description_provider:
            return await self._parse_with_native_docx_processing(document_path)

        from docx import Document

        original_path = Path(document_path)
        working_path = original_path
        cleanup_converted: bool = False
        converted_temp_file: Path | None = None

        def _attempt_convert_to_docx(input_path: Path, reason: str) -> Path:
            """Try to convert the given file to a valid .docx using soffice or pandoc.

            Returns the path to a persistent temporary .docx on success, raises ValueError on failure.
            """
            with tempfile.TemporaryDirectory() as tmpdir:
                converted: Path | None = None
                soffice = shutil.which("soffice") or shutil.which("libreoffice")
                if soffice:
                    try:
                        subprocess.run(
                            [
                                soffice,
                                "--headless",
                                "--convert-to",
                                "docx",
                                "--outdir",
                                tmpdir,
                                str(input_path),
                            ],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=120,
                        )
                        candidate = Path(tmpdir) / (input_path.stem + ".docx")
                        if candidate.exists():
                            converted = candidate
                    except Exception as e:  # pragma: no cover - best effort
                        logger.debug(f"LibreOffice conversion to .docx failed: {e}")

                if converted is None:
                    pandoc = shutil.which("pandoc")
                    if pandoc:
                        try:
                            target = Path(tmpdir) / (input_path.stem + ".docx")
                            subprocess.run(
                                [pandoc, str(input_path), "-o", str(target)],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=120,
                            )
                            if target.exists():
                                converted = target
                        except Exception as e:  # pragma: no cover - best effort
                            logger.debug(f"pandoc conversion to .docx failed: {e}")

                if converted is None:
                    raise ValueError(
                        (
                            f"Failed to convert file '{input_path.name}' to a valid .docx while handling: {reason}. "
                            "Install LibreOffice (soffice) or pandoc, or ensure the file is a valid Word document."
                        )
                    )

                # Persist outside the context manager
                persistent_tmp = (
                    Path(tempfile.gettempdir())
                    / f"agentle_docx_{uuid.uuid4().hex}.docx"
                )
                shutil.copyfile(converted, persistent_tmp)
                return persistent_tmp

        # 1. Legacy .doc -> convert
        if original_path.suffix.lower() == ".doc":
            converted_temp_file = _attempt_convert_to_docx(
                original_path, "legacy .doc format"
            )
            working_path = converted_temp_file
            cleanup_converted = True

        # 2. File claims to be .docx but isn't a zip (common when a binary .doc is misnamed)
        elif original_path.suffix.lower() == ".docx" and not zipfile.is_zipfile(
            original_path
        ):
            logger.debug(
                "File '%s' has .docx extension but is not a valid OPC zip; attempting conversion.",
                original_path,
            )
            converted_temp_file = _attempt_convert_to_docx(
                original_path, "invalid/misnamed .docx (not a zip archive)"
            )
            working_path = converted_temp_file
            cleanup_converted = True

        try:
            try:
                document = Document(str(working_path))
            except Exception as e:
                # Provide a clearer error hint for misnamed or corrupt files
                is_zip = zipfile.is_zipfile(working_path)
                if not is_zip:
                    hint = (
                        "File is not a valid .docx (missing OPC structure). If this is an older binary .doc or another format, "
                        "install LibreOffice (soffice) or pandoc for automatic conversion."
                    )
                else:
                    hint = "File may be a damaged .docx archive. Try re-saving it in Word or exporting to a new .docx and retry."
                raise ValueError(
                    f"Failed to open Word document '{document_path}'. {hint} Original error: {e}"
                ) from e
            # Base Markdown via MarkItDown (best-effort)
            md_text: str | None = None
            try:
                try:
                    from markitdown import MarkItDown  # type: ignore

                    md_converter = MarkItDown(enable_plugins=False)
                    md_result = md_converter.convert(str(working_path))
                    if hasattr(md_result, "markdown") and md_result.markdown:
                        md_text = str(md_result.markdown)
                except ImportError:
                    md_text = None
                except Exception as e:
                    logger.warning(f"MarkItDown conversion failed for DOCX: {e}")
                    md_text = None
            except Exception:
                md_text = None

            if not md_text:
                # Fallback: join paragraphs with spacing
                paragraph_texts = [
                    p.text for p in document.paragraphs if p.text.strip()
                ]
                md_text = "\n\n".join(paragraph_texts)

            # Extract embedded images (kept as Image objects; OCR left empty to avoid duplication)
            # NOTE:
            # The original implementation iterated over document.part._rels (a private attribute)
            # and accessed rel.target_part directly. For relationships whose target mode is
            # External (e.g. linked images rather than embedded ones), python-docx does not
            # expose a target_part (or may raise on access), producing the error you observed:
            #   "target_part property on _Relationship is undefined when target mode is External".
            # We now:
            #   * Use the public .rels collection instead of the private ._rels
            #   * Skip external relationships (linked images) since they have no embedded bytes
            #   * Guard all attribute access and continue on failure, logging at debug level
            doc_images: list[tuple[str, bytes]] = []
            relationships = getattr(document.part, "rels", {})
            for rel in relationships.values():  # type: ignore[assignment]
                try:
                    reltype = getattr(rel, "reltype", "") or ""
                    if "image" not in reltype:
                        continue
                    # Skip external (linked) images; they are not embedded in the package
                    if getattr(rel, "is_external", False):
                        logger.debug(
                            "Skipping external image relationship (linked image): %s",
                            getattr(rel, "target_ref", None),
                        )
                        continue
                    image_part = getattr(rel, "target_part", None)
                    if image_part is None:
                        # Defensive: if python-docx changes behavior or part missing
                        logger.debug(
                            "Image relationship without target_part encountered; skipping: %s",
                            rel,
                        )
                        continue
                    image_name = Path(getattr(image_part, "partname", "image")).name
                    image_bytes = getattr(image_part, "blob", None)
                    if not image_bytes:
                        logger.debug(
                            "Image part missing blob data; skipping image %s",
                            image_name,
                        )
                        continue
                    # Ensure bytes type (python-docx already gives bytes but be safe)
                    if not isinstance(image_bytes, (bytes, bytearray)):
                        logger.debug(
                            "Image part blob not bytes (type=%s); skipping image %s",
                            type(image_bytes).__name__,
                            image_name,
                        )
                        continue
                    doc_images.append((image_name, bytes(image_bytes)))
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to extract image from relationship %s: %s", rel, e
                    )

            final_images: list[Image] = [
                Image(name=name, contents=bytes(data), ocr_text="")
                for name, data in doc_images
            ]

            image_descriptions: list[str] = []
            image_cache: dict[str, tuple[str, str]] = {}

            if (
                self.visual_description_provider
                and self.strategy == "high"
                and doc_images
            ):
                # Only support the page-screenshot path. If conversion or rendering fails, raise a clear error.
                try:
                    try:
                        import fitz as pymupdf_module  # type: ignore
                    except Exception as e:
                        raise ValueError(
                            "Page screenshot analysis requires PyMuPDF (import fitz). Please install 'pymupdf'."
                        ) from e

                    def _try_convert_docx_to_pdf_headless(
                        input_path: str, out_dir: str
                    ) -> str | None:
                        pdf_out = os.path.join(out_dir, f"{Path(input_path).stem}.pdf")
                        soffice = shutil.which("soffice") or shutil.which("libreoffice")
                        if soffice:
                            try:
                                subprocess.run(
                                    [
                                        soffice,
                                        "--headless",
                                        "--convert-to",
                                        "pdf",
                                        "--outdir",
                                        out_dir,
                                        input_path,
                                    ],
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    timeout=120,
                                )
                                if os.path.exists(pdf_out):
                                    return pdf_out
                            except Exception as e:
                                logger.warning(
                                    f"LibreOffice (soffice) conversion failed: {e}"
                                )

                        pandoc = shutil.which("pandoc")
                        if pandoc:
                            try:
                                subprocess.run(
                                    [pandoc, input_path, "-o", pdf_out],
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    timeout=120,
                                )
                                if os.path.exists(pdf_out):
                                    return pdf_out
                            except Exception as e:
                                logger.warning(f"pandoc conversion failed: {e}")

                        return None

                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_path = _try_convert_docx_to_pdf_headless(
                            document_path, temp_dir
                        )
                        if not pdf_path:
                            raise ValueError(
                                "DOCX->PDF conversion failed or is unavailable. Install either 'libreoffice' (soffice) or 'pandoc' to enable page screenshot analysis."
                            )

                        try:
                            mu_doc = pymupdf_module.open(pdf_path)  # type: ignore
                        except Exception as e:
                            raise ValueError(
                                f"Failed to open converted PDF with PyMuPDF. Ensure the document is valid. Details: {e}"
                            ) from e

                        try:
                            for page_idx in range(mu_doc.page_count):  # type: ignore[attr-defined]
                                page_obj = mu_doc[page_idx]  # type: ignore[index]

                                # Heuristically check for images
                                get_images = getattr(
                                    page_obj, "get_images", None
                                ) or getattr(page_obj, "getImages", None)
                                page_has_images = False
                                if callable(get_images):
                                    try:
                                        img_list = get_images(full=True)  # type: ignore[call-arg]
                                        page_has_images = bool(img_list)
                                    except Exception:
                                        page_has_images = True
                                if not page_has_images:
                                    continue

                                # Render page screenshot at 2x
                                matrix = getattr(pymupdf_module, "Matrix")(2.0, 2.0)  # type: ignore
                                get_pixmap = getattr(
                                    page_obj, "get_pixmap", None
                                ) or getattr(page_obj, "getPixmap", None)
                                if not callable(get_pixmap):
                                    continue
                                raw_pix = get_pixmap(matrix=matrix)  # type: ignore[call-arg]
                                pix: Any = raw_pix  # Help static analyzer; PyMuPDF provides tobytes
                                if not hasattr(pix, "tobytes"):
                                    raise ValueError(
                                        "Pixmap object from PyMuPDF does not expose 'tobytes'. Check PyMuPDF version compatibility."
                                    )
                                # PyMuPDF Pixmap.tobytes("png") returns PNG bytes
                                page_image_bytes: bytes = pix.tobytes("png")  # type: ignore[call-arg]

                                page_hash = hashlib.sha256(page_image_bytes).hexdigest()
                                if page_hash in image_cache:
                                    page_description = image_cache[page_hash][0]
                                else:
                                    agent_input = FilePart(
                                        mime_type="image/png", data=page_image_bytes
                                    )
                                    agent_response = await self.visual_description_provider.generate_by_prompt_async(
                                        agent_input,
                                        developer_prompt=(
                                            "You are a highly precise visual analyst. You are given a screenshot of a Word document page. "
                                            "Only identify and describe the images/graphics/figures present on this page. "
                                            "Do NOT transcribe or repeat the page's regular text content. "
                                            "If an image contains important embedded text (e.g., labels in a chart), summarize it succinctly as part of the image description. "
                                            "Output clear, concise descriptions suitable for a 'Visual Content' section."
                                        ),
                                        response_schema=VisualMediaDescription,
                                    )
                                    page_description = agent_response.parsed.md
                                    image_cache[page_hash] = (page_description, "")

                                image_descriptions.append(
                                    f"Page Visual Content: {page_description}"
                                )
                        finally:
                            try:
                                mu_doc.close()  # type: ignore[attr-defined]
                            except Exception:
                                pass
                except Exception:
                    # Raise a clear error instead of falling back to per-image processing
                    raise

            if image_descriptions:
                md_text += "\n" + "\n".join(
                    [
                        "\n\n## Visual Content",
                        *[f"- {desc}" for desc in image_descriptions],
                    ]
                )

            return ParsedFile(
                name=document_path,
                sections=[
                    SectionContent(
                        number=1, text=md_text, md=md_text, images=final_images
                    )
                ],
            )
        finally:
            if (
                cleanup_converted
                and converted_temp_file
                and converted_temp_file.exists()
            ):
                try:
                    converted_temp_file.unlink()
                except Exception:
                    pass

    async def _parse_with_native_docx_processing(
        self, document_path: str
    ) -> ParsedFile:
        """Parse DOCX using native AI provider processing via PDF conversion.

        This method converts the DOCX to PDF and then delegates to PDFFileParser
        with native processing enabled. This leverages the PDF parser's native
        AI processing capabilities.

        Args:
            document_path: Path to the DOCX file

        Returns:
            ParsedFile: Structured representation with AI-extracted content

        Raises:
            DocxFileValidationError: If file validation fails
            DocxProviderError: If conversion or processing fails
        """
        # Validate path
        try:
            resolved_path = resolve_file_path(document_path)
            validate_file_exists(resolved_path)
        except FileValidationError as e:
            logger.error("DOCX file validation failed: %s", e)
            raise DocxFileValidationError(str(e)) from e

        display_path = Path(resolved_path).name
        logger.debug(
            "Parsing DOCX with native AI processing (via PDF conversion): %s",
            display_path,
        )

        # Convert DOCX to PDF
        def _convert_docx_to_pdf(input_path: str, out_dir: str) -> str | None:
            """Convert DOCX to PDF using LibreOffice or pandoc."""
            pdf_out = os.path.join(out_dir, f"{Path(input_path).stem}.pdf")

            # Try LibreOffice first
            soffice = shutil.which("soffice") or shutil.which("libreoffice")
            if soffice:
                try:
                    subprocess.run(
                        [
                            soffice,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            out_dir,
                            input_path,
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120,
                    )
                    if os.path.exists(pdf_out):
                        logger.debug(
                            "Successfully converted DOCX to PDF using LibreOffice"
                        )
                        return pdf_out
                except Exception as e:
                    logger.warning(f"LibreOffice (soffice) conversion failed: {e}")

            # Try pandoc as fallback
            pandoc = shutil.which("pandoc")
            if pandoc:
                try:
                    subprocess.run(
                        [pandoc, input_path, "-o", pdf_out],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120,
                    )
                    if os.path.exists(pdf_out):
                        logger.debug("Successfully converted DOCX to PDF using pandoc")
                        return pdf_out
                except Exception as e:
                    logger.warning(f"pandoc conversion failed: {e}")

            return None

        # Create temporary directory for PDF conversion
        temp_dir = tempfile.mkdtemp()
        try:
            pdf_path = _convert_docx_to_pdf(resolved_path, temp_dir)
            if not pdf_path:
                raise DocxProviderError(
                    "DOCX->PDF conversion failed. Install either 'libreoffice' (soffice) or 'pandoc' "
                    + "to enable native DOCX processing."
                )

            # Import PDFFileParser here to avoid circular imports
            from agentle.parsing.parsers.pdf import PDFFileParser

            # Create PDF parser with native processing enabled
            pdf_parser = PDFFileParser(
                visual_description_provider=self.visual_description_provider,
                model=self.model,
                use_native_pdf_processing=True,
                strategy=self.strategy,
            )

            logger.debug("Delegating to PDFFileParser with native processing")

            # Parse the PDF using native processing
            parsed = await pdf_parser.parse_async(pdf_path)

            # Update metadata to reflect DOCX origin
            if hasattr(parsed, "metadata") and isinstance(parsed.metadata, dict):
                parsed.metadata["original_format"] = "docx"
                parsed.metadata["parser"] = "docx"
                parsed.metadata["conversion_method"] = "docx_to_pdf"

            # Update the name to reflect original DOCX file
            parsed.name = Path(resolved_path).name

            return parsed

        except DocxProviderError:
            raise
        except Exception as e:
            logger.error(
                "Native DOCX processing failed: %s. Model used: %s, Provider: %s",
                e,
                self.model or "default",
                type(self.visual_description_provider).__name__
                if self.visual_description_provider
                else "None",
            )
            raise DocxProviderError(
                f"Failed to process DOCX with AI provider: {e}. "
                + f"Model: {self.model or 'default'}, "
                + f"Provider: {type(self.visual_description_provider).__name__ if self.visual_description_provider else 'None'}"
            ) from e
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.debug(f"Failed to clean up temporary directory {temp_dir}: {e}")
