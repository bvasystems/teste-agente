"""PDF Document Parser Module

Enhanced PDF parsing with:
- Structured exception hierarchy
- Configurable parsing & visual analysis strategies
- Page screenshot vs per-image modes with auto selection
- Concurrency, retries & timeouts for provider calls
- Duplicate image & page screenshot caching
- Optional whole-document markdown (disabled by default)
- Encryption handling with password support
- Whitespace normalization & optional skipping of empty pages
- Progress callbacks & rich metrics collection
- Image size / count limiting to protect resources
- Robust MIME detection for images without extensions
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from collections.abc import MutableSequence
from pathlib import Path
from typing import Any, Callable, Literal, cast, Awaitable

from rsb.functions.ext2mime import ext2mime
from rsb.models.field import Field

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.image import Image
from agentle.parsing.parsed_file import ParsedFile
from agentle.parsing.section_content import SectionContent
from agentle.utils.file_validation import (
    FileValidationError,
    resolve_file_path,
    validate_file_exists,
)

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------
class PDFParserError(Exception):
    """Base exception for PDF parsing."""


class PDFDependencyMissingError(PDFParserError):
    """Raised when a required dependency is missing."""


class PDFFileValidationError(PDFParserError):
    """Raised when file validation fails."""


class PDFEmptyFileError(PDFParserError):
    """Raised when the PDF file is empty."""


class PDFEncryptedError(PDFParserError):
    """Raised when the PDF is encrypted and cannot be decrypted."""


class PDFReadError(PDFParserError):
    """Raised on I/O read failures."""


class PDFCorruptError(PDFParserError):
    """Raised when PDF appears corrupted or invalid."""


class PDFProviderError(PDFParserError):
    """Raised on fatal provider (visual description) failures."""


@dataclass
class PageMetrics:
    page_number: int
    text_chars: int
    images_found: int
    images_described: int
    duration_s: float
    mode: str


@dataclass
class PDFParseMetrics:
    total_pages: int
    start_time_s: float
    end_time_s: float = 0.0
    pages: list[PageMetrics] | None = None
    provider_calls: int = 0
    provider_success: int = 0
    provider_failures: int = 0

    @property
    def total_duration_s(self) -> float:
        return self.end_time_s - self.start_time_s if self.end_time_s else 0.0


class PDFFileParser(DocumentParser):
    """
    Parser for processing PDF documents into structured representations.

    This parser extracts content from PDF files, including text and embedded images.
    Each page in the PDF is represented as a separate section in the resulting ParsedFile.
    With the "high" strategy, embedded images are analyzed using a visual description agent
    to extract text via OCR and generate descriptions.

    **Attributes:**

    *   `strategy` (Literal["high", "low"]):
        The parsing strategy to use. Defaults to "high".
        - "high": Performs thorough parsing including OCR and image analysis
        - "low": Performs basic text extraction without analyzing images

        **Example:**
        ```python
        parser = PDFFileParser(strategy="low")  # Use faster, less intensive parsing
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        An optional custom agent for visual media description. If provided and strategy
        is "high", this agent will be used to analyze images embedded in the PDF.
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

        parser = PDFFileParser(visual_description_agent=custom_agent)
        ```

    **Usage Examples:**

    Basic parsing of a PDF file:
    ```python
    from agentle.parsing.parsers.pdf import PDFFileParser

    # Create a parser with default settings
    parser = PDFFileParser()

    # Parse a PDF file
    parsed_doc = parser.parse("document.pdf")

    # Access the pages as sections
    for i, section in enumerate(parsed_doc.sections):
        print(f"Page {i+1} content:")
        print(section.text[:100] + "...")  # Print first 100 chars of each page
    ```

    Processing a PDF with focus on image analysis:
    ```python
    from agentle.parsing.parsers.pdf import PDFFileParser

    # Create a parser with high-detail strategy
    parser = PDFFileParser(strategy="high")

    # Parse a PDF with images
    report = parser.parse("annual_report.pdf")

    # Extract and process images
    for i, section in enumerate(report.sections):
        page_num = i + 1
        print(f"Page {page_num} has {len(section.images)} images")

        for j, image in enumerate(section.images):
            print(f"  Image {j+1}:")
            if image.ocr_text:
                print(f"    OCR text: {image.ocr_text}")
    ```
    """

    type: Literal["pdf"] = "pdf"
    strategy: Literal["high", "low"] = Field(default="high")
    visual_description_provider: GenerationProvider | None = Field(default=None)
    # Feature toggles & configuration
    include_visuals: bool = Field(default=True)
    image_processing_mode: Literal["auto", "page_screenshot", "per_image"] = Field(
        default="auto"
    )
    always_capture_page_visuals: bool = Field(default=False)
    enable_whole_document_markdown: bool = Field(default=False)
    normalize_whitespace: bool = Field(default=True)
    skip_empty_pages: bool = Field(default=False)
    continue_on_provider_error: bool = Field(default=True)
    image_description_retries: int = Field(default=2)
    image_description_timeout: float = Field(default=25.0)
    max_concurrent_provider_tasks: int = Field(default=4)
    max_images_per_page: int | None = Field(default=None)
    max_total_images: int | None = Field(default=None)
    max_image_bytes: int | None = Field(default=None)
    render_scale: float = Field(default=2.0)
    password: str | None = Field(default=None)
    log_file_names_only: bool = Field(default=False)
    collect_metrics: bool = Field(default=True)
    progress_callback: Callable[[int, int], None] | None = Field(default=None)
    use_temp_copy: bool = Field(default=False)

    # Metrics state
    last_parse_metrics: PDFParseMetrics | None = None

    async def parse_async(self, document_path: str) -> ParsedFile:
        """
        Asynchronously parse a PDF document and convert it to a structured representation.

        This method reads a PDF file, extracts text content from each page, and processes
        any embedded images. With the "high" strategy, images are analyzed using the
        visual description agent to extract text and generate descriptions.

        Args:
            document_path (str): Path to the PDF file to be parsed

        Returns:
            ParsedFile: A structured representation of the PDF where:
                - Each PDF page is a separate section
                - Text content is extracted from each page
                - Images are extracted and (optionally) analyzed

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.pdf import PDFFileParser

            async def process_pdf():
                parser = PDFFileParser(strategy="high")
                result = await parser.parse_async("whitepaper.pdf")

                # Get the total number of pages
                print(f"Document has {len(result.sections)} pages")

                # Extract text from the first page
                if result.sections:
                    first_page = result.sections[0]
                    print(f"First page text: {first_page.text[:200]}...")

                    # Count images on the first page
                    print(f"First page has {len(first_page.images)} images")

            asyncio.run(process_pdf())
            ```
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover - dependency absent env
            logger.error("pypdf library not available for PDF parsing")
            raise PDFDependencyMissingError(
                "Missing dependency 'pypdf'. Install with: pip install pypdf"
            ) from e

        # Validate path
        try:
            resolved_path = resolve_file_path(document_path)
            validate_file_exists(resolved_path)
        except FileValidationError as e:
            logger.error("PDF file validation failed: %s", e)
            raise PDFFileValidationError(str(e)) from e

        display_path = (
            Path(resolved_path).name if self.log_file_names_only else resolved_path
        )
        logger.debug("Parsing PDF: %s", display_path)

        pdf_path_to_use = resolved_path
        temp_dir: str | None = None
        if self.use_temp_copy:
            try:
                with open(resolved_path, "rb") as src:
                    pdf_bytes = src.read()
                if not pdf_bytes:
                    raise PDFEmptyFileError(f"PDF file '{document_path}' is empty")
            except PermissionError as e:
                raise PDFReadError(
                    f"Permission denied reading '{document_path}'"
                ) from e
            except OSError as e:
                raise PDFReadError(f"OS error reading '{document_path}': {e}") from e
            temp_dir = tempfile.mkdtemp()
            pdf_path_to_use = os.path.join(temp_dir, Path(resolved_path).name)
            try:
                with open(pdf_path_to_use, "wb") as f:
                    f.write(pdf_bytes)
            except OSError as e:
                raise PDFReadError(f"Failed writing temp copy: {e}") from e

        # Load PDF reader
        try:
            reader = PdfReader(pdf_path_to_use)
        except Exception as e:
            raise PDFCorruptError(f"Failed to open PDF '{document_path}': {e}") from e

        # Handle encryption
        try:
            if getattr(reader, "is_encrypted", False):
                if self.password:
                    try:
                        decrypt_result = reader.decrypt(self.password)
                        if decrypt_result == 0:  # pypdf returns 0/False if failed
                            raise PDFEncryptedError(
                                "Incorrect password for encrypted PDF"
                            )
                    except Exception as e:
                        raise PDFEncryptedError(f"Failed to decrypt PDF: {e}") from e
                else:
                    raise PDFEncryptedError(
                        "PDF is encrypted. Provide 'password' parameter to decrypt."
                    )
        except AttributeError:
            pass  # Older pypdf versions

        total_pages = len(reader.pages)
        if total_pages == 0:
            logger.warning("PDF has zero pages: %s", display_path)

        metrics = PDFParseMetrics(
            total_pages=total_pages,
            start_time_s=time.time(),
            pages=[] if self.collect_metrics else None,
        )

        # Optional PyMuPDF open (best effort)
        pymupdf_module = None
        mu_doc = None
        if self.include_visuals and self.strategy == "high":
            try:  # pragma: no cover - optional dependency
                import fitz as pymupdf_module  # type: ignore

                mu_doc = pymupdf_module.open(pdf_path_to_use)  # type: ignore
            except ImportError:
                pymupdf_module = None
                mu_doc = None
            except Exception as e:
                logger.debug(
                    "PyMuPDF open failed (%s) - continuing without page screenshots", e
                )
                pymupdf_module = None
                mu_doc = None

        # Whole document markdown (optional heavy)
        if self.enable_whole_document_markdown:
            try:  # pragma: no cover - optional dependency
                from markitdown import MarkItDown  # type: ignore

                md_converter = MarkItDown(enable_plugins=False)
                _ = md_converter.convert(
                    pdf_path_to_use
                )  # Currently not integrated per-page
            except Exception as e:
                logger.debug("Whole-document markdown extraction skipped (%s)", e)

        # Global image tracking
        image_cache: dict[str, tuple[str, str]] = {}
        total_images_seen = 0
        sections: MutableSequence[SectionContent] = []

        def progress(i: int):
            if self.progress_callback:
                try:
                    self.progress_callback(i, total_pages)
                except Exception:
                    logger.debug("Progress callback raised", exc_info=True)

        # Provider convenience variable
        provider = (
            self.visual_description_provider
            if (self.strategy == "high" and self.include_visuals)
            else None
        )
        semaphore = asyncio.Semaphore(self.max_concurrent_provider_tasks)

        for page_index, page in enumerate(reader.pages):
            progress(page_index + 1)
            start_page = time.time()
            page_images: MutableSequence[Image] = []
            image_descriptions: MutableSequence[str] = []
            mode_used = "none"

            # Extract images metadata (pypdf provides .images list) - may be empty
            page_raw_images = getattr(page, "images", [])
            # Iterate with limits
            for img_idx, img in enumerate(page_raw_images):
                if self.max_images_per_page and img_idx >= self.max_images_per_page:
                    logger.info(
                        "Skipping remaining images on page %d due to max_images_per_page=%d",
                        page_index + 1,
                        self.max_images_per_page,
                    )
                    break
                if self.max_total_images and total_images_seen >= self.max_total_images:
                    logger.info(
                        "Global image limit reached (%d); skipping further images",
                        self.max_total_images,
                    )
                    break
                img_bytes = img.data
                if self.max_image_bytes and len(img_bytes) > self.max_image_bytes:
                    logger.debug(
                        "Skipping oversized image (%d bytes > %d)",
                        len(img_bytes),
                        self.max_image_bytes,
                    )
                    continue
                short_hash = hashlib.sha256(img_bytes).hexdigest()[:8]
                unique_name = f"page_{page_index + 1}_img_{img_idx + 1}_{short_hash}"
                page_images.append(
                    Image(
                        contents=img_bytes,
                        name=unique_name,
                        ocr_text=None,  # Not performing OCR here
                    )
                )
                total_images_seen += 1

            # Decide processing mode
            selected_mode = self.image_processing_mode
            if selected_mode == "auto":
                if provider and (page_images or self.always_capture_page_visuals):
                    selected_mode = "page_screenshot"
                else:
                    selected_mode = "none"

            # Page screenshot mode
            if (
                selected_mode == "page_screenshot"
                and provider
                and (page_images or self.always_capture_page_visuals)
                and mu_doc is not None
                and pymupdf_module is not None
            ):
                try:
                    mode_used = "page_screenshot"
                    mu_page = cast(Any, mu_doc[page_index])  # type: ignore
                    matrix = pymupdf_module.Matrix(self.render_scale, self.render_scale)  # type: ignore
                    get_pixmap = getattr(mu_page, "get_pixmap", None) or getattr(
                        mu_page, "getPixmap", None
                    )
                    if not callable(get_pixmap):  # pragma: no cover
                        raise AttributeError("PyMuPDF page lacks get_pixmap/getPixmap")
                    pix = get_pixmap(matrix=matrix)  # type: ignore[call-arg]
                    png_bytes = pix.tobytes("png")  # type: ignore[attr-defined]
                    assert isinstance(png_bytes, (bytes, bytearray))
                    if isinstance(png_bytes, bytearray):  # normalize
                        png_bytes = bytes(png_bytes)
                    page_hash = hashlib.sha256(png_bytes).hexdigest()
                    if page_hash in image_cache:
                        cached_md, _ = image_cache[page_hash]
                        image_descriptions.append(f"Page Visual Content: {cached_md}")
                    else:
                        provider_result = await self._call_provider_with_retry(
                            provider,
                            FilePart(mime_type="image/png", data=png_bytes),
                            semaphore,
                            developer_prompt=(
                                "You are a highly precise visual analyst. You are given a screenshot of a PDF page. "
                                "Only identify and describe the images/graphics/figures present on this page. "
                                "Do NOT transcribe or repeat the page's regular text content. "
                                "If an image contains important embedded text (e.g., labels in a chart), summarize it succinctly. "
                                "Output concise bullet descriptions under a 'Visual Content' interpretation."
                            ),
                            metrics=metrics,
                        )
                        if isinstance(provider_result, str) and provider_result:
                            image_cache[page_hash] = (provider_result, "")
                            image_descriptions.append(
                                f"Page Visual Content: {provider_result}"
                            )
                except Exception as e:
                    logger.warning(
                        "Page screenshot processing failed on page %d (%s) -> falling back to per_image if enabled",
                        page_index + 1,
                        e,
                    )
                    if self.image_processing_mode == "auto":
                        selected_mode = "per_image"
                    else:
                        selected_mode = "none"

            # Per-image mode
            if selected_mode == "per_image" and provider and page_images:
                mode_used = "per_image"
                tasks: list[Awaitable[tuple[int, str | None]]] = []
                for img_idx, img_obj in enumerate(page_images, start=1):
                    img_hash_full = hashlib.sha256(img_obj.contents).hexdigest()
                    if img_hash_full in image_cache:
                        cached_md, _ = image_cache[img_hash_full]
                        image_descriptions.append(
                            f"Page {page_index + 1} - Image {img_idx}: {cached_md}"
                        )
                        continue

                    mime = self._deduce_mime_type(img_obj.name or "image")
                    fp = FilePart(mime_type=mime, data=img_obj.contents)
                    task = self._call_provider_with_retry(
                        provider,
                        fp,
                        semaphore,
                        developer_prompt=(
                            "You are a precise visual analyst. Describe the content of the image/figure succinctly. "
                            "Do not repeat surrounding page text. Include embedded text only if critical."
                        ),
                        metrics=metrics,
                        return_with_index=img_idx,
                    )
                    tasks.append(task)  # type: ignore[arg-type]

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=False)
                    for img_idx, md in results:  # type: ignore[misc]
                        if md:  # type: ignore[truthy-bool]
                            img_hash_full = hashlib.sha256(
                                page_images[img_idx - 1].contents
                            ).hexdigest()
                            image_cache[img_hash_full] = (md, "")
                            image_descriptions.append(
                                f"Page {page_index + 1} - Image {img_idx}: {md}"
                            )
                        else:
                            image_descriptions.append(
                                f"Page {page_index + 1} - Image {img_idx}: (Description unavailable)"
                            )

            # Extract text (pypdf)
            raw_text = page.extract_text() or ""
            if self.normalize_whitespace:
                raw_text = self._normalize_whitespace(raw_text)

            if (
                self.skip_empty_pages
                and not raw_text.strip()
                and not image_descriptions
            ):
                logger.debug("Skipping empty page %d (no text/images)", page_index + 1)
                continue

            # Build markdown
            header = f"## Page {page_index + 1}"
            text_block = raw_text.strip()
            visual_block = (
                "### Visual Content\n" + "\n".join(f"- {d}" for d in image_descriptions)
                if image_descriptions
                else ""
            )
            md_parts = [header]
            if text_block:
                md_parts.append(text_block)
            if visual_block:
                md_parts.append(visual_block)
            md_page = "\n\n".join(part for part in md_parts if part)

            section = SectionContent(
                number=page_index + 1,
                text=md_page,
                md=md_page,
                images=page_images,
            )
            sections.append(section)

            # Metrics per page
            if metrics.pages is not None:
                metrics.pages.append(
                    PageMetrics(
                        page_number=page_index + 1,
                        text_chars=len(raw_text),
                        images_found=len(page_images),
                        images_described=sum(1 for _ in image_descriptions),
                        duration_s=time.time() - start_page,
                        mode=mode_used,
                    )
                )

        # Close PyMuPDF doc
        if mu_doc is not None:
            try:  # pragma: no cover
                mu_doc.close()  # type: ignore
            except Exception:
                pass

        metrics.end_time_s = time.time()
        self.last_parse_metrics = metrics if self.collect_metrics else None

        parsed = ParsedFile(name=Path(resolved_path).name, sections=sections)
        if hasattr(parsed, "metadata") and isinstance(
            parsed.metadata, dict
        ):  # runtime guard
            parsed.metadata.update(
                {
                    "parser": "pdf",
                    "pages": total_pages,
                    "strategy": self.strategy,
                    "visuals": self.include_visuals,
                    "processing_mode": self.image_processing_mode,
                }
            )
        return parsed

    # ------------------------------------------------------------------
    # Provider Call Helpers & Utilities
    # ------------------------------------------------------------------
    async def _call_provider_with_retry(
        self,
        provider: GenerationProvider,
        file_part: FilePart,
        semaphore: asyncio.Semaphore,
        developer_prompt: str,
        metrics: PDFParseMetrics,
        return_with_index: int | None = None,
    ) -> str | tuple[int, str | None] | None:
        """Call provider with retry, timeout & optional index return.

        If return_with_index is provided, returns (index, md | None).
        """
        last_error: Exception | None = None
        for attempt in range(self.image_description_retries + 1):
            try:
                async with semaphore:
                    metrics.provider_calls += 1
                    response = await asyncio.wait_for(
                        provider.generate_by_prompt_async(
                            file_part,
                            developer_prompt=developer_prompt,
                            response_schema=VisualMediaDescription,
                        ),
                        timeout=self.image_description_timeout,
                    )
                md = response.parsed.md
                metrics.provider_success += 1
                if return_with_index is not None:
                    return (return_with_index, md)
                return md
            except Exception as e:  # pragma: no cover - diverse providers
                last_error = e
                metrics.provider_failures += 1
                await asyncio.sleep(0.2 * (attempt + 1))
        # Failed all attempts
        if not self.continue_on_provider_error:
            raise PDFProviderError(f"Provider failure: {last_error}") from last_error
        logger.warning("Provider failed after retries: %s", last_error)
        if return_with_index is not None:
            return (return_with_index, None)
        return None

    def _normalize_whitespace(self, text: str) -> str:
        lines = [ln.rstrip() for ln in text.splitlines()]
        collapsed: list[str] = []
        blank = False
        for ln in lines:
            if ln.strip():
                collapsed.append(ln)
                blank = False
            else:
                if not blank:
                    collapsed.append("")
                blank = True
        return "\n".join(collapsed).strip()

    def _deduce_mime_type(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if not suffix:
            return "application/octet-stream"
        try:
            return ext2mime(suffix)
        except Exception:  # pragma: no cover
            return "application/octet-stream"
