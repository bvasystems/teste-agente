"""
PDF Document Parser Module

This module provides functionality for parsing PDF documents into structured representations.
It can extract text content, process embedded images, and organize the document by pages.
"""

import hashlib
import logging
import os
import tempfile
from collections.abc import MutableSequence
from pathlib import Path
from typing import Any, Literal

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
    """
    The provider to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

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
        except ImportError as e:
            logger.error("pypdf library not available for PDF parsing")
            raise ValueError(
                "PDF parsing requires the 'pypdf' library. Please install it with: pip install pypdf"
            ) from e

        try:
            # Validate and resolve the file path
            resolved_path = resolve_file_path(document_path)
            validate_file_exists(resolved_path)

            logger.debug(f"Reading PDF file: {resolved_path}")

            # Read file bytes with error handling
            try:
                _bytes = Path(resolved_path).read_bytes()
            except PermissionError as e:
                logger.error(f"Permission denied reading PDF file: {resolved_path}")
                raise ValueError(
                    f"Permission denied: Cannot read PDF file '{document_path}'. Please check file permissions."
                ) from e
            except OSError as e:
                logger.error(f"OS error reading PDF file: {resolved_path} - {e}")
                raise ValueError(
                    f"Failed to read PDF file '{document_path}': {e}"
                ) from e

            if not _bytes:
                logger.warning(f"PDF file appears to be empty: {resolved_path}")
                raise ValueError(f"PDF file '{document_path}' is empty")

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, Path(resolved_path).name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(_bytes)
                except OSError as e:
                    logger.error(f"Failed to write temporary PDF file: {e}")
                    raise ValueError(
                        f"Failed to process PDF file '{document_path}': {e}"
                    ) from e

                try:
                    reader = PdfReader(file_path)
                except Exception as e:
                    logger.error(f"Failed to parse PDF file: {resolved_path} - {e}")
                    raise ValueError(
                        f"Invalid or corrupted PDF file '{document_path}': {e}"
                    ) from e

                if len(reader.pages) == 0:
                    logger.warning(f"PDF file has no pages: {resolved_path}")

                section_contents: MutableSequence[SectionContent] = []
                image_cache: dict[str, tuple[str, str]] = {}

            for page_num, page in enumerate(reader.pages):
                page_images: MutableSequence[Image] = []
                image_descriptions: MutableSequence[str] = []

                # Extract individual images for the Image objects
                for image in page.images:
                    page_images.append(
                        Image(
                            contents=image.data,
                            name=image.name,
                            ocr_text="",  # Will be filled by page screenshot analysis
                        )
                    )

                # If there are images and we have a visual description provider, 
                # capture a screenshot of the entire page instead of processing each image individually
                if page_images and self.visual_description_provider and self.strategy == "high":
                    try:
                        import pymupdf
                        
                        # Create a temporary PDF with just this page for screenshot
                        temp_pdf_path = os.path.join(temp_dir, f"page_{page_num}.pdf")
                        temp_doc = pymupdf.open()
                        temp_doc.insert_pdf(pymupdf.open(file_path), from_page=page_num, to_page=page_num)
                        temp_doc.save(temp_pdf_path)
                        temp_doc.close()
                        
                        # Render the page as an image
                        doc = pymupdf.open(temp_pdf_path)  # type: ignore
                        page_obj = doc[0]  # type: ignore
                        pix = page_obj.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))  # type: ignore # 2x scale for better quality
                        page_image_bytes: bytes = pix.tobytes("png")  # type: ignore
                        doc.close()  # type: ignore
                        
                        # Generate hash for caching
                        page_hash = hashlib.sha256(page_image_bytes).hexdigest()  # type: ignore
                        
                        if page_hash in image_cache:
                            cached_md, cached_ocr = image_cache[page_hash]
                            page_description = cached_md
                            page_ocr_text = cached_ocr
                        else:
                            # Send the page screenshot to the visual description agent
                            agent_input = FilePart(
                                mime_type="image/png",
                                data=page_image_bytes,  # type: ignore
                            )

                            agent_response = await self.visual_description_provider.generate_by_prompt_async(
                                agent_input,
                                developer_prompt="You are a helpful assistant that deeply understands visual media. Analyze this PDF page screenshot and extract all text content and describe any visual elements like images, charts, diagrams, etc.",
                                response_schema=VisualMediaDescription,
                            )

                            page_description = agent_response.parsed.md
                            page_ocr_text = agent_response.parsed.ocr_text or ""
                            image_cache[page_hash] = (page_description, page_ocr_text)
                        
                        # Update all images on this page with the OCR text from the page analysis
                        for image in page_images:
                            image.ocr_text = page_ocr_text
                        
                        # Add the page description
                        image_descriptions.append(f"Page Visual Content: {page_description}")
                        
                    except ImportError:
                        logger.warning("PyMuPDF not available for page screenshot optimization. Falling back to individual image processing.")
                        # Fallback to original individual image processing
                        await self._process_images_individually(page, page_images, image_descriptions, image_cache)
                    except Exception as e:
                        logger.warning(f"Failed to capture page screenshot: {e}. Falling back to individual image processing.")
                        # Fallback to original individual image processing
                        await self._process_images_individually(page, page_images, image_descriptions, image_cache)

                page_text = [page.extract_text(), "".join(image_descriptions)]
                md = "".join(page_text)
                section_content = SectionContent(
                    number=page_num + 1,
                    text=md,
                    md=md,
                    images=page_images,
                )
                section_contents.append(section_content)

            logger.debug(
                f"Successfully parsed PDF file: {resolved_path} ({len(section_contents)} pages)"
            )

            return ParsedFile(
                name=Path(resolved_path).name,
                sections=section_contents,
            )

        except FileValidationError as e:
            logger.error(f"File validation failed for PDF file: {e}")
            raise ValueError(f"PDF file validation failed: {e}") from e

    async def _process_images_individually(
        self, 
        page: Any, 
        page_images: MutableSequence[Image], 
        image_descriptions: MutableSequence[str], 
        image_cache: dict[str, tuple[str, str]]
    ) -> None:
        """
        Fallback method to process images individually when page screenshot optimization fails.
        
        This method processes each image on a PDF page individually using the visual description
        provider, which is the original behavior before the optimization.
        
        Args:
            page: The PDF page object from pypdf
            page_images: List of Image objects to update with OCR text
            image_descriptions: List to append image descriptions to
            image_cache: Cache dictionary for storing processed image results
        """
        if not self.visual_description_provider:
            return
            
        for image_num, (image, page_image) in enumerate(zip(page.images, page_images)):
            image_bytes = image.data
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            if image_hash in image_cache:
                cached_md, cached_ocr = image_cache[image_hash]
                image_md = cached_md
                ocr_text = cached_ocr
            else:
                agent_input = FilePart(
                    mime_type=ext2mime(Path(image.name).suffix),
                    data=image.data,
                )

                agent_response = await self.visual_description_provider.generate_by_prompt_async(
                    agent_input,
                    developer_prompt="You are a helpful assistant that deeply understands visual media.",
                    response_schema=VisualMediaDescription,
                )

                image_md = agent_response.parsed.md
                ocr_text = agent_response.parsed.ocr_text or ""
                image_cache[image_hash] = (image_md, ocr_text or "")

            image_descriptions.append(
                f"Page Image {image_num + 1}: {image_md}"
            )
            page_image.ocr_text = ocr_text
