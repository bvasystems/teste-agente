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
from pathlib import Path
from typing import Literal

from rsb.functions.ext2mime import ext2mime
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

logger = logging.getLogger(__name__)


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

    async def parse_async(
        self,
        document_path: str,
    ) -> ParsedFile:
        """
        Asynchronously parse a Word document and generate a structured representation.

        This method reads a Word document, extracts its text content, and processes
        any embedded images. With the "high" strategy, images are analyzed using the
        visual description agent to extract text and generate descriptions.

        Args:
            document_path (str): Path to the Word document to be parsed

        Returns:
            ParsedFile: A structured representation where:
                - The document is represented as a single section
                - The section includes all text content from the document
                - Embedded images are extracted and (optionally) analyzed

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.docx import DocxFileParser

            async def process_document():
                parser = DocxFileParser(strategy="high")
                result = await parser.parse_async("report.docx")

                # Access the document content
                print(f"Document text: {result.sections[0].text[:200]}...")

                # Process images
                print(f"Document contains {len(result.sections[0].images)} images")
                for img in result.sections[0].images:
                    if img.ocr_text:
                        print(f"Image text: {img.ocr_text}")

            asyncio.run(process_document())
            ```

        Note:
            This method uses the python-docx library to read Word documents. For optimal
            results, use .docx format rather than the older .doc format.
        """
        from docx import Document

        document = Document(document_path)
        image_cache: dict[str, tuple[str, str]] = {}  # (md, ocr_text)

        # Prefer high-quality Markdown via MarkItDown when available
        md_text: str | None = None
        try:
            try:
                from markitdown import MarkItDown  # type: ignore

                md_converter = MarkItDown(enable_plugins=False)  # safe default
                md_result = md_converter.convert(document_path)
                if hasattr(md_result, "markdown") and md_result.markdown:
                    md_text = str(md_result.markdown)
            except ImportError:
                md_text = None
            except Exception as e:  # Conversion failed; fall back gracefully
                logger.warning(f"MarkItDown conversion failed for DOCX: {e}")
                md_text = None
        except Exception:
            # Extra-guard: never fail the parser just because of markdown conversion
            md_text = None

        # Fallback: basic paragraph join when MarkItDown isn't available
        if not md_text:
            paragraph_texts = [p.text for p in document.paragraphs if p.text.strip()]
            md_text = "\n\n".join(paragraph_texts)

        doc_images: list[tuple[str, bytes]] = []
        for rel in document.part._rels.values():  # type: ignore[reportPrivateUsage]
            if "image" in rel.reltype:
                image_part = rel.target_part
                image_name = image_part.partname.split("/")[-1]
                image_bytes = image_part.blob
                doc_images.append((image_name, image_bytes))

        final_images: list[Image] = []
        image_descriptions: list[str] = []

        # Always collect the Image objects first (OCR text may be filled later)
        for image_name, image_bytes in doc_images:
            final_images.append(
                Image(
                    name=image_name,
                    contents=image_bytes,
                    ocr_text="",
                )
            )

        if self.visual_description_provider and self.strategy == "high" and doc_images:
            # Optimization path: attempt to render page screenshots by converting DOCX -> PDF
            # using a headless converter (LibreOffice/soffice or pandoc) and then use PyMuPDF
            # to render pages that contain images. If any step fails, fall back to individual
            # image processing as before.
            used_page_screenshots = False
            try:
                try:
                    import fitz as pymupdf_module  # type: ignore
                except Exception:
                    pymupdf_module = None  # type: ignore

                def _try_convert_docx_to_pdf_headless(
                    input_path: str, out_dir: str
                ) -> str | None:
                    """Try headless DOCX->PDF using soffice/libreoffice or pandoc. Return PDF path or None."""
                    pdf_out = os.path.join(out_dir, f"{Path(input_path).stem}.pdf")
                    # Prefer soffice (LibreOffice)
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

                    # Fallback to pandoc
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

                if pymupdf_module is not None:  # Only try if PyMuPDF is available
                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_path = _try_convert_docx_to_pdf_headless(
                            document_path, temp_dir
                        )
                        if not pdf_path:
                            raise RuntimeError(
                                "Headless DOCX->PDF conversion not available or failed"
                            )

                        try:
                            mu_doc = pymupdf_module.open(pdf_path)  # type: ignore
                        except Exception as e:
                            logger.warning(
                                f"Failed to open converted PDF with PyMuPDF. Falling back: {e}"
                            )
                            raise

                        try:
                            page_ocr_texts: list[str] = []
                            for page_idx in range(mu_doc.page_count):  # type: ignore[attr-defined]
                                page_obj = mu_doc[page_idx]  # type: ignore[index]

                                # Determine if the page contains images
                                get_images = getattr(
                                    page_obj, "get_images", None
                                ) or getattr(page_obj, "getImages", None)
                                page_has_images = False
                                if callable(get_images):
                                    try:
                                        img_list = get_images(full=True)  # type: ignore[call-arg]
                                        page_has_images = bool(img_list)
                                    except Exception:
                                        page_has_images = (
                                            True  # if unsure, try rendering
                                        )

                                if not page_has_images:
                                    continue

                                # Render the page at higher resolution
                                matrix = getattr(pymupdf_module, "Matrix")(2.0, 2.0)  # type: ignore
                                get_pixmap = getattr(
                                    page_obj, "get_pixmap", None
                                ) or getattr(page_obj, "getPixmap", None)
                                if not callable(get_pixmap):
                                    continue
                                pix = get_pixmap(matrix=matrix)  # type: ignore[call-arg]
                                page_image_bytes: bytes = pix.tobytes("png")  # type: ignore[attr-defined]

                                # Cache by screenshot hash
                                page_hash = hashlib.sha256(page_image_bytes).hexdigest()
                                if page_hash in image_cache:
                                    cached_md, cached_ocr = image_cache[page_hash]
                                    page_description = cached_md
                                    page_ocr_text = cached_ocr
                                else:
                                    agent_input = FilePart(
                                        mime_type="image/png",
                                        data=page_image_bytes,
                                    )
                                    agent_response = await self.visual_description_provider.generate_by_prompt_async(
                                        agent_input,
                                        developer_prompt=(
                                            "You are a helpful assistant that deeply understands visual media. "
                                            "Analyze this Word document page screenshot and extract all text content and "
                                            "describe any visual elements like images, charts, diagrams, etc."
                                        ),
                                        response_schema=VisualMediaDescription,
                                    )
                                    page_description = agent_response.parsed.md
                                    page_ocr_text = agent_response.parsed.ocr_text or ""
                                    image_cache[page_hash] = (
                                        page_description,
                                        page_ocr_text,
                                    )

                                image_descriptions.append(
                                    f"Page Visual Content: {page_description}"
                                )
                                if page_ocr_text:
                                    page_ocr_texts.append(page_ocr_text)

                            if image_descriptions:
                                used_page_screenshots = True
                                # Best-effort: distribute combined OCR text to images (page-level granularity isn't available in DOCX)
                                if page_ocr_texts:
                                    combined_ocr = "\n\n".join(
                                        [
                                            f"OCR (page {i + 1}):\n{t}"
                                            for i, t in enumerate(page_ocr_texts)
                                        ]
                                    )
                                    for img in final_images:
                                        img.ocr_text = combined_ocr
                        finally:
                            try:
                                mu_doc.close()  # type: ignore[attr-defined]
                            except Exception:
                                pass

            except Exception:
                # Any failure in the screenshot optimization will fall through to per-image processing
                used_page_screenshots = False

            if not used_page_screenshots:
                # Fallback: process each embedded image individually
                for idx, (image_name, image_bytes) in enumerate(doc_images, start=1):
                    image_hash = hashlib.sha256(image_bytes).hexdigest()

                    if image_hash in image_cache:
                        cached_md, cached_ocr = image_cache[image_hash]
                        image_md = cached_md
                        ocr_text = cached_ocr
                    else:
                        agent_input = FilePart(
                            mime_type=ext2mime(Path(image_name).suffix),
                            data=image_bytes,
                        )
                        agent_response = await self.visual_description_provider.generate_by_prompt_async(
                            agent_input,
                            developer_prompt="You are a helpful assistant that deeply understands visual media.",
                            response_schema=VisualMediaDescription,
                        )
                        image_md = agent_response.parsed.md
                        ocr_text = agent_response.parsed.ocr_text or ""
                        image_cache[image_hash] = (image_md, ocr_text or "")

                    image_descriptions.append(f"Docx Image {idx}: {image_md}")
                    # Update the corresponding Image object in final_images
                    try:
                        img_obj = next(
                            img for img in final_images if img.name == image_name
                        )
                        img_obj.ocr_text = ocr_text
                    except StopIteration:
                        pass

            if image_descriptions:
                # Append a structured Visual Content section to the Markdown
                visual_md = [
                    "\n\n## Visual Content",
                    *(f"- {desc}" for desc in image_descriptions),
                ]
                md_text += "\n" + "\n".join(visual_md)

        return ParsedFile(
            name=document_path,
            sections=[
                SectionContent(
                    number=1,
                    text=md_text,
                    md=md_text,
                    images=final_images,
                )
            ],
        )
