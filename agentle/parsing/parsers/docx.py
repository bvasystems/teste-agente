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
from typing import Literal, cast

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
        Parse a Word document into a single Markdown section and describe visuals without duplicating page OCR.
        """
        from docx import Document

        document = Document(document_path)

        # Base Markdown via MarkItDown (best-effort)
        md_text: str | None = None
        try:
            try:
                from markitdown import MarkItDown  # type: ignore

                md_converter = MarkItDown(enable_plugins=False)
                md_result = md_converter.convert(document_path)
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
            paragraph_texts = [p.text for p in document.paragraphs if p.text.strip()]
            md_text = "\n\n".join(paragraph_texts)

        # Extract embedded images (kept as Image objects; OCR left empty to avoid duplication)
        doc_images: list[tuple[str, bytes]] = []
        for rel in document.part._rels.values():  # type: ignore[reportPrivateUsage]
            if "image" in rel.reltype:
                image_part = rel.target_part
                image_name = image_part.partname.split("/")[-1]
                image_bytes = image_part.blob
                doc_images.append((image_name, image_bytes))

        final_images: list[Image] = [
            Image(name=name, contents=bytes(data), ocr_text="")
            for name, data in doc_images
        ]

        image_descriptions: list[str] = []
        image_cache: dict[str, tuple[str, str]] = {}

        if self.visual_description_provider and self.strategy == "high" and doc_images:
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
                            pix = get_pixmap(matrix=matrix)  # type: ignore[call-arg]
                            page_image_bytes: bytes = cast(bytes, pix.tobytes("png"))  # type: ignore[attr-defined]

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
                ["\n\n## Visual Content", *[f"- {desc}" for desc in image_descriptions]]
            )

        return ParsedFile(
            name=document_path,
            sections=[
                SectionContent(number=1, text=md_text, md=md_text, images=final_images)
            ],
        )
