"""
Module for PDF page extraction structured output.

This schema is used when sending an entire PDF to an AI provider that supports
native PDF processing, allowing the AI to extract and format content directly.
"""

from typing import Sequence

from pydantic import BaseModel, Field


class PDFPageContent(BaseModel):
    """Content extracted from a single PDF page.

    Attributes:
        page_number: The page number (1-indexed)
        markdown: The markdown-formatted content of the page
        has_images: Whether this page contains images or visual elements
        image_descriptions: Optional descriptions of images on this page
    """

    page_number: int = Field(
        description="The page number (1-indexed) this content belongs to."
    )

    markdown: str = Field(
        description=(
            "The complete content of this page formatted as markdown. "
            "Include all text content, preserving structure with markdown formatting. "
            "For tables, use markdown table syntax. "
            "For lists, use markdown list syntax. "
            "For headings, use markdown heading syntax. "
            "If the page contains images or charts, describe their content inline using markdown image syntax or blockquotes."
        )
    )

    has_images: bool = Field(
        default=False,
        description="Indicates whether this page contains images, charts, diagrams, or other visual elements.",
    )

    image_descriptions: Sequence[str] | None = Field(
        default=None,
        description=(
            "List of descriptions for any images, charts, or visual elements on this page. "
            "Each description should be concise but informative."
        ),
    )


class PDFPageExtraction(BaseModel):
    """Complete extraction of a PDF document into structured page content.

    This schema represents the output when an AI provider processes an entire
    PDF document and extracts its content page by page.

    Attributes:
        pages: Sequence of extracted page contents
        total_pages: Total number of pages in the document
        document_title: Optional title extracted from the document
    """

    pages: Sequence[PDFPageContent] = Field(
        description=(
            "A sequence of page contents, one for each page in the PDF document. "
            "Each page should contain the markdown-formatted content of that page."
        )
    )

    total_pages: int = Field(
        description="The total number of pages in the PDF document."
    )

    document_title: str | None = Field(
        default=None,
        description="The title of the document if it can be determined from the content or metadata.",
    )
