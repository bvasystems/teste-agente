"""
Compressed Archive Parser Module

This module provides functionality for parsing compressed archive files (ZIP, RAR, PKZ) into
structured representations. It can extract contents from compressed archives and parse the
contained files using appropriate parsers for each file type.
"""

from collections.abc import MutableSequence
from pathlib import Path
from typing import Literal, cast
import os as _os

from rsb.models.field import Field


from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_file import ParsedFile


class CompressedFileParser(DocumentParser):
    """
    Parser for processing compressed archive files (ZIP, RAR, PKZ).

    This parser extracts files from compressed archives and processes each file using
    the appropriate parser for its file type. It acts as a container parser that delegates
    the actual parsing work to specialized parsers for each contained file type.
    The results from all contained files are then combined into a single ParsedFile.

    **Attributes:**

    *   `inner_parser` (DocumentParser):
        The parser to use for parsing files extracted from the archive. Defaults to
        FileParser, which automatically selects the appropriate parser based on file extension.

        **Example:**
        ```python
        from agentle.parsing.parsers.file_parser import FileParser

        # Create a specialized inner parser with custom settings
        inner_parser = FileParser(strategy="low")

        # Use it in the compressed file parser
        parser = CompressedFileParser(inner_parser=inner_parser)
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe visual content. This agent is passed
        to the inner parsers when processing image files and other visual content.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        Note: You cannot use both visual_description_agent and multi_modal_provider
        at the same time.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Focus on technical content in diagrams and charts",
            response_schema=VisualMediaDescription
        )

        parser = CompressedFileParser(visual_description_agent=custom_agent)
        ```

    *   `multi_modal_provider` (GenerationProvider):
        An alternative to using a visual_description_agent. This is a generation
        provider capable of handling multi-modal content (text and images).
        Defaults to GoogleGenerationProvider().

        Note: You cannot use both visual_description_agent and multi_modal_provider
        at the same time.

    **Usage Examples:**

    Basic parsing of a compressed archive:
    ```python
    from agentle.parsing.parsers.compressed import CompressedFileParser

    # Create a parser with default settings
    parser = CompressedFileParser()

    # Parse a ZIP file
    parsed_archive = parser.parse("documents.zip")

    # Access the combined content from all files
    print(f"Archive contains {len(parsed_archive.sections)} sections")

    # Iterate through all sections from all files
    for i, section in enumerate(parsed_archive.sections):
        print(f"Section {i+1}: {section.text[:100]}...")
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse different archive types
    zip_result = parse("files.zip")
    rar_result = parse("documents.rar")

    # Process the results
    for result in [zip_result, rar_result]:
        print(f"Archive: {result.name}")
        print(f"Contains {len(result.sections)} sections")
    ```
    """

    type: Literal["compressed"] = "compressed"

    visual_description_provider: GenerationProvider | None = Field(
        default=None,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    audio_description_provider: GenerationProvider | None = Field(
        default=None,
    )
    """
    The agent to use for generating the audio description of the document.
    Useful when you want to customize the prompt for the audio description.
    """

    async def parse_async(self, document_path: str) -> ParsedFile:
        """
        Asynchronously parse a compressed archive file and process its contents.

        This method extracts files from a compressed archive (ZIP, RAR, or PKZ),
        processes each file using the appropriate parser for its file type, and
        combines the results into a single ParsedFile.

        Args:
            document_path (str): Path to the compressed archive file to be parsed

        Returns:
            ParsedFile: A structured representation combining the parsed content
                from all files in the archive

        Raises:
            ValueError: If the file extension is not supported (not ZIP, RAR, or PKZ)

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.compressed import CompressedFileParser

            async def process_archive():
                parser = CompressedFileParser()
                result = await parser.parse_async("documents.zip")

                # Print information about the contents
                print(f"Archive contains content from multiple files")
                print(f"Total sections: {len(result.sections)}")

                # Access the combined content
                for i, section in enumerate(result.sections):
                    print(f"Section {i+1} content type: {type(section).__name__}")
                    if section.images:
                        print(f"  Contains {len(section.images)} images")

            asyncio.run(process_archive())
            ```

        Note:
            This method creates a temporary copy of the archive file for processing.
            The temporary files are automatically cleaned up after processing.
        """
        import tempfile
        import zipfile

        import rarfile
        from agentle.parsing.parsers.file_parser import FileParser

        path = Path(document_path)
        # Normalize extension robustly: lowercase, strip whitespace, remove leading dot.
        # This guards against odd filenames like "archive.ZIP " or accidental trailing spaces.
        raw_suffix = path.suffix
        ext = raw_suffix.lower().strip().lstrip(".")  # '.zip' -> 'zip'

        # We'll accumulate ParsedFile objects from each extracted child file
        parsed_files: MutableSequence[ParsedFile] = []

        # Open archive directly from path (avoid redundant temp copy)
        match ext:
            case "zip" | "pkz":
                with zipfile.ZipFile(path, "r") as zip_ref:
                    for info in zip_ref.infolist():
                        if info.is_dir():
                            continue
                        # Extract member to a temporary file to hand off to FileParser
                        member_basename = Path(info.filename).name
                        if not member_basename:
                            continue
                        # Skip unsafe traversal attempts
                        if ".." in Path(info.filename).parts:
                            continue

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(member_basename).suffix
                        ) as child_tmp:
                            try:
                                with zip_ref.open(info, "r") as src:
                                    data_bytes = src.read()  # Expected bytes
                                    # Fallback conversion only if an unexpected type appears
                                    try:
                                        child_tmp.write(data_bytes)  # type: ignore[arg-type]
                                    except TypeError:
                                        child_tmp.write(bytes(data_bytes))  # type: ignore[arg-type]
                                child_tmp.flush()
                                parser = FileParser(
                                    visual_description_provider=self.visual_description_provider,
                                    audio_description_provider=self.audio_description_provider,
                                )
                                child_parsed = await parser.parse_async(child_tmp.name)
                                # Rename the parsed file's name to reflect original archive member
                                child_parsed.name = member_basename
                                parsed_files.append(child_parsed)
                            finally:
                                try:
                                    _os.unlink(child_tmp.name)
                                except Exception:
                                    pass

            case "rar":
                # Pre-flight: ensure rarfile found an external extraction utility.
                # rarfile relies on one of: unrar, unar, bsdtar. If none present, give a clear error.
                external_tools = [
                    getattr(rarfile, "UNRAR_TOOL", None),
                    getattr(rarfile, "UNAR_TOOL", None),
                    getattr(rarfile, "BSDTAR_TOOL", None),
                ]
                if not any(t for t in external_tools):
                    raise ValueError(
                        "RAR archive parsing requires an external tool. Install one of: 'unrar' (non-free), 'unar' (The Unarchiver), or 'bsdtar'."
                    )
                try:
                    with rarfile.RarFile(path, "r") as rar_ref:
                        for info in rar_ref.infolist():
                            try:
                                is_dir = info.isdir()  # type: ignore[attr-defined]
                            except Exception:
                                # Best-effort; if attribute missing treat as file
                                is_dir = False
                            if is_dir:
                                continue
                            member_basename = Path(cast(str, info.filename)).name  # type: ignore
                            if (
                                not member_basename
                                or ".." in Path(member_basename).parts
                            ):
                                continue

                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=Path(member_basename).suffix
                            ) as child_tmp:
                                try:
                                    with rar_ref.open(info, "r") as src:  # type: ignore[arg-type]
                                        data_bytes = src.read()
                                        try:
                                            child_tmp.write(data_bytes)  # type: ignore[arg-type]
                                        except TypeError:
                                            child_tmp.write(bytes(data_bytes))  # type: ignore[arg-type]
                                    child_tmp.flush()
                                    parser = FileParser(
                                        visual_description_provider=self.visual_description_provider,
                                        audio_description_provider=self.audio_description_provider,
                                    )
                                    child_parsed = await parser.parse_async(
                                        child_tmp.name
                                    )
                                    child_parsed.name = member_basename
                                    parsed_files.append(child_parsed)
                                finally:
                                    try:
                                        _os.unlink(child_tmp.name)
                                    except Exception:
                                        pass
                except rarfile.RarCannotExec as e:
                    raise ValueError(
                        "Failed to process RAR archive: no working extraction tool found. Install one of 'unrar', 'unar', or 'bsdtar' and ensure it is on PATH."
                    ) from e
                except rarfile.BadRarFile as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        f"The file '{path.name}' is not a valid or is a corrupted RAR archive."
                    ) from e
                except rarfile.NeedFirstVolume as e:  # type: ignore[attr-defined]
                    raise ValueError(
                        f"The file '{path.name}' is part of a multi-volume RAR set. Provide the first volume (.part1.rar)."
                    ) from e

            case _:
                raise ValueError(
                    f"CompressedFileParser does not handle extension: '{path.suffix or '(none)'}'. Supported extensions: .zip, .pkz, .rar"
                )

        # Merge all the parsed files into a single ParsedFile
        return ParsedFile.from_parsed_files(parsed_files)
