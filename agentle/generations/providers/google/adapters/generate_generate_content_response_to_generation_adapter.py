from __future__ import annotations

import datetime
import uuid
from collections.abc import AsyncIterator
from logging import Logger
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from pydantic import BaseModel
from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.part import Part
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.providers.google.adapters.google_content_to_generated_assistant_message_adapter import (
    GoogleContentToGeneratedAssistantMessageAdapter,
)
from agentle.generations.providers.google.adapters.google_part_to_part_adapter import (
    GooglePartToPartAdapter,
)
from agentle.utils.make_fields_optional import make_fields_optional
from agentle.utils.parse_streaming_json import parse_streaming_json

if TYPE_CHECKING:
    from google.genai.types import (
        Candidate,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
    )


class GenerateGenerateContentResponseToGenerationAdapter[T](
    Adapter[
        "GenerateContentResponse | AsyncIterator[GenerateContentResponse]",
        Generation[T] | AsyncIterator[Generation[T]],
    ]
):
    """
    Adapter for converting Google AI GenerateContentResponse objects to Agentle Generation objects.
    Supports both single responses and streaming responses.
    """

    response_schema: type[T] | None
    preferred_id: uuid.UUID | None
    model: str
    google_content_to_message_adapter: (
        GoogleContentToGeneratedAssistantMessageAdapter[T] | None
    )

    def __init__(
        self,
        *,
        model: str,
        response_schema: type[T] | None,
        google_content_to_generated_assistant_message_adapter: GoogleContentToGeneratedAssistantMessageAdapter[
            T
        ]
        | None = None,
        preferred_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__()
        self.response_schema = response_schema
        self._logger = Logger(self.__class__.__name__)
        self.google_content_to_message_adapter = (
            google_content_to_generated_assistant_message_adapter
        )
        self.preferred_id = preferred_id
        self.model = model

    @overload
    def adapt(self, _f: "GenerateContentResponse") -> Generation[T]: ...

    @overload
    def adapt(
        self, _f: AsyncIterator["GenerateContentResponse"]
    ) -> AsyncIterator[Generation[T]]: ...

    def adapt(
        self, _f: "GenerateContentResponse | AsyncIterator[GenerateContentResponse]"
    ) -> Generation[T] | AsyncIterator[Generation[T]]:
        """
        Convert Google response(s) to Agentle Generation object(s).

        Args:
            _f: Either a single GenerateContentResponse or an async iterator of responses

        Returns:
            Either a single Generation or an async iterator of Generation objects
        """
        # Check if it's an async iterator by looking for __aiter__ method
        if hasattr(_f, "__aiter__"):
            return self._adapt_streaming(
                cast(AsyncIterator["GenerateContentResponse"], _f)
            )
        else:
            return self._adapt_single(cast("GenerateContentResponse", _f))

    def _adapt_single(self, response: "GenerateContentResponse") -> Generation[T]:
        """Adapt a single response (non-streaming)."""
        from google.genai import types

        parsed: T | None = cast(T | None, response.parsed)
        candidates: list[types.Candidate] | None = response.candidates

        if candidates is None:
            raise ValueError("The provided candidates by Google are NONE.")

        choices: list[Choice[T]] = self._build_choices(
            candidates=candidates,
            generate_content_parsed_response=parsed,
        )

        usage = self._extract_usage(response.usage_metadata)

        return Generation[T](
            id=self.preferred_id or uuid.uuid4(),
            object="chat.generation",
            created=datetime.datetime.now(),
            model=self.model,
            choices=choices,
            usage=usage,
        )

    async def _adapt_streaming(
        self, response_stream: AsyncIterator["GenerateContentResponse"]
    ) -> AsyncIterator[Generation[T]]:
        """Adapt a streaming response with proper text accumulation."""
        generation_id = self.preferred_id or uuid.uuid4()
        created_time = datetime.datetime.now()

        # Keep track of accumulated content for final parsing
        accumulated_text_parts: list[str] = []  # Store all text chunks
        final_usage: Usage | None = None
        final_parsed: T | None = None

        _response_schema = self.response_schema

        _all_parts: list[Part] = []
        _optional_model = (
            make_fields_optional(cast(type[BaseModel], _response_schema))
            if _response_schema is not None
            else None
        )

        async for chunk in response_stream:
            # Process structured output if needed
            if _response_schema:
                candidates = chunk.candidates
                if candidates is None:
                    continue

                content = candidates[0].content
                if content is None:
                    continue

                _part_adapter = GooglePartToPartAdapter()
                _content_parts = content.parts
                if _content_parts is None:
                    continue

                _parts = [_part_adapter.adapt(part) for part in _content_parts]
                _all_parts.extend(_parts)

                if _optional_model is not None:
                    optional_model = parse_streaming_json(
                        "".join([str(p.text) for p in _all_parts]),
                        model=_optional_model,
                    )
                    chunk.parsed = optional_model
                else:
                    chunk.parsed = None

            # Extract parsed data (usually only available in final chunk)
            if hasattr(chunk, "parsed") and chunk.parsed is not None:
                final_parsed = cast(T | None, chunk.parsed)

            # Extract usage (usually only in final chunk)
            if chunk.usage_metadata is not None:
                final_usage = self._extract_usage(chunk.usage_metadata)

            # Process candidates in this chunk
            if chunk.candidates:
                # Extract new text from this chunk
                current_chunk_text = ""
                for candidate in chunk.candidates:
                    candidate_content = candidate.content
                    if candidate_content and candidate_content.parts:
                        for part in candidate_content.parts:
                            if part.text:
                                current_chunk_text += part.text

                # Add this chunk's text to our accumulator
                if current_chunk_text:
                    accumulated_text_parts.append(current_chunk_text)

                # Create accumulated text up to this point
                full_accumulated_text = "".join(accumulated_text_parts)

                # Build choices from candidates, optionally with accumulated text
                choices = self._build_choices(
                    candidates=chunk.candidates,
                    accumulated_text=full_accumulated_text
                    if full_accumulated_text
                    else None,
                    generate_content_parsed_response=final_parsed
                    if self.response_schema
                    else None,
                )

                # Use accumulated usage or default
                current_usage = final_usage or Usage(
                    prompt_tokens=0, completion_tokens=0
                )

                yield Generation[T](
                    id=generation_id,  # Same ID for all chunks
                    object="chat.generation",
                    created=created_time,  # Same timestamp for all chunks
                    model=self.model,
                    choices=choices,
                    usage=current_usage,
                )

    def _build_choices(
        self,
        candidates: list["Candidate"] | None = None,
        accumulated_text: str | None = None,
        generate_content_parsed_response: T | None = None,
    ) -> list[Choice[T]]:
        """
        Build Choice objects from candidates, optionally replacing text with accumulated text.

        This unified method handles both streaming and non-streaming scenarios:
        - Always processes candidates if available (to extract tool calls, etc.)
        - For streaming: replaces text parts with accumulated_text if provided
        - Always ensures at least one choice is created, even if only tool calls are present

        Args:
            candidates: List of Google candidate responses
            accumulated_text: Accumulated text content (for streaming text replacement)
            generate_content_parsed_response: Parsed structured output

        Returns:
            List of Choice objects
        """
        from google.genai import types

        choices: list[Choice[T]] = []

        # Case 1: Process candidates (both streaming and non-streaming)
        if candidates is not None:
            content_to_message_adapter = (
                self.google_content_to_message_adapter
                or GoogleContentToGeneratedAssistantMessageAdapter(
                    generate_content_response_parsed=generate_content_parsed_response,
                )
            )

            index = 0
            for candidate in candidates:
                candidate_content: types.Content | None = candidate.content
                if candidate_content is None:
                    continue

                # Get the adapted message from the candidate
                adapted_message = content_to_message_adapter.adapt(candidate_content)

                # If we have accumulated_text, replace text parts but keep other parts (like tool calls)
                if accumulated_text is not None:
                    # Separate text parts from non-text parts (tool execution suggestions, etc.)
                    non_text_parts = [
                        part
                        for part in adapted_message.parts
                        if not isinstance(part, TextPart)
                    ]

                    # Create new parts list with accumulated text + non-text parts
                    new_parts: list[Any] = []
                    if (
                        accumulated_text
                    ):  # Only add text part if there's accumulated text
                        new_parts.append(TextPart(text=accumulated_text))
                    new_parts.extend(non_text_parts)

                    # Create new message with updated parts
                    message = GeneratedAssistantMessage[T](
                        parts=new_parts,
                        parsed=adapted_message.parsed,
                    )
                else:
                    # Use the adapted message as-is
                    message = adapted_message

                choices.append(Choice[T](index=index, message=message))
                index += 1

            return choices

        # Case 2: No candidates but have accumulated text - create choice with just text
        if accumulated_text is not None:
            message = GeneratedAssistantMessage[T](
                parts=[TextPart(text=accumulated_text)],
                parsed=generate_content_parsed_response
                if generate_content_parsed_response
                else cast(T, None),
            )
            choices.append(Choice[T](index=0, message=message))
            return choices

        # Case 3: No candidates and no accumulated text - create empty choice
        # This ensures we always have at least one choice even if there's no content yet
        message = GeneratedAssistantMessage[T](
            parts=[],
            parsed=generate_content_parsed_response
            if generate_content_parsed_response
            else cast(T, None),
        )

        choices.append(Choice[T](index=0, message=message))
        return choices

    def _extract_usage(
        self, usage_metadata: GenerateContentResponseUsageMetadata | None
    ) -> Usage:
        """Extract usage information from Google's usage metadata."""
        if usage_metadata is None:
            self._logger.warning(
                "WARNING: No usage metadata returned by Google. Assuming 0"
            )
            return Usage(prompt_tokens=0, completion_tokens=0)

        prompt_token_count = (
            usage_metadata.prompt_token_count
            if usage_metadata.prompt_token_count
            else self._warn_and_default(field_name="prompt_token_count")
        )

        candidates_token_count = (
            usage_metadata.candidates_token_count
            if usage_metadata.candidates_token_count
            else self._warn_and_default(field_name="candidates_token_count")
        )

        return Usage(
            prompt_tokens=prompt_token_count,
            completion_tokens=candidates_token_count,
        )

    def _warn_and_default(self, field_name: str) -> Literal[0]:
        """Log a warning about missing metadata and return a default value."""
        self._logger.warning(
            f"WARNING: No information found about {field_name}. Defaulting to 0."
        )
        return 0
