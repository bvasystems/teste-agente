from __future__ import annotations

import datetime
import uuid
from collections.abc import AsyncIterator
from logging import Logger
from typing import TYPE_CHECKING, Literal, cast, overload

from pydantic import BaseModel
from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.part import Part
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
    is_stream: bool

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
            candidates,
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
        """Adapt a streaming response."""
        generation_id = self.preferred_id or uuid.uuid4()
        created_time = datetime.datetime.now()

        # Keep track of accumulated content for final parsing
        accumulated_text = ""
        final_usage: Usage | None = None
        final_parsed: T | None = None

        _response_schema = self.response_schema

        _all_parts: list[Part] = []
        _optional_model = make_fields_optional(cast(type[BaseModel], _response_schema))

        async for chunk in response_stream:
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

                optional_model = parse_streaming_json(
                    "".join([str(p.text) for p in _all_parts]),
                    model=_optional_model,
                )

                chunk.parsed = optional_model

            # Extract parsed data (usually only available in final chunk)
            if hasattr(chunk, "parsed") and chunk.parsed is not None:
                final_parsed = cast(T | None, chunk.parsed)

            # Extract usage (usually only in final chunk)
            if chunk.usage_metadata is not None:
                final_usage = self._extract_usage(chunk.usage_metadata)

            # Process candidates in this chunk
            if chunk.candidates:
                choices = self._build_choices(
                    chunk.candidates,
                    generate_content_parsed_response=final_parsed
                    if self.response_schema
                    else None,
                    is_streaming=True,
                )

                # Accumulate text content for debugging/logging
                for choice in choices:
                    if choice.message.parts:
                        for part in choice.message.parts:
                            if hasattr(part, "text"):
                                accumulated_text += getattr(part, "text", "")

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
        candidates: list["Candidate"],
        generate_content_parsed_response: T | None,
        is_streaming: bool = False,
    ) -> list[Choice[T]]:
        """Build Choice objects from Google candidate responses."""
        from google.genai import types

        content_to_message_adapter = (
            self.google_content_to_message_adapter
            or GoogleContentToGeneratedAssistantMessageAdapter(
                generate_content_response_parsed=generate_content_parsed_response,
            )
        )

        choices: list[Choice[T]] = []
        index = 0

        for candidate in candidates:
            candidate_content: types.Content | None = candidate.content
            if candidate_content is None:
                continue

            choices.append(
                Choice[T](
                    index=index,
                    message=content_to_message_adapter.adapt(candidate_content),
                )
            )
            index += 1

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
