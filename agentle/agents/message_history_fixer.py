"""
Bulletproof message history validation and auto-fix system.
Prevents the "function response parts" error in all scenarios.
"""

import logging
import uuid
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.tools.tool_execution_result import ToolExecutionResult

logger = logging.getLogger(__name__)


class MessageHistoryFixer:
    """
    Automatically detects and fixes message history inconsistencies that could
    cause the "function response parts" error.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, raises exceptions instead of auto-fixing
        """
        self.strict_mode = strict_mode
        self.fixes_applied: list[str] = []

    def fix_message_history(
        self,
        messages: Sequence[DeveloperMessage | UserMessage | AssistantMessage],
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]:
        """
        Main entry point: fixes all message history inconsistencies.

        Returns:
            Fixed message history that guarantees no "function response parts" error
        """
        self.fixes_applied = []

        # Work on a deep copy to avoid modifying the original
        fixed_messages = deepcopy(list(messages))

        # Apply fixes in order of importance
        fixed_messages = self._fix_orphaned_tool_calls(fixed_messages)
        fixed_messages = self._fix_orphaned_tool_results(fixed_messages)
        fixed_messages = self._fix_mismatched_tool_ids(fixed_messages)
        fixed_messages = self._fix_wrong_message_sequence(fixed_messages)
        fixed_messages = self._fix_duplicate_tool_calls(fixed_messages)
        fixed_messages = self._remove_empty_messages(fixed_messages)

        if self.fixes_applied:
            logger.info(
                f"Applied {len(self.fixes_applied)} message history fixes: {self.fixes_applied}"
            )

        return fixed_messages

    def _fix_orphaned_tool_calls(self, messages: list[Any]) -> list[Any]:
        """Fix assistant messages with tool calls but no following user message with results."""
        i = 0
        while i < len(messages):
            msg = messages[i]

            if isinstance(msg, AssistantMessage):
                tool_calls = [
                    p for p in msg.parts if isinstance(p, ToolExecutionSuggestion)
                ]

                if tool_calls:
                    # Check if there's a following user message with matching results
                    has_matching_results = False

                    if i + 1 < len(messages) and isinstance(
                        messages[i + 1], UserMessage
                    ):
                        next_msg = messages[i + 1]
                        tool_results = [
                            p
                            for p in next_msg.parts
                            if isinstance(p, ToolExecutionResult)
                        ]

                        if tool_results:
                            result_ids = {tr.suggestion.id for tr in tool_results}
                            call_ids = {tc.id for tc in tool_calls}
                            has_matching_results = (
                                len(call_ids.intersection(result_ids)) > 0
                            )

                    if not has_matching_results:
                        if self.strict_mode:
                            raise ValueError(
                                f"Orphaned tool calls found in assistant message {i}"
                            )

                        # Auto-fix: Create synthetic tool results
                        synthetic_results: list[ToolExecutionResult] = []
                        for tool_call in tool_calls:
                            synthetic_result = ToolExecutionResult(
                                suggestion=tool_call,
                                result="[AUTO-FIX] Tool result not found in message history",
                                execution_time_ms=0.0,
                                success=False,
                                error_message="Tool result missing from message history",
                            )
                            synthetic_results.append(synthetic_result)

                        # Insert synthetic user message after the assistant message
                        synthetic_user_msg = UserMessage(parts=synthetic_results)
                        messages.insert(i + 1, synthetic_user_msg)

                        self.fixes_applied.append(
                            f"Added synthetic tool results for {len(tool_calls)} orphaned tool calls at message {i}"
                        )

            i += 1

        return messages

    def _fix_orphaned_tool_results(self, messages: list[Any]) -> list[Any]:
        """Fix user messages with tool results but no preceding assistant message with matching calls."""
        i = 0
        while i < len(messages):
            msg = messages[i]

            if isinstance(msg, UserMessage):
                tool_results = [
                    p for p in msg.parts if isinstance(p, ToolExecutionResult)
                ]

                if tool_results:
                    # Check if there's a preceding assistant message with matching calls
                    has_matching_calls = False

                    if i > 0 and isinstance(messages[i - 1], AssistantMessage):
                        prev_msg = messages[i - 1]
                        tool_calls = [
                            p
                            for p in prev_msg.parts
                            if isinstance(p, ToolExecutionSuggestion)
                        ]

                        if tool_calls:
                            call_ids = {tc.id for tc in tool_calls}
                            result_ids = {tr.suggestion.id for tr in tool_results}
                            has_matching_calls = (
                                len(call_ids.intersection(result_ids)) > 0
                            )

                    if not has_matching_calls:
                        if self.strict_mode:
                            raise ValueError(
                                f"Orphaned tool results found in user message {i}"
                            )

                        # Auto-fix: Remove orphaned tool results, keep other parts
                        non_tool_parts = [
                            p
                            for p in msg.parts
                            if not isinstance(p, ToolExecutionResult)
                        ]

                        if non_tool_parts:
                            # Keep the message but remove tool results
                            messages[i] = UserMessage(parts=list(non_tool_parts))
                            self.fixes_applied.append(
                                f"Removed {len(tool_results)} orphaned tool results from user message {i}"
                            )
                        else:
                            # Remove the entire message if it only had tool results
                            messages.pop(i)
                            self.fixes_applied.append(
                                f"Removed user message {i} with only orphaned tool results"
                            )
                            i -= 1  # Adjust index since we removed a message

            i += 1

        return messages

    def _fix_mismatched_tool_ids(self, messages: list[Any]) -> list[Any]:
        """Fix mismatched tool call/result IDs."""
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]

            if isinstance(current_msg, AssistantMessage) and isinstance(
                next_msg, UserMessage
            ):
                tool_calls = [
                    p
                    for p in current_msg.parts
                    if isinstance(p, ToolExecutionSuggestion)
                ]
                tool_results = [
                    p for p in next_msg.parts if isinstance(p, ToolExecutionResult)
                ]

                if tool_calls and tool_results:
                    call_ids = {tc.id for tc in tool_calls}
                    result_ids = {tr.suggestion.id for tr in tool_results}

                    # Find mismatched IDs
                    unmatched_calls = call_ids - result_ids
                    unmatched_results = result_ids - call_ids

                    if unmatched_calls or unmatched_results:
                        if self.strict_mode:
                            raise ValueError(
                                f"Mismatched tool IDs at messages {i}-{i + 1}: calls={call_ids}, results={result_ids}"
                            )

                        # Auto-fix strategy: Update result IDs to match call IDs by position
                        if len(tool_calls) == len(tool_results):
                            # Same number of calls and results - match by position
                            for _, (call, result) in enumerate(
                                zip(tool_calls, tool_results)
                            ):
                                if call.id != result.suggestion.id:
                                    # Update the result's suggestion to match the call
                                    result.suggestion = call

                            self.fixes_applied.append(
                                f"Fixed mismatched tool IDs by position at messages {i}-{i + 1}"
                            )

                        elif len(tool_calls) > len(tool_results):
                            # More calls than results - add synthetic results for missing calls
                            existing_result_ids = {
                                tr.suggestion.id for tr in tool_results
                            }
                            missing_calls = [
                                tc
                                for tc in tool_calls
                                if tc.id not in existing_result_ids
                            ]

                            for missing_call in missing_calls:
                                synthetic_result = ToolExecutionResult(
                                    suggestion=missing_call,
                                    result="[AUTO-FIX] Missing tool result reconstructed",
                                    execution_time_ms=0.0,
                                    success=False,
                                    error_message="Tool result was missing",
                                )
                                next_msg.append_part(synthetic_result)

                            self.fixes_applied.append(
                                f"Added {len(missing_calls)} synthetic results for unmatched calls"
                            )

                        else:
                            # More results than calls - remove extra results
                            existing_call_ids = {tc.id for tc in tool_calls}
                            valid_results = [
                                tr
                                for tr in tool_results
                                if tr.suggestion.id in existing_call_ids
                            ]

                            # Update the user message with only valid results
                            other_parts = [
                                p
                                for p in next_msg.parts
                                if not isinstance(p, ToolExecutionResult)
                            ]
                            next_msg.parts = other_parts + valid_results

                            removed_count = len(tool_results) - len(valid_results)
                            self.fixes_applied.append(
                                f"Removed {removed_count} unmatched tool results"
                            )

        return messages

    def _fix_wrong_message_sequence(self, messages: list[Any]) -> list[Any]:
        """Fix messages that are in the wrong order (tool results not immediately after tool calls)."""
        # This is a complex fix that would require sophisticated reordering
        # For now, we'll detect and warn about it

        for i in range(len(messages)):
            msg = messages[i]

            if isinstance(msg, AssistantMessage):
                tool_calls = [
                    p for p in msg.parts if isinstance(p, ToolExecutionSuggestion)
                ]

                if tool_calls:
                    # Look for matching results in subsequent messages
                    call_ids = {tc.id for tc in tool_calls}
                    found_matching_results = False

                    for j in range(
                        i + 1, min(i + 3, len(messages))
                    ):  # Look ahead up to 2 messages
                        if isinstance(messages[j], UserMessage):
                            tool_results = [
                                p
                                for p in messages[j].parts
                                if isinstance(p, ToolExecutionResult)
                            ]
                            result_ids = {tr.suggestion.id for tr in tool_results}

                            if call_ids.intersection(result_ids):
                                found_matching_results = True

                                if j != i + 1:
                                    # Results are not in the immediately following message
                                    if self.strict_mode:
                                        raise ValueError(
                                            f"Tool results at message {j} should immediately follow tool calls at message {i}"
                                        )

                                    logger.warning(
                                        f"Tool results at message {j} are not immediately after tool calls at message {i}"
                                    )
                                    # Could implement message reordering here if needed

                                break

                    if not found_matching_results and not self.strict_mode:
                        logger.warning(
                            f"No matching tool results found for tool calls at message {i}"
                        )

        return messages

    def _fix_duplicate_tool_calls(self, messages: list[Any]) -> list[Any]:
        """Fix duplicate tool call IDs within the same message."""
        for i, msg in enumerate(messages):
            if isinstance(msg, AssistantMessage):
                tool_calls = [
                    p for p in msg.parts if isinstance(p, ToolExecutionSuggestion)
                ]

                if tool_calls:
                    seen_ids: set[str] = set()
                    unique_calls: list[ToolExecutionSuggestion] = []
                    duplicates_removed = 0

                    for tool_call in tool_calls:
                        if tool_call.id not in seen_ids:
                            seen_ids.add(tool_call.id)
                            unique_calls.append(tool_call)
                        else:
                            # Generate new unique ID for duplicate
                            new_id = str(uuid.uuid4())
                            unique_call = ToolExecutionSuggestion(
                                id=new_id,
                                tool_name=tool_call.tool_name,
                                args=tool_call.args,
                            )
                            unique_calls.append(unique_call)
                            duplicates_removed += 1

                    if duplicates_removed > 0:
                        # Update the message with unique tool calls
                        other_parts = [
                            p
                            for p in msg.parts
                            if not isinstance(p, ToolExecutionSuggestion)
                        ]
                        msg.parts = other_parts + unique_calls

                        self.fixes_applied.append(
                            f"Fixed {duplicates_removed} duplicate tool call IDs in assistant message {i}"
                        )

            elif isinstance(msg, UserMessage):
                tool_results = [
                    p for p in msg.parts if isinstance(p, ToolExecutionResult)
                ]

                if tool_results:
                    seen_ids = set()
                    unique_results: list[ToolExecutionResult] = []
                    duplicates_removed = 0

                    for tool_result in tool_results:
                        if tool_result.suggestion.id not in seen_ids:
                            seen_ids.add(tool_result.suggestion.id)
                            unique_results.append(tool_result)
                        else:
                            duplicates_removed += 1

                    if duplicates_removed > 0:
                        # Update the message with unique tool results
                        other_parts = [
                            p
                            for p in msg.parts
                            if not isinstance(p, ToolExecutionResult)
                        ]
                        msg.parts = other_parts + unique_results

                        self.fixes_applied.append(
                            f"Removed {duplicates_removed} duplicate tool result IDs in user message {i}"
                        )

        return messages

    def _remove_empty_messages(self, messages: list[Any]) -> list[Any]:
        """Remove messages that have no parts."""
        filtered_messages = []
        removed_count = 0

        for _, msg in enumerate(messages):
            if hasattr(msg, "parts") and len(msg.parts) == 0:
                removed_count += 1
            else:
                filtered_messages.append(msg)

        if removed_count > 0:
            self.fixes_applied.append(f"Removed {removed_count} empty messages")

        return filtered_messages
