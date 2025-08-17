import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage


class JSONFileConversationStore(ConversationStore):
    """A conversation store that persists conversations to JSON files.

    Each conversation is stored as a separate JSON file in the specified directory.
    The filename format is: {chat_id}.json
    """

    _storage_dir: Path

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        message_limit: int | None = None,
        override_old_messages: bool | None = None,
    ) -> None:
        """Initialize the JSON file conversation store.

        Args:
            storage_dir: Directory where conversation JSON files will be stored.
                        Defaults to './conversations' if not provided.
            message_limit: Maximum number of messages to store per conversation
            override_old_messages: Whether to remove old messages when limit is reached
        """
        super().__init__(message_limit, override_old_messages)
        if storage_dir is None:
            storage_dir = "./.conversations"
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, chat_id: str) -> Path:
        """Get the file path for a given chat ID."""
        # Sanitize chat_id to be filesystem-safe
        safe_chat_id = "".join(
            c for c in chat_id if c.isalnum() or c in ("-", "_", ".")
        )
        return self._storage_dir / f"{safe_chat_id}.json"

    def _load_messages(self, chat_id: str) -> list[dict[str, Any]]:
        """Load messages from the JSON file for a given chat ID."""
        file_path = self._get_file_path(chat_id)
        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_messages(self, chat_id: str, messages: list[dict[str, Any]]) -> None:
        """Save messages to the JSON file for a given chat ID."""
        file_path = self._get_file_path(chat_id)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    messages,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=self._json_serializer,
                )
        except OSError:
            # Handle file write errors gracefully
            pass

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer to handle bytes and other non-serializable objects."""
        if isinstance(obj, bytes):
            # Convert bytes to base64 string
            import base64

            return base64.b64encode(obj).decode("utf-8")

        # For other non-serializable objects, convert to string
        return str(obj)

    def _message_to_dict(
        self, message: DeveloperMessage | UserMessage | AssistantMessage
    ) -> dict[str, Any]:
        """Convert a message object to a dictionary for JSON serialization."""
        # Get the basic message dictionary
        message_dict = message.model_dump()

        # Add message type for proper deserialization
        message_dict["_message_type"] = type(message).__name__

        # Process parts to handle FilePart objects with bytes data
        if "parts" in message_dict:
            processed_parts = []
            for part in message_dict["parts"]:
                if isinstance(part, dict) and part.get("type") == "file":
                    # Handle FilePart with potential bytes data
                    processed_part = part.copy()
                    if "data" in processed_part and isinstance(
                        processed_part["data"], bytes
                    ):
                        # Convert bytes to base64 string
                        import base64

                        processed_part["data"] = base64.b64encode(
                            processed_part["data"]
                        ).decode("utf-8")
                        processed_part["_data_encoding"] = (
                            "base64"  # Mark as base64 encoded
                        )
                    processed_parts.append(processed_part)
                else:
                    processed_parts.append(part)
            message_dict["parts"] = processed_parts

        return message_dict

    def _dict_to_message(
        self, message_dict: dict[str, Any]
    ) -> DeveloperMessage | UserMessage | AssistantMessage:
        """Convert a dictionary back to a message object."""
        # Make a copy to avoid modifying the original
        message_data = message_dict.copy()
        message_type = message_data.pop("_message_type", None)

        # Process parts to restore FilePart objects with bytes data
        if "parts" in message_data:
            processed_parts = []
            for part in message_data["parts"]:
                if isinstance(part, dict) and part.get("type") == "file":
                    # Handle FilePart that might have base64 encoded data
                    processed_part: dict[str, Any] = part.copy()
                    if (
                        "data" in processed_part
                        and processed_part.get("_data_encoding") == "base64"
                    ):
                        # Convert base64 string back to bytes
                        import base64

                        processed_part["data"] = base64.b64decode(
                            processed_part["data"]
                        )
                        processed_part.pop(
                            "_data_encoding", None
                        )  # Remove encoding marker
                    processed_parts.append(processed_part)
                else:
                    processed_parts.append(part)
            message_data["parts"] = processed_parts

        if message_type == "DeveloperMessage":
            return DeveloperMessage.model_validate(message_data)
        elif message_type == "UserMessage":
            return UserMessage.model_validate(message_data)
        elif message_type == "AssistantMessage":
            return AssistantMessage.model_validate(message_data)
        else:
            # Fallback: try to determine type from content
            if "role" in message_data:
                role = message_data.get("role")
                if role == "developer":
                    return DeveloperMessage.model_validate(message_data)
                elif role == "user":
                    return UserMessage.model_validate(message_data)
                elif role == "assistant":
                    return AssistantMessage.model_validate(message_data)

            # Default fallback to UserMessage
            return UserMessage.model_validate(message_data)

    @override
    async def add_message_async(
        self, chat_id: str, message: DeveloperMessage | UserMessage | AssistantMessage
    ) -> None:
        """Add a message to the conversation."""
        messages_data = self._load_messages(chat_id)

        # Apply message limit logic
        if self.message_limit is not None:
            if len(messages_data) >= self.message_limit:
                if self.override_old_messages:
                    # Remove oldest messages to make room
                    messages_to_remove = len(messages_data) - self.message_limit + 1
                    messages_data = messages_data[messages_to_remove:]
                else:
                    # Don't add message if limit reached and not overriding
                    return

        # Add the new message using our custom serialization
        message_dict = self._message_to_dict(message)
        messages_data.append(message_dict)

        # Save to file
        self._save_messages(chat_id, messages_data)

    @override
    async def get_conversation_history_async(
        self, chat_id: str
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]:
        """Get the conversation history for a given chat ID."""
        messages_data = self._load_messages(chat_id)

        # Convert dictionaries back to message objects
        messages = []
        for message_dict in messages_data:
            try:
                message = self._dict_to_message(message_dict.copy())
                messages.append(message)
            except Exception:
                # Skip malformed messages
                continue

        return messages

    @override
    async def clear_conversation_async(self, chat_id: str) -> None:
        """Clear the conversation history for a given chat ID."""
        file_path = self._get_file_path(chat_id)

        try:
            if file_path.exists():
                os.remove(file_path)
        except OSError:
            # Handle file deletion errors gracefully
            pass
