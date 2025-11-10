from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, override

from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from agentle.generations.models.messages.user_message import UserMessage

if TYPE_CHECKING:
    import asyncpg


class PostgresConversationStore(ConversationStore):
    """A conversation store that persists conversations to PostgreSQL using asyncpg.

    This implementation stores messages in a PostgreSQL table with configurable
    table name and connection parameters.
    """

    _pool: asyncpg.Pool
    _table_name: str
    _chat_id_column: str
    _message_data_column: str
    _timestamp_column: str
    _id_column: str

    def __init__(
        self,
        pool: asyncpg.Pool,
        table_name: str = "conversations",
        chat_id_column: str = "chat_id",
        message_data_column: str = "message_data",
        timestamp_column: str = "created_at",
        id_column: str = "id",
        message_limit: int | None = None,
        override_old_messages: bool | None = None,
    ) -> None:
        """Initialize the PostgreSQL conversation store.

        Args:
            pool: asyncpg connection pool
            table_name: Name of the table to store conversations
            chat_id_column: Name of the column storing chat IDs
            message_data_column: Name of the column storing message JSON data
            timestamp_column: Name of the column storing timestamps
            id_column: Name of the primary key column
            message_limit: Maximum number of messages to store per conversation
            override_old_messages: Whether to remove old messages when limit is reached
        """
        super().__init__(message_limit, override_old_messages)
        self._pool = pool
        self._table_name = table_name
        self._chat_id_column = chat_id_column
        self._message_data_column = message_data_column
        self._timestamp_column = timestamp_column
        self._id_column = id_column

    async def ensure_table_exists(self) -> None:
        """Create the conversations table if it doesn't exist.

        This method creates a table with the following structure:
        - id: SERIAL PRIMARY KEY
        - chat_id: TEXT (configurable column name)
        - message_data: JSONB (configurable column name)
        - created_at: TIMESTAMP WITH TIME ZONE (configurable column name)
        """
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    {self._id_column} SERIAL PRIMARY KEY,
                    {self._chat_id_column} TEXT NOT NULL,
                    {self._message_data_column} JSONB NOT NULL,
                    {self._timestamp_column} TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_{self._chat_id_column} 
                ON {self._table_name} ({self._chat_id_column});
                
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_{self._chat_id_column}_{self._timestamp_column} 
                ON {self._table_name} ({self._chat_id_column}, {self._timestamp_column});
            """)

    def _message_to_dict(
        self,
        message: DeveloperMessage
        | UserMessage
        | AssistantMessage
        | GeneratedAssistantMessage[Any],
    ) -> dict[str, Any]:
        """Convert a message object to a dictionary for JSON serialization."""
        message_dict = message.model_dump()
        # Add message type for proper deserialization
        message_dict["_message_type"] = type(message).__name__
        return message_dict

    def _dict_to_message(
        self, message_dict: dict[str, Any]
    ) -> DeveloperMessage | UserMessage | AssistantMessage:
        """Convert a dictionary back to a message object."""
        message_type = message_dict.pop("_message_type", None)

        if message_type == "DeveloperMessage":
            return DeveloperMessage.model_validate(message_dict)
        elif message_type == "UserMessage":
            return UserMessage.model_validate(message_dict)
        elif message_type == "AssistantMessage":
            return AssistantMessage.model_validate(message_dict)
        else:
            # Fallback: try to determine type from content
            if "role" in message_dict:
                role = message_dict.get("role")
                if role == "developer":
                    return DeveloperMessage.model_validate(message_dict)
                elif role == "user":
                    return UserMessage.model_validate(message_dict)
                elif role == "assistant":
                    return AssistantMessage.model_validate(message_dict)

            # Default fallback to UserMessage
            return UserMessage.model_validate(message_dict)

    @override
    async def add_message_async[T = Any](
        self,
        chat_id: str,
        message: DeveloperMessage
        | UserMessage
        | AssistantMessage
        | GeneratedAssistantMessage[T],
    ) -> None:
        """Add a message to the conversation."""
        async with self._pool.acquire() as conn:
            # Apply message limit logic
            if self.message_limit is not None:
                # Get current message count
                count_query = f"""
                    SELECT COUNT(*) FROM {self._table_name} 
                    WHERE {self._chat_id_column} = $1
                """
                current_count = await conn.fetchval(count_query, chat_id)

                if current_count >= self.message_limit:
                    if self.override_old_messages:
                        # Remove oldest messages to make room
                        messages_to_remove = current_count - self.message_limit + 1
                        delete_query = f"""
                            DELETE FROM {self._table_name} 
                            WHERE {self._id_column} IN (
                                SELECT {self._id_column} FROM {self._table_name}
                                WHERE {self._chat_id_column} = $1
                                ORDER BY {self._timestamp_column} ASC
                                LIMIT $2
                            )
                        """
                        await conn.execute(delete_query, chat_id, messages_to_remove)
                    else:
                        # Don't add message if limit reached and not overriding
                        return

            # Add the new message
            message_dict = self._message_to_dict(message)
            insert_query = f"""
                INSERT INTO {self._table_name} ({self._chat_id_column}, {self._message_data_column})
                VALUES ($1, $2)
            """
            await conn.execute(insert_query, chat_id, json.dumps(message_dict))

    @override
    async def get_conversation_history_async(
        self, chat_id: str
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]:
        """Get the conversation history for a given chat ID."""
        async with self._pool.acquire() as conn:
            query = f"""
                SELECT {self._message_data_column} FROM {self._table_name}
                WHERE {self._chat_id_column} = $1
                ORDER BY {self._timestamp_column} ASC
            """
            rows = await conn.fetch(query, chat_id)

            messages = []
            for row in rows:
                try:
                    # Get the message data from the row
                    raw_data = row[self._message_data_column]

                    # Handle both string and dict types from JSONB column
                    if isinstance(raw_data, str):
                        message_dict: dict[str, Any] = json.loads(raw_data)
                    elif isinstance(raw_data, dict):
                        message_dict = raw_data
                    else:
                        # Skip if data type is unexpected
                        continue

                    message = self._dict_to_message(message_dict.copy())
                    messages.append(message)
                except Exception:
                    # Skip malformed messages
                    continue

            return messages

    @override
    async def clear_conversation_async(self, chat_id: str) -> None:
        """Clear the conversation history for a given chat ID."""
        async with self._pool.acquire() as conn:
            query = f"""
                DELETE FROM {self._table_name}
                WHERE {self._chat_id_column} = $1
            """
            await conn.execute(query, chat_id)

    async def get_all_chat_ids(self) -> list[str]:
        """Get all unique chat IDs in the store."""
        async with self._pool.acquire() as conn:
            query = f"""
                SELECT DISTINCT {self._chat_id_column} FROM {self._table_name}
                ORDER BY {self._chat_id_column}
            """
            rows = await conn.fetch(query)
            return [row[self._chat_id_column] for row in rows]

    async def get_conversation_count(self, chat_id: str) -> int:
        """Get the number of messages in a conversation."""
        async with self._pool.acquire() as conn:
            query = f"""
                SELECT COUNT(*) FROM {self._table_name}
                WHERE {self._chat_id_column} = $1
            """
            return await conn.fetchval(query, chat_id)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()
