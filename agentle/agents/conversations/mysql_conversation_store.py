from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, override

from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage

if TYPE_CHECKING:
    from aiomysql import Pool

logger = logging.getLogger(__name__)


class MySQLConversationStore(ConversationStore):
    """
    A conversation store that uses MySQL database for persistence.
    
    This class provides a MySQL-based implementation of the ConversationStore
    abstract base class, using aiomysql for asynchronous database operations.
    
    Args:
        pool: An aiomysql connection pool for database operations
        table_name: Name of the table to store conversations (default: "conversations")
        chat_id_column: Name of the column for chat ID (default: "chat_id")
        message_data_column: Name of the column for message data (default: "message_data")
        timestamp_column: Name of the column for timestamp (default: "timestamp")
        id_column: Name of the primary key column (default: "id")
        message_limit: Maximum number of messages to keep per conversation
        override_old_messages: Whether to override old messages when limit is reached
    """
    
    def __init__(
        self,
        pool: Pool,
        table_name: str = "conversations",
        chat_id_column: str = "chat_id",
        message_data_column: str = "message_data",
        timestamp_column: str = "timestamp",
        id_column: str = "id",
        message_limit: int | None = None,
        override_old_messages: bool = True,
    ):
        super().__init__(message_limit, override_old_messages)
        self.pool = pool
        self.table_name = table_name
        self.chat_id_column = chat_id_column
        self.message_data_column = message_data_column
        self.timestamp_column = timestamp_column
        self.id_column = id_column
        
    async def _ensure_table_exists(self) -> None:
        """
        Ensure the conversations table exists, create it if it doesn't.
        """
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {self.id_column} INT AUTO_INCREMENT PRIMARY KEY,
            {self.chat_id_column} VARCHAR(255) NOT NULL,
            {self.message_data_column} JSON NOT NULL,
            {self.timestamp_column} TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_chat_id ({self.chat_id_column}),
            INDEX idx_timestamp ({self.timestamp_column})
        )
        """
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(create_table_query)
                await conn.commit()
                
    def _message_to_dict(self, message: DeveloperMessage | UserMessage | AssistantMessage) -> dict[str, Any]:
        """
        Convert a Message object to a dictionary for JSON serialization.
        
        Args:
            message: The Message object to convert
            
        Returns:
            A dictionary representation of the message
        """
        message_dict = message.model_dump()
        # Add message type for proper deserialization
        message_dict["_message_type"] = type(message).__name__
        return message_dict
        
    def _dict_to_message(self, message_dict: dict[str, Any]) -> DeveloperMessage | UserMessage | AssistantMessage:
        """
        Convert a dictionary back to a Message object.
        
        Args:
            message_dict: The dictionary representation of the message
            
        Returns:
            A Message object
        """
        message_type = message_dict.pop("_message_type", None)
        
        if message_type == "DeveloperMessage":
            return DeveloperMessage.model_validate(message_dict)
        elif message_type == "UserMessage":
            return UserMessage.model_validate(message_dict)
        elif message_type == "AssistantMessage":
            return AssistantMessage.model_validate(message_dict)
        else:
            # Fallback: try to determine type from role
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
    async def add_message_async(self, chat_id: str, message: DeveloperMessage | UserMessage | AssistantMessage) -> None:
        """
        Add a message to the conversation store.
        
        Args:
            chat_id: The unique identifier for the conversation
            message: The message to add
        """
        await self._ensure_table_exists()
        
        # Convert message to dictionary for JSON storage
        message_dict = self._message_to_dict(message)
        message_json = json.dumps(message_dict)
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Insert the new message
                insert_query = f"""
                INSERT INTO {self.table_name} 
                ({self.chat_id_column}, {self.message_data_column}) 
                VALUES (%s, %s)
                """
                await cursor.execute(insert_query, (chat_id, message_json))
                
                # Handle message limit if specified
                if self.message_limit is not None:
                    # Count current messages for this chat
                    count_query = f"""
                    SELECT COUNT(*) FROM {self.table_name} 
                    WHERE {self.chat_id_column} = %s
                    """
                    await cursor.execute(count_query, (chat_id,))
                    count_result = await cursor.fetchone()
                    message_count = count_result[0] if count_result else 0
                    
                    # If we exceed the limit, remove old messages
                    if message_count > self.message_limit:
                        if self.override_old_messages:
                            excess_count = message_count - self.message_limit
                            delete_query = f"""
                            DELETE FROM {self.table_name} 
                            WHERE {self.chat_id_column} = %s 
                            ORDER BY {self.timestamp_column} ASC 
                            LIMIT %s
                            """
                            await cursor.execute(delete_query, (chat_id, excess_count))
                        else:
                            # If not overriding, remove the message we just added
                            delete_query = f"""
                            DELETE FROM {self.table_name} 
                            WHERE {self.chat_id_column} = %s 
                            ORDER BY {self.timestamp_column} DESC 
                            LIMIT 1
                            """
                            await cursor.execute(delete_query, (chat_id,))
                            
                await conn.commit()
                
    @override
    async def get_conversation_history_async(
        self, chat_id: str
    ) -> Sequence[DeveloperMessage | UserMessage | AssistantMessage]:
        """
        Retrieve the conversation history for a given chat ID.
        
        Args:
            chat_id: The unique identifier for the conversation
            
        Returns:
            A list of Message objects representing the conversation history
        """
        await self._ensure_table_exists()
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                select_query = f"""
                SELECT {self.message_data_column} FROM {self.table_name} 
                WHERE {self.chat_id_column} = %s 
                ORDER BY {self.timestamp_column} ASC
                """
                await cursor.execute(select_query, (chat_id,))
                rows = await cursor.fetchall()
                
                messages = []
                for row in rows:
                    raw_data = row[0]
                    
                    # Handle both string and dict cases
                    if isinstance(raw_data, str):
                        message_dict: dict[str, Any] = json.loads(raw_data)
                    elif isinstance(raw_data, dict):
                        message_dict = raw_data
                    else:
                        # Skip invalid data
                        continue
                        
                    message = self._dict_to_message(message_dict)
                    messages.append(message)
                    
                return messages
                
    @override
    async def clear_conversation_async(self, chat_id: str) -> None:
        """
        Clear all messages for a given chat ID.
        
        Args:
            chat_id: The unique identifier for the conversation to clear
        """
        await self._ensure_table_exists()
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                delete_query = f"""
                DELETE FROM {self.table_name} 
                WHERE {self.chat_id_column} = %s
                """
                await cursor.execute(delete_query, (chat_id,))
                await conn.commit()
                
    async def get_all_chat_ids(self) -> list[str]:
        """
        Get all unique chat IDs from the store.
        
        Returns:
            A list of all unique chat IDs
        """
        await self._ensure_table_exists()
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                select_query = f"""
                SELECT DISTINCT {self.chat_id_column} FROM {self.table_name}
                """
                await cursor.execute(select_query)
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
                
    async def get_conversation_count(self, chat_id: str) -> int:
        """
        Get the number of messages in a conversation.
        
        Args:
            chat_id: The unique identifier for the conversation
            
        Returns:
            The number of messages in the conversation
        """
        await self._ensure_table_exists()
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                count_query = f"""
                SELECT COUNT(*) FROM {self.table_name} 
                WHERE {self.chat_id_column} = %s
                """
                await cursor.execute(count_query, (chat_id,))
                result = await cursor.fetchone()
                return result[0] if result else 0
                
    async def close(self) -> None:
        """
        Close the database connection pool.
        """
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()