# git clone https://github.com/mem0ai/mem0.git
# cd openmemory
# docker compose up -d

import logging
import os

from agentle.agents.agent import Agent
from agentle.agents.conversations.local_conversation_store import LocalConversationStore
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.mcp.servers.stdio_mcp_server import StdioMCPServer

logging.basicConfig(level=logging.DEBUG)

open_memory_server = StdioMCPServer(
    server_name="Open Memory",
    command="npx -y openmemory",
    server_env={
        "OPENMEMORY_API_KEY": os.getenv("OPENMEMORY_API_KEY") or "",
        "CLIENT_NAME": os.getenv("CLIENT_NAME") or "",
    },
)
conversation_store = LocalConversationStore()

agent = Agent(
    # mcp_servers=[open_memory_server],
    conversation_store=conversation_store
)

print("🤖 OpenMemory Agent started! Type 'quit' to exit.")
print("-" * 50)

with agent.start_mcp_servers():
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Create user message
            user_message = UserMessage(parts=[TextPart(text=user_input)])

            # Run the agent
            print("🤔 Agent is thinking...")
            result = agent.run(user_message, chat_id="example")

            # Get the text response
            response_text = result.text

            # Create assistant message from the response
            assistant_message = AssistantMessage(parts=[TextPart(text=response_text)])

            # Print the response
            print(f"🤖 Assistant: {response_text}")

            # Print context information
            print("\n📊 Context Info:")
            print(f"   - Steps taken: {len(result.context.steps)}")
            print(f"   - Context ID: {result.context.context_id}")
            print(f"   - Is suspended: {result.is_suspended}")

            # Print the assistant message object for verification
            print("\n📝 Assistant Message Object:")
            print(f"   - Role: {assistant_message.role}")
            print(f"   - Text content: {assistant_message.parts[0].text[:100]}...")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please make sure OpenMemory is running and accessible.")
