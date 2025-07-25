"""
MCP Servers Integration Example (Synchronous Version)

This example demonstrates how to use the Agentle framework with Model Context Protocol (MCP) servers.
It uses synchronous code only and is structured as a simple script.

Note: This example assumes MCP servers are already running elsewhere. You'll need to
substitute the server URLs and commands with your actual server information.
"""

import logging
import os
import sys

from agentle.agents.agent import Agent

from agentle.mcp.servers.stdio_mcp_server import StdioMCPServer

# Configure logging to show debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

http_server = StdioMCPServer(
    server_name="Open Memory",
    command="npx -y openmemory",
    server_env={
        "OPENMEMORY_API_KEY": os.getenv("OPENMEMORY_API_KEY") or "",
        "CLIENT_NAME": os.getenv("CLIENT_NAME") or "",
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") or "",
    },
)

# Create agent with MCP servers
agent = Agent(
    instructions="You are a helpful assistant with access to external tools",
    mcp_servers=[http_server],
)

# Use ONLY the context manager for all MCP operations
with agent.start_mcp_servers():
    # Get the tools from the server
    print(f"\n🔧 Tools from {http_server.name}:")
    sse_tools = http_server.list_tools()
    for tool in sse_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Example query
    math_response = agent.run("What is 2+2?")

    print(math_response.pretty_formatted())
