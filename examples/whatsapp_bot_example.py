# examples/whatsapp_bot_example.py
"""
Example of using Agentle agents as WhatsApp bots with simplified configuration.
"""

import logging
import os

import uvicorn
from blacksheep import Application
from dotenv import load_dotenv

from agentle.agents.agent import Agent
from agentle.agents.conversations.json_file_conversation_store import (
    JSONFileConversationStore,
)
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import (
    EvolutionAPIConfig,
)
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import (
    EvolutionAPIProvider,
)
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)


def create_bot() -> Application:
    """Example 2: Production bot with optimized configuration."""

    agent = Agent(
        instructions="Você é um assistente profissional. Seja útil, cortês e eficiente. Responda todas as perguntas.",
        conversation_store=JSONFileConversationStore(),
    )

    session_manager = SessionManager[WhatsAppSession](
        session_store=InMemorySessionStore[WhatsAppSession](),
        default_ttl_seconds=3600,
        enable_events=True,
    )

    # Create provider with session management
    provider = EvolutionAPIProvider(
        config=EvolutionAPIConfig(
            base_url=os.getenv("EVOLUTION_API_URL", "http://localhost:8080"),
            instance_name=os.getenv("EVOLUTION_INSTANCE_NAME", "production-bot"),
            api_key=os.getenv("EVOLUTION_API_KEY", "your-api-key"),
        ),
        session_manager=session_manager,
        session_ttl_seconds=3600,
    )

    # Use production configuration preset
    bot_config = WhatsAppBotConfig.production(
        welcome_message="Teste",
        quote_messages=False,  # Don't quote by default
        enable_spam_protection=True,
    )

    # Validate configuration
    issues = bot_config.validate_config()
    if issues:
        logging.warning(f"Production configuration issues: {issues}")

    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider, config=bot_config)
    whatsapp_bot.start()
    # Send a test message to the correct number
    # run_sync(whatsapp_bot.send_message(
    #     to="553491107754",
    #     message="Testando"
    # ))
    # exit()

    return whatsapp_bot.to_blacksheep_app(
        webhook_path="/webhook/whatsapp2",
        show_error_details=False,
    )


app = create_bot()
port = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    # Log which bot type is being used
    bot_type = os.getenv("BOT_TYPE", "development")
    logging.info(
        f"Starting WhatsApp bot server with '{bot_type}' configuration on port {port}"
    )

    uvicorn.run(app, host="0.0.0.0", port=port)
