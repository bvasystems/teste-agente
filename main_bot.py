import logging
import os
import uvicorn
from blacksheep import Application
from dotenv import load_dotenv

from agentle.agents.agent import Agent
from agentle.agents.conversations.json_file_conversation_store import JSONFileConversationStore
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.models.whatsapp_session import WhatsAppSession
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import EvolutionAPIConfig
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import EvolutionAPIProvider
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
from agentle.sessions.in_memory_session_store import InMemorySessionStore
from agentle.sessions.session_manager import SessionManager
from calendario_vacinas_tool import calendarioVacinas

# Carrega variáveis de ambiente
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)

# Lê o prompt a ser passado para os agentes (em um app real você pode ter prompts diferentes para cada um)
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.md")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SOFIA_INSTRUCTIONS = f.read()

def criar_bot(nome_agente: str, instancia_evolution: str, instructions: str, evolution_api_key: str) -> Application:
    """Função utilitária para criar um bot e devolver a aplicação dele isolada."""
    evolution_url = os.getenv("EVOLUTION_API_URL", "https://evolution.axobot.pro")

    from agentle.generations.providers.openai.openai import OpenaiGenerationProvider
    
    agent = Agent(
        name=nome_agente,
        instructions=instructions,
        tools=[calendarioVacinas], # Se as ferramentas forem diferentes, basta mudar aqui
        # Cada agente manterá suas próprias conversas na estrutura local
        conversation_store=JSONFileConversationStore(),
        generation_provider=OpenaiGenerationProvider(),
    )

    session_manager = SessionManager[WhatsAppSession](
        session_store=InMemorySessionStore[WhatsAppSession](),
        default_ttl_seconds=3600,
        enable_events=True,
    )

    provider = EvolutionAPIProvider(
        config=EvolutionAPIConfig(
            base_url=evolution_url,
            instance_name=instancia_evolution,
            api_key=evolution_api_key,
        ),
        session_manager=session_manager,
    )

    bot_config = WhatsAppBotConfig.production(
        welcome_message=f"Olá! Sou a IA da equipe {nome_agente}. Em que posso ajudar?", 
        quote_messages=False,
    )

    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider, config=bot_config)
    whatsapp_bot.start()

    from blacksheep.server.routing import Router
    
    # O webhook padrão desse mini-app será "/webhook"
    return whatsapp_bot.to_blacksheep_app(router=Router(), webhook_path="/webhook", show_error_details=True)

# ---------------------------------------------------------
# APLICAÇÃO PRINCIPAL (ROTEADOR DE BOTS)
# ---------------------------------------------------------
main_app = Application()

# Puxa as keys do .env ou usa uma default
api_key_padrao = os.getenv("EVOLUTION_API_KEY", "6242b406242c15bbb180")

# 1. Bot Comercial (Cliente 1 / Instância 1)
bot_comercial = criar_bot(
    nome_agente="SofIA Comercial", 
    instancia_evolution="BvaComercial",
    instructions=SOFIA_INSTRUCTIONS,
    evolution_api_key=api_key_padrao
)
# Monta a rota: http://SEU_IP:8000/cliente1/webhook
main_app.mount("/cliente1", bot_comercial)

# 2. Bot Suporte (Cliente 2 / Instância 2)
# Exemplo de um segundo cliente no mesmo servidor, usando outro prompt
bot_suporte = criar_bot(
    nome_agente="Assistente Suporte", 
    instancia_evolution="ClienteSuporteNode", # o nome exato da instancia na Evolution
    instructions="Você é o suporte técnico do Cliente 2. Seja breve e objetivo.",
    evolution_api_key=api_key_padrao # Aqui poderia ser a apikey específica dessa Evolution
)
# Monta a rota: http://SEU_IP:8000/cliente2/webhook
main_app.mount("/cliente2", bot_suporte)

# 3. Você pode criar quantos quiser e ir dando 'mount' aqui...

port = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    logging.info(f"Starting Multi-Bot server on port {port}")
    logging.info("Rotas ativas de Webhook:")
    logging.info(f" - Comercial: http://localhost:{port}/cliente1/webhook")
    logging.info(f" - Suporte:   http://localhost:{port}/cliente2/webhook")
    
    uvicorn.run(main_app, host="0.0.0.0", port=port)
