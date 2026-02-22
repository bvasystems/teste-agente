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
from consultar_planilha_tool import consultarPlanilhaVacinas

# Carrega variáveis de ambiente
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)

# Lê o prompt a ser passado para os agentes (em um app real você pode ter prompts diferentes para cada um)
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.md")
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SOFIA_INSTRUCTIONS = f.read()

from blacksheep import Application, FromJSON, Response, json
from agentle.agents.whatsapp.models.whatsapp_webhook_payload import WhatsAppWebhookPayload
from agentle.generations.providers.openai.openai import OpenaiGenerationProvider

# APLICAÇÃO PRINCIPAL BASE
main_app = Application()

def criar_e_registrar_bot(
    app: Application,
    nome_agente: str, 
    instancia_evolution: str, 
    instructions: str, 
    evolution_api_key: str,
    webhook_route: str
):
    evolution_url = os.getenv("EVOLUTION_API_URL", "https://evolution.axobot.pro")
    
    agent = Agent(
        name=nome_agente,
        instructions=instructions,
        tools=[consultarPlanilhaVacinas],
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
        quote_messages=False,
    )

    whatsapp_bot = WhatsAppBot(agent=agent, provider=provider, config=bot_config)
    whatsapp_bot.start()

    # Registrando a rota de forma exclusiva para o app pai, evitando conflito no gerador automático
    @app.router.post(webhook_route)
    async def webhook_handler(webhook_payload: FromJSON[WhatsAppWebhookPayload]) -> Response:
        try:
            payload_data: WhatsAppWebhookPayload = webhook_payload.value
            await whatsapp_bot.handle_webhook(payload_data)
            return json({"status": "success", "message": "Webhook processed"})
        except Exception as e:
            logging.error(f"Erro no webhook de {instancia_evolution}: {e}")
            return json({"status": "error", "message": str(e)}, status=500)

api_key_padrao = os.getenv("EVOLUTION_API_KEY", "6242b406242c15bbb180")

# 1. Bot Comercial (Cliente 1 / Instância 1)
criar_e_registrar_bot(
    app=main_app,
    nome_agente="SofIA Comercial", 
    instancia_evolution="BvaComercial",
    instructions=SOFIA_INSTRUCTIONS,
    evolution_api_key=api_key_padrao,
    webhook_route="/cliente1/webhook"
)

# 2. Bot Suporte (Cliente 2 / Instância 2)
criar_e_registrar_bot(
    app=main_app,
    nome_agente="Assistente Suporte", 
    instancia_evolution="ClienteSuporteNode",
    instructions="Você é o suporte técnico do Cliente 2. Seja breve e objetivo.",
    evolution_api_key=api_key_padrao,
    webhook_route="/cliente2/webhook"
)

port = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    logging.info(f"Starting Multi-Bot server on port {port}")
    logging.info("Rotas ativas de Webhook conectadas na Evolution:")
    logging.info(f" - Comercial: http://SEU_DOMINIO_OU_IP:{port}/cliente1/webhook")
    logging.info(f" - Suporte:   http://SEU_DOMINIO_OU_IP:{port}/cliente2/webhook")
    
    uvicorn.run(main_app, host="0.0.0.0", port=port)
