# Gerenciamento de Agentes (Evolution API + Agentle)

O fluxo de conexão para cada novo cliente funciona assim:

1. **Evolution API**: Crie a instância nova (ex: `ClienteNovoNode`). É na Evolution API que você vai ler o QR Code do WhatsApp do cliente.
2. **Agentle (Código)**: 
   - Crie a pasta do cliente (`clientes/cliente_novo/`).
   - Coloque o `prompt.md` dele lá dentro.
   - Adicione 3 linhas no `main_bot.py` apontando para o nome da instância que você criou na Evolution (`ClienteNovoNode`).
3. **Deploy (Portainer)**: Dê um "Git Pull" (ou reinicie a stack do Portainer caso use volumes/github) para o Docker subir as novas configurações.
4. **Webhook**: Volte na Evolution API e aponte o webhook para `http://SEU_IP:8000/cliente_novo/webhook`.
