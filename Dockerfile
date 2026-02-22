FROM python:3.13-slim

WORKDIR /app

# Instalar dependências de compilação essenciais
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar o uv (gerenciador de pacotes pip super rápido)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
ENV PATH="/usr/local/bin:$PATH"

# Copia os arquivos de configuração (pode ajustar isso dependendo de como você empacotar)
COPY pyproject.toml .
COPY README.md .

# Instalar as dependências do Agentle globalmente usando o UV
RUN pip install uvicorn blacksheep python-dotenv openai

# Copia tudo para dentro da imagem Docker
COPY . .

# Se o projeto precisar usar como pacote local
RUN pip install -e .

EXPOSE 8000

ENV PORT=8000
# Comando final que subirá o bot
CMD ["python", "main_bot.py"]
