# ==============================================================================
# Dockerfile - Oracle OCI Agent (RAG com LangChain, LangGraph e Streamlit)
# Otimizado para OCI Compute, OCI Container Instances e Kubernetes (OKE)
# ==============================================================================

FROM python:3.11-slim AS base

# Define variáveis de ambiente do Python e Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Instala pacotes do sistema necessários para compilação e healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root para execução segura
RUN useradd -m -u 1000 appuser

# Define diretório de trabalho
WORKDIR /app

# Instala dependências primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cria diretórios de persistência do ChromaDB e de dados com permissões adequadas
RUN mkdir -p /app/chroma_db /app/data_files && chown -R appuser:appuser /app

# Copia o código da aplicação
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser app.py /app/
COPY --chown=appuser:appuser .env.example /app/
COPY --chown=appuser:appuser .streamlit/ /app/.streamlit/

# Alterna para o usuário não-root
USER appuser

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Healthcheck para monitoramento em clusters e instâncias da nuvem OCI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização
CMD ["streamlit", "run", "app.py"]
