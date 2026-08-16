"""
Facilita a importação e tambem a minha vida
"""

from src.state import EstadoAgente
from src.ingestion import (
    obter_banco_vetorial,
    processar_e_indexar_arquivos,
    limpar_banco_vetorial,
    obter_estatisticas_banco,
)
from src.graph import (
    obter_modelo_llm,
    construir_grafo_rag,
    executar_fluxo_rag,
)

__all__ = [
    "EstadoAgente",
    "obter_banco_vetorial",
    "processar_e_indexar_arquivos",
    "limpar_banco_vetorial",
    "obter_estatisticas_banco",
    "obter_modelo_llm",
    "construir_grafo_rag",
    "executar_fluxo_rag",
]
