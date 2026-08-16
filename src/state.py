try:
    from typing import TypedDict, List, Dict, Any, Optional
except ImportError:
    from typing_extensions import TypedDict, List, Dict, Any, Optional

from langchain_core.documents import Document


class EstadoAgente(TypedDict):
    pergunta: str
    historico_conversa: Optional[List[Dict[str, str]]]
    documentos: Optional[List[Document]]
    tipo_consulta: Optional[str]
    resultado_tabular: Optional[str]
    codigo_executado: Optional[str]
    resposta: Optional[str]
    fontes: Optional[List[Dict[str, Any]]]
