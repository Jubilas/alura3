"""
Coração da aplicação, analisa a pergunta, direciona para o NO mais adequado
fallback para o outro NO se der erro e por fim gera  a resposta
"""

import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from src.state import EstadoAgente
from src.ingestion import (
    obter_dataframes_disponiveis,
    obter_esquema_tabelas,
    executar_codigo_pandas,
)

load_dotenv()

def obter_modelo_llm(
    chave_api: Optional[str] = None,
    modelo: Optional[str] = None,
    temperatura: float = 0.0,
):
    """Instancia e retorna o modelo de linguagem Groq (Llama 3.3 70B)."""
    from langchain_groq import ChatGroq

    api_key = chave_api or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("A Chave de API não está configurada (GROQ_API_KEY).")

    nome_modelo = modelo or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model_name=nome_modelo,
        groq_api_key=api_key,
        temperature=temperatura,
    )


def formatar_fontes_para_exibicao(documentos: List[Document]) -> List[Dict[str, Any]]:
    """
    Extrai metadados estruturados dos documentos recuperados para exibição na UI.

    Args:
        documentos: Lista de documentos retornados pelo retriever.

    Returns:
        Lista de dicionários com nome do arquivo, página/linha e trecho.
    """
    fontes_formatadas = []
    for doc in documentos:
        meta = doc.metadata or {}
        nome_arquivo = meta.get("arquivo") or meta.get("source", "Documento desconhecido")
        
        localizacao = ""
        if "pagina" in meta:
            localizacao = f"Página {meta['pagina']}"
        elif "linhas_intervalo" in meta:
            localizacao = f"Linhas {meta['linhas_intervalo']}"
        elif "linha" in meta:
            localizacao = f"Linha {meta['linha']}"
        elif meta.get("tipo") == "resumo_tabela":
            localizacao = "Resumo Estrutural"
        elif meta.get("tipo") == "estatisticas_tabela":
            localizacao = "Estatísticas e Extremos"

        trecho_preview = doc.page_content.strip()
        if len(trecho_preview) > 300:
            trecho_preview = trecho_preview[:300] + "..."

        fontes_formatadas.append({
            "arquivo": nome_arquivo,
            "localizacao": localizacao,
            "tipo": meta.get("tipo", "documento"),
            "trecho": trecho_preview,
            "conteudo_completo": doc.page_content,
        })
    return fontes_formatadas


def formatar_contexto_para_prompt(documentos: List[Document], limite_max_caracteres: int = 8000) -> str:
    """
    Formata os documentos recuperados em uma string única para o prompt do LLM.
    Aplica orçamento de caracteres para respeitar limites de taxa (TPM) como o do Groq (12k TPM).

    Args:
        documentos: Lista de documentos recuperados.
        limite_max_caracteres: Limite máximo acumulado de caracteres de contexto.

    Returns:
        Texto consolidado contendo contexto e identificação de fonte.
    """
    if not documentos:
        return "Nenhum documento ou contexto relevante foi encontrado na base de conhecimento."

    blocos_contexto = []
    total_caracteres = 0

    for i, doc in enumerate(documentos, start=1):
        meta = doc.metadata or {}
        fonte = meta.get("arquivo") or meta.get("source", f"Documento #{i}")
        detalhe = ""
        if "pagina" in meta:
            detalhe = f" (Pág. {meta['pagina']})"
        elif "linhas_intervalo" in meta:
            detalhe = f" (Linhas {meta['linhas_intervalo']})"
        elif "linha" in meta:
            detalhe = f" (Linha {meta['linha']})"
        elif meta.get("tipo") == "resumo_tabela":
            detalhe = " (Resumo Geral)"
        elif meta.get("tipo") == "estatisticas_tabela":
            detalhe = " (Estatísticas e Extremos)"

        conteudo = doc.page_content.strip()
        espaco_restante = limite_max_caracteres - total_caracteres

        if espaco_restante <= 150:
            break

        if len(conteudo) > espaco_restante:
            conteudo = conteudo[:espaco_restante] + "\n[... trecho truncado para respeitar limite de tokens ...]"

        bloco = f"--- [DOCUMENTO {i}: {fonte}{detalhe}] ---\n{conteudo}\n"
        blocos_contexto.append(bloco)
        total_caracteres += len(bloco)

    return "\n".join(blocos_contexto)


def no_rotear_intencao(
    estado: EstadoAgente,
    llm,
    diretorio_dados: str = "./data_files",
) -> Dict[str, Any]:
    """
    Nó do LangGraph: Analisa a pergunta e o esquema das tabelas carregadas para
    decidir se deve direcionar para o motor analítico Pandas ou para a busca vetorial em textos.

    Args:
        estado: Estado atual com a pergunta do usuário.
        llm: Modelo LLM.
        diretorio_dados: Pasta de arquivos de dados salvos.

    Returns:
        Atualização do campo 'tipo_consulta'.
    """
    pergunta = estado.get("pergunta", "").strip()
    dfs = obter_dataframes_disponiveis(diretorio_dados=diretorio_dados)

    # (importante) fallback caso não haja tabelas ele da um fallback para a busca vetorial
    if not dfs:
        return {"tipo_consulta": "vetorial"}

    esquema = obter_esquema_tabelas(diretorio_dados=diretorio_dados)

    prompt_roteador = f"""Você é o módulo de decisão e roteamento de consultas de uma plataforma corporativa de IA.

ESQUEMA DAS TABELAS DISPONÍVEIS E SUAS COLUNAS:
{esquema}

PERGUNTA DO USUÁRIO: '{pergunta}'

REGRAS DE CLASSIFICAÇÃO:
1. Responda 'TABELA' SOMENTE se a pergunta exigir cálculos matemáticos, contagens ou filtros sobre as COLUNAS EXISTENTES nas tabelas listadas acima (ex: maior/menor classificacao_agente, média de tempo_entrega, ID_pedido com nota máxima, contagem de pedidos por clima/veículo/tráfego).
2. Responda 'VETORIAL' se a pergunta for sobre:
   - Políticas de frete grátis, diretrizes, manuais de envio, regras de devolução, prazos, termos de garantia;
   - Regras comerciais por estados ou regiões que não existam como colunas nas tabelas acima;
   - Qualquer conteúdo conceitual, descritivo ou de arquivos de texto (.txt, .pdf).

Responda ESTRITAMENTE com uma única palavra: TABELA ou VETORIAL"""

    try:
        resposta_rot = llm.invoke([HumanMessage(content=prompt_roteador)])
        conteudo_rot = str(resposta_rot.content).strip().upper()
        if "TABELA" in conteudo_rot and "VETORIAL" not in conteudo_rot:
            return {"tipo_consulta": "tabela"}
        else:
            return {"tipo_consulta": "vetorial"}
    except Exception:
        return {"tipo_consulta": "vetorial"}


def no_analisar_tabela(
    estado: EstadoAgente,
    llm,
    banco_vetorial,
    top_k: int = 4,
    diretorio_dados: str = "./data_files",
) -> Dict[str, Any]:
    pergunta = estado.get("pergunta", "")
    esquema = obter_esquema_tabelas(diretorio_dados=diretorio_dados)

    prompt_codigo = f"""Você é um especialista em análise de dados com Python e Pandas.
Temos os seguintes dataframes carregados na variável `df` (e no dicionário `dfs`):
{esquema}

Pergunta do usuário: '{pergunta}'

Instruções estritas:
1. Verifique se as colunas necessárias para responder existem em `df`.
2. Se a pergunta for sobre dados ou políticas NÃO existentes nas colunas de `df` (ex: frete grátis por estado, regras textuais), gere `resultado = None`.
3. Se as colunas existirem, gere código Python/Pandas para calcular e filtrar a resposta exata nos dados reais de `df` e atribua a `resultado`.
4. Para perguntas de 'maior' ou 'menor', busque TODOS os registros que empatam no valor máximo/mínimo (ex: `resultado = df[df['classificacao_agente'] == df['classificacao_agente'].max()][['ID_pedido', 'classificacao_agente']]`).
5. Retorne APENAS o código Python dentro de um bloco ```python ... ``` sem explicações adicionais."""

    resultado_tabular = ""
    codigo_executado = ""
    arquivo_fonte = "dados.csv"
    sucesso_tabular = False

    try:
        resposta_cod = llm.invoke([HumanMessage(content=prompt_codigo)])
        texto_cod = str(resposta_cod.content)
        
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", texto_cod, re.DOTALL)
        codigo_python = match.group(1) if match else texto_cod.strip()

        exec_res = executar_codigo_pandas(codigo_python, diretorio_dados=diretorio_dados)

        if not exec_res.get("sucesso"):
            prompt_correcao = f"""O seguinte código Pandas gerou um erro:
```python
{codigo_python}
```
Mensagem de erro: {exec_res.get('erro')}
Esquema do DataFrame `df`:
{esquema}

Pergunta original: {pergunta}

Reescreva o código Python/Pandas corrigido atribuindo o resultado final à variável `resultado`.
Retorne APENAS o bloco ```python ... ``` sem texto adicional."""
            try:
                resposta_corr = llm.invoke([HumanMessage(content=prompt_correcao)])
                texto_corr = str(resposta_corr.content)
                match_corr = re.search(r"```(?:python)?\s*(.*?)\s*```", texto_corr, re.DOTALL)
                codigo_corr = match_corr.group(1) if match_corr else texto_corr.strip()
                exec_res_corr = executar_codigo_pandas(codigo_corr, diretorio_dados=diretorio_dados)
                if exec_res_corr.get("sucesso"):
                    exec_res = exec_res_corr
            except Exception:
                pass

        if exec_res.get("sucesso"):
            resultado_tabular = exec_res.get("resultado", "")
            codigo_executado = exec_res.get("codigo", "")
            arquivo_fonte = exec_res.get("arquivo", "dados.csv")
            
            # Valida se a resposta tabular trouxe dados concretos
            res_str = str(resultado_tabular).strip()
            if (
                "Total de registros encontrados: 0" not in res_str
                and "Empty DataFrame" not in res_str
                and res_str not in ["None", "", "nan"]
            ):
                sucesso_tabular = True

    except Exception as erro:
        resultado_tabular = f"Erro na análise: {str(erro)}"

    # Se a análise tabular não encontrou registros, realiza Fallback automático para o RAG Vetorial
    if not sucesso_tabular:
        try:
            documentos_recuperados = banco_vetorial.similarity_search(pergunta, k=top_k)
        except Exception:
            documentos_recuperados = []

        if documentos_recuperados:
            fontes = formatar_fontes_para_exibicao(documentos_recuperados)
            return {
                "tipo_consulta": "vetorial",
                "documentos": documentos_recuperados,
                "resultado_tabular": "",
                "codigo_executado": "",
                "fontes": fontes,
            }

    fontes = [{
        "arquivo": arquivo_fonte,
        "localizacao": "Motor Analítico Pandas (Base Completa)",
        "tipo": "consulta_tabular",
        "trecho": f"Código executado:\n{codigo_executado}\n\nResultado:\n{resultado_tabular[:350]}",
        "conteudo_completo": resultado_tabular,
    }]

    return {
        "tipo_consulta": "tabela",
        "resultado_tabular": resultado_tabular,
        "codigo_executado": codigo_executado,
        "fontes": fontes,
    }


def no_recuperar_documentos(estado: EstadoAgente, banco_vetorial, top_k: int = 4) -> Dict[str, Any]:
    pergunta = estado.get("pergunta", "")
    
    if not pergunta:
        return {"documentos": [], "fontes": []}

    try:
        documentos_recuperados = banco_vetorial.similarity_search(pergunta, k=top_k)
    except Exception as erro:
        print(f"Erro durante a recuperação dos documentos: {erro}")
        documentos_recuperados = []

    fontes = formatar_fontes_para_exibicao(documentos_recuperados)

    return {
        "documentos": documentos_recuperados,
        "fontes": fontes,
    }


def no_gerar_resposta(estado: EstadoAgente, llm) -> Dict[str, Any]:
    pergunta = estado.get("pergunta", "")
    tipo_consulta = estado.get("tipo_consulta", "vetorial")
    historico = estado.get("historico_conversa", [])

    if tipo_consulta == "tabela" and estado.get("resultado_tabular"):
        resultado_tabular = estado.get("resultado_tabular", "")
        codigo_executado = estado.get("codigo_executado", "")

        instrucoes_sistema = f"""Você é o assistente corporativo 'Oracle OCI Agent', especialista em análise de dados e inteligência documental.

Diretrizes de resposta:
1. Responda à pergunta do usuário de forma clara, direta, executiva e minimalista em Português (PT-BR).
2. NÃO utilize emojis ou emoticons nas respostas.
3. Baseie sua resposta nos RESULTADOS ANALÍTICOS EXATOS calculados diretamente na base de dados abaixo.
4. Liste TODOS os IDs e registros relevantes que atendam ao critério da pergunta (sem omitir dados).
5. Ao final, cite a fonte de forma sóbria (ex: [Motor Analítico Pandas, dados_entregas.csv]).

--- RESULTADO EXATO CALCULADO NA BASE COMPLETA ---
{resultado_tabular}
Código de consulta executado:
{codigo_executado}
--------------------------------------------------"""
    else:
        documentos = estado.get("documentos", [])
        contexto_formatado = formatar_contexto_para_prompt(documentos)

        instrucoes_sistema = f"""Você é o assistente corporativo 'Oracle OCI Agent', especialista em análise de documentos corporativos, políticas internas, manuais e guias operacionais.

Diretrizes de redação:
1. Responda à pergunta de forma direta, executiva, elegante e minimalista em Português (PT-BR).
2. NÃO use emojis ou emoticons em nenhuma hipótese. Mantenha um estilo puramente profissional e textual.
3. Baseie sua resposta estritamente no CONTEXTO fornecido abaixo.
4. Indique as fontes consultadas de forma sóbria (ex: [Guia de Envios e Entregas.txt, Seção X] ou [Arquivo.pdf, Pág. Y]).
5. Se a resposta não constar no contexto, informe objetivamente que a informação não consta na base indexada.

--- CONTEXTO RECUPERADO DOS DOCUMENTOS ---
{contexto_formatado}
------------------------------------------"""

    mensagens = [SystemMessage(content=instrucoes_sistema)]

    # Adiciona histórico de conversa se existir
    if historico:
        for msg in historico[-6:]:
            if isinstance(msg, dict):
                papel = msg.get("role", "")
                conteudo = msg.get("content", "")
                if papel == "user":
                    mensagens.append(HumanMessage(content=conteudo))
                elif papel == "assistant":
                    mensagens.append(AIMessage(content=conteudo))
            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                mensagens.append(msg)

    # Pergunta atual do usuário
    mensagens.append(HumanMessage(content=pergunta))

    try:
        resposta_llm = llm.invoke(mensagens)
        conteudo_resposta = resposta_llm.content if hasattr(resposta_llm, "content") else str(resposta_llm)
    except Exception as erro:
        msg_str = str(erro)
        if "413" in msg_str or "Request too large" in msg_str or "rate_limit_exceeded" in msg_str:
            try:
                instrucoes_reduzidas = f"""Você é o assistente corporativo 'Oracle OCI Agent'.
Responda à pergunta de forma direta, executiva e minimalista em Português (PT-BR) sem emojis.

Dados:
{str(estado.get('resultado_tabular', ''))[:2000]}"""
                mensagens_reduzidas = [SystemMessage(content=instrucoes_reduzidas), HumanMessage(content=pergunta)]
                resposta_llm = llm.invoke(mensagens_reduzidas)
                conteudo_resposta = resposta_llm.content if hasattr(resposta_llm, "content") else str(resposta_llm)
            except Exception as erro_secundario:
                conteudo_resposta = f"Erro de limite de tokens da API: {str(erro_secundario)}"
        else:
            conteudo_resposta = f"Erro no processamento com o modelo de IA: {msg_str}"

    return {"resposta": conteudo_resposta}


def decidir_proximo_no(estado: EstadoAgente) -> str:
    """
    Função condicional do LangGraph para roteamento de nós.
    """
    tipo = estado.get("tipo_consulta", "vetorial")
    if tipo == "tabela":
        return "analisar_tabela"
    return "recuperar"


def construir_grafo_rag(banco_vetorial, llm, top_k: int = 4, diretorio_dados: str = "./data_files"):
    """
    Constrói e compila o StateGraph híbrido do LangGraph com suporte a consultas tabulares e RAG.

    Fluxo:
    START -> rotear -> (analisar_tabela | recuperar) -> gerar -> END
    """
    fluxo = StateGraph(EstadoAgente)

    def no_rotear_encapsulado(estado: EstadoAgente):
        return no_rotear_intencao(estado=estado, llm=llm, diretorio_dados=diretorio_dados)

    def no_analisar_tabela_encapsulado(estado: EstadoAgente):
        return no_analisar_tabela(
            estado=estado,
            llm=llm,
            banco_vetorial=banco_vetorial,
            top_k=top_k,
            diretorio_dados=diretorio_dados,
        )

    def no_recuperar_encapsulado(estado: EstadoAgente):
        return no_recuperar_documentos(estado=estado, banco_vetorial=banco_vetorial, top_k=top_k)

    def no_gerar_encapsulado(estado: EstadoAgente):
        return no_gerar_resposta(estado=estado, llm=llm)

    fluxo.add_node("rotear", no_rotear_encapsulado)
    fluxo.add_node("analisar_tabela", no_analisar_tabela_encapsulado)
    fluxo.add_node("recuperar", no_recuperar_encapsulado)
    fluxo.add_node("gerar", no_gerar_encapsulado)

    # Configuração de arestas condicionais e lineares
    fluxo.add_edge(START, "rotear")
    fluxo.add_conditional_edges(
        "rotear",
        decidir_proximo_no,
        {
            "analisar_tabela": "analisar_tabela",
            "recuperar": "recuperar",
        },
    )
    fluxo.add_edge("analisar_tabela", "gerar")
    fluxo.add_edge("recuperar", "gerar")
    fluxo.add_edge("gerar", END)

    grafo_compilado = fluxo.compile()
    return grafo_compilado


def executar_fluxo_rag(
    grafo,
    pergunta: str,
    historico_conversa: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Executa a invocação do grafo RAG/Tabular com a pergunta e histórico.
    """
    estado_inicial: EstadoAgente = {
        "pergunta": pergunta,
        "historico_conversa": historico_conversa or [],
        "documentos": [],
        "tipo_consulta": "vetorial",
        "resultado_tabular": "",
        "codigo_executado": "",
        "resposta": "",
        "fontes": [],
    }

    resultado_final = grafo.invoke(estado_inicial)
    return resultado_final
