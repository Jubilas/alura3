"""
Aplicação Principal em Streamlit - Oracle OCI Agent (RAG com LangChain e LangGraph).
Interface interativa com chat, gestão de documentos (PDF/CSV) e visualização de fontes.
Design sóbrio, minimalista e corporativo sem emojis.
Todas as funções, comentários e textos em Português (PT-BR).
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Importação dos módulos do projeto
from src.state import EstadoAgente
from src.ingestion import (
    processar_e_indexar_arquivos,
    obter_banco_vetorial,
    limpar_banco_vetorial,
    obter_estatisticas_banco,
)
from src.graph import (
    obter_modelo_llm,
    construir_grafo_rag,
    executar_fluxo_rag,
)

# Carrega variáveis de ambiente
load_dotenv()


def configurar_pagina() -> None:
    """
    Configura os metadados e layout da página no Streamlit.
    """
    st.set_page_config(
        page_title="Oracle OCI Agent | Análise Documental",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def aplicar_estilos_customizados() -> None:
    """
    Injeta regras de CSS para conferir visual sóbrio, minimalista e corporativo à interface.
    """
    st.markdown(
        """
        <style>
        /* Tipografia e cabeçalhos */
        .main-header {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
            border-bottom: 2px solid #C74634;
            padding-bottom: 8px;
            display: inline-block;
        }
        .sub-header {
            font-size: 0.95rem;
            opacity: 0.8;
            margin-top: 0.4rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: rgba(125, 125, 125, 0.08);
            border: 1px solid rgba(125, 125, 125, 0.2);
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 10px;
            font-size: 0.88rem;
        }
        .chunk-box {
            background-color: rgba(125, 125, 125, 0.06);
            border-left: 3px solid #C74634;
            padding: 10px 14px;
            border-radius: 0 4px 4px 0;
            margin-top: 6px;
            font-size: 0.86rem;
            font-family: inherit;
        }
        /* Borda sutil na sidebar sem forçar cor de fundo conflitante */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(125, 125, 125, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inicializar_estado_sessao() -> None:
    """
    Garante a inicialização correta de todas as variáveis no st.session_state.
    """
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    if "provedor_ia" not in st.session_state:
        st.session_state.provedor_ia = os.getenv("LLM_PROVIDER", "groq").lower()

    if "top_k" not in st.session_state:
        st.session_state.top_k = int(os.getenv("TOP_K", "4"))

    if "temperatura" not in st.session_state:
        st.session_state.temperatura = float(os.getenv("TEMPERATURE", "0.2"))

    if "diretorio_db" not in st.session_state:
        st.session_state.diretorio_db = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    if "diretorio_dados" not in st.session_state:
        st.session_state.diretorio_dados = os.getenv("DATA_FILES_DIR", "./data_files")

    if "ultima_atualizacao_db" not in st.session_state:
        st.session_state.ultima_atualizacao_db = time.time()


def renderizar_barra_lateral() -> Dict[str, Any]:
    """
    Renderiza os controles de upload, estatísticas e configurações na barra lateral.

    Returns:
        Dicionário com as configurações ativas selecionadas pelo usuário.
    """
    with st.sidebar:
        st.markdown("### Configurações do Sistema")
        st.caption("Oracle OCI Document Intelligence Engine")
        st.divider()

        # Configuração do Modelo Groq
        st.markdown("**Modelo de Linguagem:**")
        st.info("Groq — Llama 3.3 70B Versatile")

        chave_env = os.getenv("GROQ_API_KEY", "")
        placeholder_texto = "Chave configurada via .env" if chave_env else "gsk_..."
        chave_input = st.text_input(
            "Chave Groq API:",
            type="password",
            placeholder=placeholder_texto,
            help="Chave de API obtida no console Groq (https://console.groq.com).",
        )
        chave_api_informada = chave_input.strip() if chave_input.strip() else (chave_env or None)
        st.caption("Embeddings locais rápidos (FastEmbed) ativos sem custos de API.")

        with st.expander("Parâmetros de Recuperação e Modelo", expanded=False):
            top_k = st.slider(
                "Documentos recuperados (Top K):",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                help="Quantidade de trechos contextuais enviados ao modelo.",
            )
            st.session_state.top_k = top_k

            temperatura = st.slider(
                "Temperatura:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperatura,
                step=0.05,
                help="0.0 para respostas factuais e objetivas.",
            )
            st.session_state.temperatura = temperatura

        st.divider()

        # Ingestão de Documentos
        st.markdown("### Base de Conhecimento")
        arquivos_enviados = st.file_uploader(
            "Upload de arquivos:",
            type=["pdf", "csv", "xlsx", "xls", "txt", "md"],
            accept_multiple_files=True,
            help="Formatos suportados: PDF, CSV, Excel e Texto.",
        )

        col_proc, col_limp = st.columns(2)
        
        with col_proc:
            if st.button("Indexar Documentos", use_container_width=True, type="primary"):
                if not arquivos_enviados:
                    st.warning("Selecione ao menos um arquivo para indexação.")
                else:
                    with st.spinner("Processando e indexando documentos..."):
                        resultado = processar_e_indexar_arquivos(
                            arquivos=arquivos_enviados,
                            diretorio_persistencia=st.session_state.diretorio_db,
                        )
                        if resultado.get("sucesso"):
                            st.success(f"Concluído: {resultado['total_chunks']} chunks indexados.")
                            st.session_state.ultima_atualizacao_db = time.time()
                        else:
                            st.error(resultado.get("mensagem", "Falha na indexação."))

        with col_limp:
            if st.button("Limpar Base", use_container_width=True):
                limpar_banco_vetorial(st.session_state.diretorio_db, st.session_state.diretorio_dados)
                st.session_state.ultima_atualizacao_db = time.time()
                st.info("Base vetorial limpa com sucesso.")

        # Estatísticas do Banco Vetorial
        stats = {"total_chunks": 0, "fontes": []}
        try:
            stats = obter_estatisticas_banco(
                diretorio_persistencia=st.session_state.diretorio_db,
                diretorio_dados=st.session_state.diretorio_dados,
            )
        except Exception:
            stats = {"total_chunks": 0, "fontes": []}

        st.markdown("#### Estatísticas da Base")
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Total de Chunks Indexados:</b> {stats.get('total_chunks', 0)}<br>
                <b>Documentos Registrados:</b> {len(stats.get('fontes', []))}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if stats.get("fontes"):
            with st.expander("Arquivos indexados", expanded=False):
                for f in stats["fontes"]:
                    st.write(f"- `{f}`")

        st.divider()

        if st.button("Nova Conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.rerun()

        return {
            "chave_api": chave_api_informada,
            "top_k": st.session_state.top_k,
            "temperatura": st.session_state.temperatura,
            "diretorio_db": st.session_state.diretorio_db,
            "diretorio_dados": st.session_state.diretorio_dados,
            "total_chunks": stats.get("total_chunks", 0),
        }


def renderizar_historico_chat() -> None:
    """
    Exibe o histórico de mensagens e os blocos de fontes consultadas de forma limpa.
    """
    for msg in st.session_state.mensagens:
        papel = msg.get("papel", "user")
        conteudo = msg.get("conteudo", "")
        fontes = msg.get("fontes", [])
        
        with st.chat_message(papel):
            st.markdown(conteudo)
            
            # Exibe fontes consultadas de forma estruturada e sóbria
            if papel == "assistant" and fontes:
                with st.expander(f"Fontes Consultadas ({len(fontes)})", expanded=False):
                    for idx, fonte in enumerate(fontes, start=1):
                        nome_arq = fonte.get("arquivo", "Documento")
                        local = fonte.get("localizacao", "")
                        trecho = fonte.get("trecho", "")
                        
                        detalhe_local = f" • {local}" if local else ""
                        st.markdown(f"**Fonte {idx}:** `{nome_arq}`{detalhe_local}")
                        st.markdown(f'<div class="chunk-box">{trecho}</div>', unsafe_allow_html=True)
                        st.markdown("")


def processar_envio_mensagem(
    pergunta: str,
    configuracoes: Dict[str, Any],
) -> None:
    """
    Executa o ciclo completo de resposta do agente via LangGraph e atualiza a UI.

    Args:
        pergunta: Pergunta digitada pelo usuário.
        configuracoes: Dicionário contendo provedor, chaves de API e parâmetros.
    """
    if not pergunta.strip():
        return

    # Registra a pergunta do usuário
    st.session_state.mensagens.append({
        "papel": "user",
        "conteudo": pergunta,
        "timestamp": datetime.now().isoformat(),
    })

    with st.chat_message("user"):
        st.markdown(pergunta)

    # Validação de credenciais
    chave_api = configuracoes.get("chave_api")
    if not chave_api and configuracoes.get("provedor") != "groq":
        with st.chat_message("assistant"):
            msg_erro = "Chave de API não configurada. Forneça a credencial no arquivo .env ou na barra lateral."
            st.warning(msg_erro)
            st.session_state.mensagens.append({
                "papel": "assistant",
                "conteudo": msg_erro,
                "fontes": [],
            })
        return

    # Executa o fluxo híbrido (RAG + Pandas Engine) via LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Consultando dados e gerando resposta..."):
            try:
                # 1. Instancia o banco vetorial ChromaDB com FastEmbed
                banco_vetorial = obter_banco_vetorial(
                    diretorio_persistencia=configuracoes["diretorio_db"],
                )

                # 2. Instancia o LLM Groq (Llama 3.3 70B)
                llm = obter_modelo_llm(
                    chave_api=chave_api,
                    temperatura=configuracoes["temperatura"],
                )

                # 3. Constrói o grafo híbrido LangGraph (RAG + Pandas Engine)
                grafo_rag = construir_grafo_rag(
                    banco_vetorial=banco_vetorial,
                    llm=llm,
                    top_k=configuracoes["top_k"],
                    diretorio_dados=configuracoes.get("diretorio_dados", "./data_files"),
                )

                # Prepara o histórico recente
                historico_formatado = []
                for m in st.session_state.mensagens[:-1]:
                    role = "user" if m.get("papel") == "user" else "assistant"
                    historico_formatado.append({"role": role, "content": m.get("conteudo", "")})

                # 4. Executa o fluxo
                resultado = executar_fluxo_rag(
                    grafo=grafo_rag,
                    pergunta=pergunta,
                    historico_conversa=historico_formatado,
                )

                resposta_texto = resultado.get("resposta", "Não foi possível obter uma resposta para a consulta.")
                fontes_consultadas = resultado.get("fontes", [])

                # Renderiza a resposta
                st.markdown(resposta_texto)

                # Renderiza fontes consultadas
                if fontes_consultadas:
                    with st.expander(f"Fontes Consultadas ({len(fontes_consultadas)})", expanded=False):
                        for idx, fonte in enumerate(fontes_consultadas, start=1):
                            nome_arq = fonte.get("arquivo", "Documento")
                            local = fonte.get("localizacao", "")
                            trecho = fonte.get("trecho", "")
                            detalhe_local = f" • {local}" if local else ""
                            st.markdown(f"**Fonte {idx}:** `{nome_arq}`{detalhe_local}")
                            st.markdown(f'<div class="chunk-box">{trecho}</div>', unsafe_allow_html=True)
                            st.markdown("")

                # Salva no histórico da sessão
                st.session_state.mensagens.append({
                    "papel": "assistant",
                    "conteudo": resposta_texto,
                    "fontes": fontes_consultadas,
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as erro:
                msg_erro = f"Falha no processamento da consulta: {str(erro)}"
                st.error(msg_erro)
                st.session_state.mensagens.append({
                    "papel": "assistant",
                    "conteudo": msg_erro,
                    "fontes": [],
                })


def renderizar_tela_boas_vindas(configuracoes: Dict[str, Any]) -> None:
    """
    Exibe painel introdutório minimalista e sugestões de consulta.
    """
    total_chunks = configuracoes.get("total_chunks", 0)
    
    if total_chunks > 0:
        status_banco_html = f"""
        <div style="margin-top: 14px; padding: 10px 14px; background-color: rgba(16, 185, 129, 0.12); border-left: 3px solid #10b981; border-radius: 4px; font-size: 0.9rem;">
            <b>Base de Conhecimento Ativa:</b> {total_chunks} chunks indexados e persistidos no ChromaDB prontos para consulta.
        </div>
        """
    else:
        status_banco_html = """
        <div style="margin-top: 14px; padding: 10px 14px; background-color: rgba(125, 125, 125, 0.08); border-left: 3px solid rgba(125, 125, 125, 0.4); border-radius: 4px; font-size: 0.9rem;">
            Nenhum documento indexado. Envie arquivos PDF ou CSV através da barra lateral para iniciar.
        </div>
        """

    st.markdown(
        f"""
        <div style="background-color: rgba(125, 125, 125, 0.05); border: 1px solid rgba(125, 125, 125, 0.2); border-radius: 8px; padding: 22px; margin-bottom: 22px;">
            <h3 style="margin-top: 0; font-weight: 600; font-size: 1.25rem;">Oracle OCI Agent — Sistema de Inteligência Documental</h3>
            <p style="margin-bottom: 0; font-size: 0.95rem; line-height: 1.5; opacity: 0.85;">
                Plataforma de recuperação e análise contextualizada de dados corporativos fundamentada em LangChain, LangGraph e ChromaDB.
            </p>
            {status_banco_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Sugestões de Consulta:")
    col1, col2 = st.columns(2)

    perguntas_sugestoes = [
        "Apresente um resumo estruturado dos tópicos abordados nos documentos.",
        "Quais são os principais números, métricas ou valores quantitativos presentes?",
        "Quais diretrizes, requisitos técnicos ou procedimentos foram definidos?",
        "Quais conclusões ou encaminhamentos constam nos relatórios?",
    ]

    for i, sugestao in enumerate(perguntas_sugestoes):
        coluna = col1 if i % 2 == 0 else col2
        with coluna:
            if st.button(sugestao, key=f"sugestao_{i}", use_container_width=True):
                processar_envio_mensagem(sugestao, configuracoes)
                st.rerun()


def principal() -> None:
    """
    Função principal de inicialização e execução do Streamlit.
    """
    configurar_pagina()
    aplicar_estilos_customizados()
    inicializar_estado_sessao()

    # Cabeçalho Principal
    st.markdown('<div class="main-header">Oracle OCI Agent — Inteligência Documental</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Assistente corporativo fundamentado em documentos com orquestração LangGraph</div>', unsafe_allow_html=True)

    # Renderiza Barra Lateral e obtém configurações
    configuracoes = renderizar_barra_lateral()

    # Se o chat estiver vazio, exibe painel introdutório
    if not st.session_state.mensagens:
        renderizar_tela_boas_vindas(configuracoes)
    else:
        renderizar_historico_chat()

    # Campo de entrada de mensagem do usuário
    entrada_usuario = st.chat_input("Digite sua consulta sobre os documentos indexados...")
    if entrada_usuario:
        processar_envio_mensagem(entrada_usuario, configuracoes)
        st.rerun()


if __name__ == "__main__":
    principal()
