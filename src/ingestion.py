"""
É a parte de ingestão de
arquivos e conversão em vetores
(tudo local)
"""

import io
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import pypdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()


def obter_modelo_embeddings():
    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        modelo_local = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        return FastEmbedEmbeddings(model_name=modelo_local)
    except Exception:
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=384)


def obter_banco_vetorial(
    diretorio_persistencia: str = "./chroma_db",
    nome_colecao: str = "documentos_rag",
) -> Chroma:
    modelo_embeddings = obter_modelo_embeddings()
    os.makedirs(diretorio_persistencia, exist_ok=True)

    try:
        banco = Chroma(
            collection_name=nome_colecao,
            embedding_function=modelo_embeddings,
            persist_directory=diretorio_persistencia,
        )
        _ = banco._collection.count()
        return banco
    except Exception as erro:
        print(f"Inconsistência detectada no banco vetorial ({erro}). Recriando base limpa...")
        try:
            if os.path.exists(diretorio_persistencia):
                shutil.rmtree(diretorio_persistencia)
                os.makedirs(diretorio_persistencia, exist_ok=True)
        except Exception as erro_limpeza:
            print(f"Aviso na limpeza automática: {erro_limpeza}")

        banco_limpo = Chroma(
            collection_name=nome_colecao,
            embedding_function=modelo_embeddings,
            persist_directory=diretorio_persistencia,
        )
        try:
            _ = banco_limpo._collection.count()
        except Exception:
            pass
        return banco_limpo


def carregar_arquivo_pdf(
    fonte_arquivo: Union[str, bytes, io.BytesIO],
    nome_arquivo: str,
) -> List[Document]:
    """
    Lê e extrai texto de arquivo PDF utilizando pypdf com localização de páginas.
    """
    documentos: List[Document] = []

    if isinstance(fonte_arquivo, (bytes, bytearray)):
        leitor_pdf = pypdf.PdfReader(io.BytesIO(fonte_arquivo))
    elif isinstance(fonte_arquivo, io.BytesIO):
        leitor_pdf = pypdf.PdfReader(fonte_arquivo)
    else:
        leitor_pdf = pypdf.PdfReader(str(fonte_arquivo))

    total_paginas = len(leitor_pdf.pages)

    for indice_pagina, pagina in enumerate(leitor_pdf.pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        if texto:
            metadados = {
                "source": nome_arquivo,
                "arquivo": nome_arquivo,
                "pagina": indice_pagina,
                "total_paginas": total_paginas,
                "tipo": "pdf",
            }
            documentos.append(Document(page_content=texto, metadata=metadados))

    return documentos


def carregar_arquivo_csv(
    fonte_arquivo: Union[str, bytes, io.BytesIO],
    nome_arquivo: str,
) -> List[Document]:
    """
    Lê planilhas CSV/Excel, pré-calcula estatísticas e gera blocos compactos para RAG.
    """
    documentos: List[Document] = []
    extensao = Path(nome_arquivo).suffix.lower()
    if isinstance(fonte_arquivo, (bytes, bytearray)):
        buffer = io.BytesIO(fonte_arquivo)
    elif isinstance(fonte_arquivo, io.BytesIO):
        buffer = fonte_arquivo
    else:
        buffer = fonte_arquivo
    if extensao in [".xlsx", ".xls"]:
        df = pd.read_excel(buffer)
    else:
        try:
            df = pd.read_csv(buffer, sep=None, engine="python")
        except Exception:
            if hasattr(buffer, "seek"):
                buffer.seek(0)
            df = pd.read_csv(buffer, sep=",")
    df = df.fillna("N/A")
    total_linhas = len(df)
    colunas = [str(c) for c in df.columns]
    nomes_colunas_str = ", ".join(colunas)

    # Identifica a coluna chave de forma híbrida (termos universais + unicidade matemática + fallback)
    col_id = None
    for c in colunas:
        c_low = c.lower()
        if any(term in c_low for term in ["id", "cod", "num", "key", "nome", "name", "pedido", "agente"]):
            col_id = c
            break
    if not col_id:
        for c in colunas:
            if df[c].nunique() == total_linhas:
                col_id = c
                break
    if not col_id and colunas:
        col_id = colunas[0]

    bloco_estatisticas = [
        f"=== ESTATÍSTICAS ANALÍTICAS GLOBAIS E EXTREMOS: {nome_arquivo} ===",
        f"Total de Registros (Linhas): {total_linhas}",
        f"Total de Colunas: {len(colunas)} ({nomes_colunas_str})",
    ]

    colunas_numericas_encontradas = []
    for col in colunas:
        serie_num = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        if serie_num.notna().sum() > (total_linhas * 0.3):
            colunas_numericas_encontradas.append(col)
            val_max = serie_num.max()
            val_min = serie_num.min()
            val_med = serie_num.mean()

            idx_maiores = serie_num.nlargest(3).index
            detalhes_maiores = [
                f"{col_id} {df.loc[idx_m, col_id]} ({col}: {df.loc[idx_m, col]})"
                for idx_m in idx_maiores
            ]

            idx_menores = serie_num.nsmallest(3).index
            detalhes_menores = [
                f"{col_id} {df.loc[idx_m, col_id]} ({col}: {df.loc[idx_m, col]})"
                for idx_m in idx_menores
            ]

            bloco_estatisticas.append(
                f"\n• Coluna '{col}':\n"
                f"  - Maior Valor (Máximo): {val_max} | Registros com Maior Valor: {', '.join(detalhes_maiores)}\n"
                f"  - Menor Valor (Mínimo): {val_min} | Registros com Menor Valor: {', '.join(detalhes_menores)}\n"
                f"  - Média Geral: {val_med:.2f}"
            )

    for col in colunas:
        if col not in colunas_numericas_encontradas and col != col_id:
            contagens = df[col].value_counts().head(4)
            if not contagens.empty:
                resumo_cat = ", ".join([f"{k}: {v}" for k, v in contagens.items()])
                bloco_estatisticas.append(f"• Frequência na coluna '{col}': {resumo_cat}")

    bloco_estatisticas.append("==================================================")
    
    documentos.append(Document(
        page_content="\n".join(bloco_estatisticas),
        metadata={
            "source": nome_arquivo,
            "arquivo": nome_arquivo,
            "tipo": "estatisticas_tabela",
            "total_linhas": total_linhas,
            "total_colunas": len(colunas),
        }
    ))
    linhas_por_bloco = 15
    df_str = df.astype(str)

    for inicio in range(0, total_linhas, linhas_por_bloco):
        fim = min(inicio + linhas_por_bloco, total_linhas)
        bloco_df = df_str.iloc[inicio:fim]

        linhas_texto = [
            f"L{idx_rel}: {', '.join([f'{col}:{row[col]}' for col in colunas])}"
            for idx_rel, (_, row) in enumerate(bloco_df.iterrows(), start=inicio + 1)
        ]

        conteudo_bloco = f"Tabela: {nome_arquivo} (Reg. {inicio + 1}-{fim} de {total_linhas})\n" + "\n".join(linhas_texto)
        metadados = {
            "source": nome_arquivo,
            "arquivo": nome_arquivo,
            "linhas_intervalo": f"{inicio + 1}-{fim}",
            "tipo": "csv_bloco_tabela",
            "total_linhas": total_linhas,
        }
        documentos.append(Document(page_content=conteudo_bloco, metadata=metadados))

    return documentos


def carregar_arquivo_texto(
    fonte_arquivo: Union[str, bytes, io.BytesIO],
    nome_arquivo: str,
) -> List[Document]:
    """
    Lê arquivos de texto simples (.txt, .md, .log).
    """
    if isinstance(fonte_arquivo, (bytes, bytearray)):
        texto = fonte_arquivo.decode("utf-8", errors="replace")
    elif isinstance(fonte_arquivo, io.BytesIO):
        texto = fonte_arquivo.read().decode("utf-8", errors="replace")
    else:
        with open(fonte_arquivo, "r", encoding="utf-8", errors="replace") as f:
            texto = f.read()

    metadados = {
        "source": nome_arquivo,
        "arquivo": nome_arquivo,
        "tipo": "texto",
    }
    return [Document(page_content=texto.strip(), metadata=metadados)]


def carregar_arquivo_generico(
    fonte_arquivo: Union[str, bytes, io.BytesIO],
    nome_arquivo: str,
) -> List[Document]:
    extensao = Path(nome_arquivo).suffix.lower()

    if extensao == ".pdf":
        return carregar_arquivo_pdf(fonte_arquivo, nome_arquivo)
    elif extensao in [".csv", ".xlsx", ".xls"]:
        return carregar_arquivo_csv(fonte_arquivo, nome_arquivo)
    else:
        return carregar_arquivo_texto(fonte_arquivo, nome_arquivo)


def dividir_documentos_em_chunks(
    documentos: List[Document],
    tamanho_chunk: int = 1000,
    sobreposicao_chunk: int = 150,
) -> List[Document]:
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao_chunk,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[Document] = []
    for doc in documentos:
        tipo = doc.metadata.get("tipo", "")
        if tipo in ["resumo_tabela", "estatisticas_tabela", "csv_bloco_tabela"]:
            chunks.append(doc)
        else:
            chunks.extend(divisor.split_documents([doc]))
    
    for indice, chunk in enumerate(chunks):
        chunk.metadata["indice_chunk"] = indice + 1

    return chunks


def armazenar_chunks_no_banco(
    chunks: List[Document],
    diretorio_persistencia: str = "./chroma_db",
    nome_colecao: str = "documentos_rag",
    tamanho_lote: int = 100,
) -> Chroma:
    banco_vetorial = obter_banco_vetorial(
        diretorio_persistencia=diretorio_persistencia,
        nome_colecao=nome_colecao,
    )

    if chunks:
        for i in range(0, len(chunks), tamanho_lote):
            lote = chunks[i : i + tamanho_lote]
            try:
                banco_vetorial.add_documents(documents=lote)
            except Exception as erro:
                print(f"Inconsistência no ChromaDB ({erro}). Resetando base limpa e inserindo novamente...")
                limpar_banco_vetorial(diretorio_persistencia=diretorio_persistencia, diretorio_dados=None)
                banco_vetorial = obter_banco_vetorial(
                    diretorio_persistencia=diretorio_persistencia,
                    nome_colecao=nome_colecao,
                )
                banco_vetorial.add_documents(documents=chunks)
                break

    return banco_vetorial


def processar_e_indexar_arquivos(
    arquivos: List[Any],
    diretorio_persistencia: str = "./chroma_db",
    tamanho_chunk: int = 1000,
    sobreposicao_chunk: int = 150,
) -> Dict[str, Any]:
    """
    Pipeline completo de ingestão: salva arquivos, divide em chunks e indexa no ChromaDB.
    """
    todos_documentos: List[Document] = []
    arquivos_processados = []

    diretorio_dados = os.getenv("DATA_FILES_DIR", "./data_files")
    os.makedirs(diretorio_dados, exist_ok=True)

    for item_arquivo in arquivos:
        if hasattr(item_arquivo, "name") and hasattr(item_arquivo, "read"):
            nome = item_arquivo.name
            conteudo = item_arquivo.read()
            if hasattr(item_arquivo, "seek"):
                item_arquivo.seek(0)
            docs = carregar_arquivo_generico(conteudo, nome)
            try:
                caminho_salvar = os.path.join(diretorio_dados, nome)
                with open(caminho_salvar, "wb") as f_out:
                    f_out.write(conteudo if isinstance(conteudo, (bytes, bytearray)) else conteudo.encode("utf-8"))
            except Exception as e:
                print(f"Aviso ao salvar arquivo no disco: {e}")
        else:
            caminho = str(item_arquivo)
            nome = Path(caminho).name
            docs = carregar_arquivo_generico(caminho, nome)
            try:
                shutil.copy2(caminho, os.path.join(diretorio_dados, nome))
            except Exception:
                pass

        todos_documentos.extend(docs)
        arquivos_processados.append(nome)

    if not todos_documentos:
        return {
            "sucesso": False,
            "mensagem": "Nenhum conteúdo válido pôde ser extraído dos arquivos fornecidos.",
            "total_documentos": 0,
            "total_chunks": 0,
            "arquivos": arquivos_processados,
        }

    chunks = dividir_documentos_em_chunks(
        documentos=todos_documentos,
        tamanho_chunk=tamanho_chunk,
        sobreposicao_chunk=sobreposicao_chunk,
    )

    armazenar_chunks_no_banco(
        chunks=chunks,
        diretorio_persistencia=diretorio_persistencia,
    )

    return {
        "sucesso": True,
        "mensagem": f"Indexação concluída com sucesso! {len(chunks)} chunks gerados.",
        "total_documentos": len(todos_documentos),
        "total_chunks": len(chunks),
        "arquivos": arquivos_processados,
    }


def obter_dataframes_disponiveis(diretorio_dados: str = "./data_files") -> Dict[str, pd.DataFrame]:
    dfs = {}
    if not os.path.exists(diretorio_dados):
        return dfs

    for arquivo in os.listdir(diretorio_dados):
        caminho_completo = os.path.join(diretorio_dados, arquivo)
        extensao = Path(arquivo).suffix.lower()
        try:
            if extensao in [".csv", ".tsv"]:
                try:
                    df = pd.read_csv(caminho_completo, sep=None, engine="python")
                except Exception:
                    df = pd.read_csv(caminho_completo, sep=",")
                dfs[arquivo] = df
            elif extensao in [".xlsx", ".xls"]:
                df = pd.read_excel(caminho_completo)
                dfs[arquivo] = df
        except Exception as e:
            print(f"Erro ao carregar dataframe de {arquivo}: {e}")

    return dfs


def obter_esquema_tabelas(diretorio_dados: str = "./data_files") -> str:
    """
    Gera resumo das tabelas (colunas, tipos e amostras) 
    para orientar a llm (faz uma diferença absurda)
    """
    dfs = obter_dataframes_disponiveis(diretorio_dados=diretorio_dados)
    if not dfs:
        return "Nenhuma tabela de dados estruturados (CSV/Excel) disponível no momento."

    resumo_esquemas = []
    for nome_tabela, df in dfs.items():
        colunas_info = [
            f"    - {col} ({df[col].dtype}): Amostra {list(df[col].dropna().unique()[:3])}"
            for col in df.columns
        ]
        resumo_esquemas.append(
            f"Tabela: '{nome_tabela}'\n"
            f"Total de Registros: {len(df)}\n"
            f"Colunas:\n" + "\n".join(colunas_info)
        )

    return "\n\n".join(resumo_esquemas)


def executar_codigo_pandas(codigo: str, diretorio_dados: str = "./data_files") -> Dict[str, Any]:
    dfs = obter_dataframes_disponiveis(diretorio_dados=diretorio_dados)
    if not dfs:
        return {"sucesso": False, "resultado": "", "erro": "Nenhum arquivo de dados encontrado."}

    primeiro_df = list(dfs.values())[0]
    primeiro_nome = list(dfs.keys())[0]

    local_vars = {
        "pd": pd,
        "df": primeiro_df,
        "dfs": dfs,
        "resultado": None,
    }

    for nome_arq, df_inst in dfs.items():
        var_nome = nome_arq.replace(".", "_").replace("-", "_")
        local_vars[var_nome] = df_inst

    codigo_limpo = codigo.strip()
    if codigo_limpo.startswith("```"):
        linhas = codigo_limpo.split("\n")
        if linhas and linhas[0].startswith("```"):
            linhas = linhas[1:]
        if linhas and linhas[-1].startswith("```"):
            linhas = linhas[:-1]
        codigo_limpo = "\n".join(linhas).strip()

    try:
        if "\n" not in codigo_limpo and not codigo_limpo.startswith("resultado") and not codigo_limpo.startswith("res"):
            codigo_exec = f"resultado = {codigo_limpo}"
        else:
            codigo_exec = codigo_limpo

        import builtins
        safe_builtins = builtins.__dict__.copy()

        exec(codigo_exec, {"__builtins__": safe_builtins}, local_vars)

        res = local_vars.get("resultado")
        if res is None:
            res = local_vars.get("res")

        if res is None:
            return {"sucesso": True, "resultado": "Consulta executada sem retorno de dados.", "codigo": codigo_limpo, "arquivo": primeiro_nome}

        if isinstance(res, pd.DataFrame):
            total_regs = len(res)
            if total_regs > 50:
                res_str = f"Total de registros encontrados: {total_regs}\n\nPrimeiros 40 resultados:\n" + res.head(40).to_string(index=False)
            else:
                res_str = f"Total de registros encontrados: {total_regs}\n\n" + res.to_string(index=False)
        elif isinstance(res, pd.Series):
            res_str = f"Total de registros: {len(res)}\n\n" + res.to_string()
        else:
            res_str = str(res)

        return {"sucesso": True, "resultado": res_str, "codigo": codigo_limpo, "arquivo": primeiro_nome}

    except Exception as e:
        return {"sucesso": False, "resultado": "", "erro": str(e), "codigo": codigo_limpo, "arquivo": primeiro_nome}


def limpar_banco_vetorial(
    diretorio_persistencia: str = "./chroma_db",
    diretorio_dados: Optional[str] = "./data_files",
) -> bool:
    sucesso = True
    if diretorio_persistencia and os.path.exists(diretorio_persistencia):
        try:
            shutil.rmtree(diretorio_persistencia)
            os.makedirs(diretorio_persistencia, exist_ok=True)
        except Exception as erro:
            print(f"Erro ao limpar banco vetorial: {erro}")
            sucesso = False

    if diretorio_dados and os.path.exists(diretorio_dados):
        try:
            shutil.rmtree(diretorio_dados)
            os.makedirs(diretorio_dados, exist_ok=True)
        except Exception as erro:
            print(f"Erro ao limpar a pasta de dados: {erro}")
            sucesso = False

    return sucesso


def obter_estatisticas_banco(
    diretorio_persistencia: str = "./chroma_db",
    diretorio_dados: str = "./data_files",
    nome_colecao: str = "documentos_rag",
) -> Dict[str, Any]:
    fontes = set()
    total_chunks = 0

    if os.path.exists(diretorio_persistencia):
        try:
            banco = obter_banco_vetorial(
                diretorio_persistencia=diretorio_persistencia,
                nome_colecao=nome_colecao,
            )
            dados = banco.get()
            ids = dados.get("ids", [])
            metadados = dados.get("metadatas", [])
            total_chunks = len(ids)
            for meta in metadados:
                if meta and "source" in meta:
                    fontes.add(meta["source"])
                elif meta and "arquivo" in meta:
                    fontes.add(meta["arquivo"])
        except Exception:
            pass

    if os.path.exists(diretorio_dados):
        for f in os.listdir(diretorio_dados):
            fontes.add(f)

    return {
        "total_chunks": total_chunks,
        "fontes": sorted(list(fontes)),
    }
