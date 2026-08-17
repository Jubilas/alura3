# Oracle OCI Agent — Sistema Híbrido RAG + Motor Analítico Pandas

Aplicação de **Inteligência Artificial Generativa e Análise Documental** construída com **LangChain**, **LangGraph**, **Groq** (*Llama 3.3 70B*)(16/08/2026), **FastEmbed** (*embeddings locais*), **ChromaDB**, **Pandas Engine** e interface em **Streamlit**.

---

## Arquitetura do Sistema

O agente adota um padrão híbrido orquestrado pelo **LangGraph**:

```text
                           ┌──────────────────────────────┐
                           │      Pergunta do Usuário     │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │   no_rotear_intencao │
                               │  (Classifica Intenção│
                               │   baseado no Schema) │
                               └──────────┬───────────┘
                                          │
                     ┌────────────────────┴───────────────────┐
                     ▼                                        ▼
       [Intenção Tabular / Matemática]               [Intenção Textual / Regras]
       ┌──────────────────────────────┐              ┌─────────────────────────┐
       │      no_analisar_tabela      │              │  no_recuperar_documentos│
       │    (Gera código Pandas +     │              │ (Busca semântica no     │
       │     Auto-correção de erros)  │              │  ChromaDB com FastEmbed)│
       └──────────────┬───────────────┘              └────────────┬────────────┘
                      │ (Se 0 registros / Fallback)               │
                      └───────────────────────┬───────────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │    no_gerar_resposta    │
                                 │   (Síntese com Groq     │
                                 │    Llama 3.3 70B)       │
                                 └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │ Resposta com Fontes UI  │
                                 └─────────────────────────┘
```

---

## Demonstração Visual (Alura Challenge)

Abaixo estão as capturas de tela demonstrando a interface em execução, upload e indexação de documentos, processamento de perguntas analíticas e exibição das fontes:

| Tela Inicial e Configurações | Ingestão e Estatísticas da Base |
| :---: | :---: |
| ![Tela Inicial](ALURACHALLENGE/Captura%20de%20tela%202026-08-17%20004923.png) | ![Ingestão e Estatísticas](ALURACHALLENGE/Captura%20de%20tela%202026-08-17%20004934.png) |

| Consulta Analítica com Pandas | Pagina da frente |
| :---: | :---: |
| ![Consulta Tabular](ALURACHALLENGE/Captura%20de%20tela%202026-08-17%20005316.png) | ![Foto4](ALURACHALLENGE/Captura%20de%20tela%202026-08-17%20005531.png) |

---

## Estrutura de pastas

```text
alura3/
├── ALURACHALLENGE/        # Capturas de tela e evidências do desafio
├── Documentos_exemplo/    # Documentos de teste (PDFs, CSV, TXTs)
├── src/
│   ├── __init__.py        # Exportações e imports centralizados
│   ├── state.py           # Definição do EstadoAgente (TypedDict)
│   ├── ingestion.py       # Ingestão, FastEmbed, ChromaDB e Pandas Engine
│   └── graph.py           # Grafo híbrido LangGraph com Groq Llama 3.3
├── app.py                 # Interface Streamlit
├── requirements.txt       # Dependências Python
├── .env.example           # Modelo de variáveis de ambiente
├── Dockerfile             # Container para deploy na OCI
├── docker-compose.yml     # Orquestração com volumes persistentes
├── .gitignore             # Regras de exclusão do Git
└── README.md              # Documentação
```

---

## Tecnologias Usadas

- **Python 3.10+ / 3.11**
- **LangChain & LangGraph:** Orquestração de grafo (`StateGraph`).
- **Groq (Llama 3.3 70B Versatile):**.
- **FastEmbed (`BAAI/bge-small-en-v1.5`):** Embeddings locais.
- **ChromaDB:** Banco vetorial persistente em disco com mecanismo de solucionar os próprios problemas.
- **Pandas:** Execução dinâmica e determinística de cálculos sobre grandes volumes de dados.
- **pypdf:** Extração de texto página por página em arquivos PDF.
- **Streamlit:** Interface web.

---

## Como Executar o Projeto

### Opção 1: Via Docker

Crie o arquivo `.env` a partir do modelo:

```bash
    cp .env.example .env
```

   Adicione sua chave do Groq no arquivo `.env`:

```env
   GROQ_API_KEY=gsk_sua_chave_aqui
```

Inicie os containers:

```bash
   docker-compose up --build
```

Acesse no navegador: `http://localhost:8501`

---

### Opção 2: Execução Local com Python

1. Crie e ative o ambiente virtual:

   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure o arquivo `.env`:

   ```bash
   copy .env.example .env
   ```

4. Inicie a aplicação Streamlit:

   ```bash
   streamlit run app.py
   ```
