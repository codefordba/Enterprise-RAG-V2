# Enterprise-RAG-V2: Multi-Tenant Knowledge Ingestion & Inference Console

A production-grade, highly scalable, multi-tenant Retrieval-Augmented Generation (RAG) platform. The system enforces strict logical tenant partitioning inside a shared vector database, extracts visually complex borderless financial tables, and routes conversational inference dynamically to specialized model weights or cloud endpoints.

---

## 🎯 Objective
Enterprise-RAG-V2 resolves the primary constraints of standard RAG architectures in corporate networks:
1.  **Cross-Tenant Data Leakage**: Enforces strict departmental data boundaries (Finance, HR, Legal) inside a single vector pool using Qdrant HNSW payload keyword constraints, avoiding the administrative overhead of managing thousands of database collections.
2.  **Structural Document Destruction**: Employs layout-aware visual element extraction (`pdfplumber`) to isolate grid balance sheets and borderless data blocks, mapping them into native Markdown grids before chunking.
3.  **Document Lineage & Version Bleeding**: Implements a robust state-flagging version control system. It archives old document revisions while securely gating search queries to reference only the latest updates, preventing outdated data from contaminating model answers.
4.  **Credential & Configuration Exposure**: Integrates FIPS-compliant symmetric key encryption to store connection endpoints and API keys securely on disk, enabling on-the-fly model swapping.

---

## 🛠️ The Tech Stack
*   **Frontend & Ops Dashboard**: [Streamlit](https://streamlit.io/) (1.35+)
*   **Vector Database**: [Qdrant](https://qdrant.tech/) (1.9+)
*   **Embeddings Compute Server**: HuggingFace [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) (`BAAI/bge-large-en-v1.5`)
*   **Structural Parsing**: [pdfplumber](https://github.com/jasonmc/pdfplumber) (layout element extraction)
*   **Text Splitters**: [LangChain Experimental](https://github.com/langchain-ai/langchain) (Semantic Chunker)
*   **Symmetric Encryption**: Python `cryptography` (FIPS-compliant Fernet AES-128 encryption)
*   **Data Layout Parsing**: `pandas` & `openpyxl` (Excel spreadsheets indexing)
*   **Benchmarking Framework**: [Ragas](https://github.com/explodinggradients/ragas) (0.1.7)

---

## 🏛️ Key System Modules & Features

### 1. Document Processing Panel (Layout-Aware Ingestion)
*   **Layout-Aware PDF Element Extraction**: Automatically isolates grid tables and structures them into valid Markdown table syntax, maintaining column alignment.
*   **Table-Preserved Semantic Text Splitting**: Uses LangChain's `SemanticChunker` (utilizing embedding distance thresholds) to group prose sentences semantically while bypassing splits for tabular elements to prevent grid data fragmentation.
*   **Progress & Workflow UI Feedback**: Renders a dynamic real-time progress bar and phase status descriptions (e.g. *Phase 1: Deprecating old versions*, *Phase 2: Generating vector embeddings*) inside Streamlit.

### 2. Document Lineage & Version Control System
*   **Logical Lineage Decoupling**: Tracks documents by a logical `document_family` identifier (e.g. `leave_policy`), decoupling them from physical file uploads.
*   **State Flagging (`is_latest`)**: Every newly ingested text chunk is indexed with the metadata property `"is_latest": true`.
*   **Atomic Pre-Ingestion Deprecation**: Before writing a new version of a document, a scroll query finds all existing chunks for that tenant and family with `is_latest: true`, and performs a partial payload update flipping them to `false` (retaining older revisions for audit trails without cluttering active searches).
*   **Deterministic Chunk Addressing (`uuid.uuid5`)**: Banned random UUIDv4 generation. Calculates point IDs deterministically via `uuid.uuid5` using a composite namespace seed string: `f"{tenant_id}_{document_family}_{document_version}_{chunk_index}"`. Re-uploading the exact same document version cleanly overwrites existing vectors instead of duplicating them.

### 3. Isolated Grounded Query Playground
*   **Strict Query Gating**: The search retrieval engine enforces a mandatory payload metadata filter: `{"key": "is_latest", "match": {"value": true}}`. Outdated document versions are completely invisible to search passes and RAG inference, permanently preventing semantic bleeding.
*   **Unified Client Integrations**: Leverages a centralized `MultiTenantQueryEngine` running the Qdrant SDK `query_points` API for all retrievals, synchronizing parameters across search, generation, and evaluation layers.

### 4. RAG Assessment & Evaluation Console
*   **Automated Benchmarking**: Runs Ragas evaluation passes to compute standard metrics: **Faithfulness** (groundedness / hallucination check), **Answer Relevance**, **Context Precision** (ranking accuracy), and **Context Recall** (fact coverage).
*   **Pre-Upload Verification**: Allows uploading custom ground-truth CSV test sets or generating synthetic test sets directly from the active vector index.
*   **Exportable Logs**: Renders interactive, row-by-row metric dataframes and provides a download button to export the complete evaluation logs (inputs, contexts, answers, and scores) directly into CSV spreadsheets.

### 5. SandyGPT Conversational Workspace & Tenant Administration
*   A ChatGPT-style workspace for model dialogue (non-RAG) with tenant-isolated history.
*   Allows registration of named connection profiles (local vLLM endpoint vs Cloud Gemini API) and saves them in encrypted format (`model_profiles.enc`) using a secure local key file (`.enc_key`).

---

## 🚀 How to Build and Deploy the Environment

### Method A: Fully Containerized Deploy using the `llm-infra-net` Bridge
Follow these steps to build and launch all containers (UI, Qdrant, and TEI) manually inside a shared bridge network named **`llm-infra-net`**:

1.  **Configure Environment Variables**:
    Clone `.env.example` to create `.env` and fill in your connection variables (such as `LLM_API_KEY`):
    ```bash
    cp .env.example .env
    ```
2.  **Create local storage folders on host**:
    ```bash
    mkdir -p qdrant_storage embedding_cache sample_data
    touch .enc_key model_profiles.enc tenant_registry.json
    ```
3.  **Create the custom Docker bridge network**:
    ```bash
    docker network create --driver bridge llm-infra-net
    ```
4.  **Launch Qdrant Container**:
    ```bash
    docker run -d \
      --name qdrant-server \
      --network llm-infra-net \
      -p 6333:6333 -p 6334:6334 \
      -v $(pwd)/qdrant_storage:/qdrant/storage \
      --restart unless-stopped \
      qdrant/qdrant:latest
    ```
5.  **Launch Embedding Server (TEI) Container**:
    ```bash
    docker run -d \
      --name embedding-server \
      --network llm-infra-net \
      -p 8080:80 \
      -v $(pwd)/embedding_cache:/data \
      --restart unless-stopped \
      ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-latest \
      --model-id BAAI/bge-large-en-v1.5 --max-client-batch-size 128
    ```
6.  **Build the Streamlit App Container**:
    ```bash
    docker build -t enterprise-rag-ui:latest .
    ```
7.  **Launch the Streamlit UI Container**:
    ```bash
    docker run -d \
      --name streamlit-ui \
      --network llm-infra-net \
      -p 8501:8501 \
      -v $(pwd)/.env:/app/.env \
      -v $(pwd)/.enc_key:/app/.enc_key \
      -v $(pwd)/model_profiles.enc:/app/model_profiles.enc \
      -v $(pwd)/tenant_registry.json:/app/tenant_registry.json \
      -e QDRANT_HOST=qdrant-server \
      -e QDRANT_PORT=6333 \
      -e TEI_ENDPOINT=http://embedding-server:80/embed \
      --restart unless-stopped \
      enterprise-rag-ui:latest
    ```

Streamlit UI will be live at `http://localhost:8501` and fully integrated with backend dependencies.

---

### Method B: Hybrid Local Deploy (Recommended for Development)
Use this method if you wish to run the Streamlit UI locally on your host machine while running the database and embedding containers in Docker.

1.  **Start Database & Embedding Services**:
    ```bash
    docker compose up -d
    ```
2.  **Initialize Python Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install Host Package Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch the Streamlit Dashboard**:
    Ensure the path resolver loads correctly:
    ```bash
    .venv/bin/streamlit run src/main.py
    ```

Open your browser to `http://localhost:8501`.
