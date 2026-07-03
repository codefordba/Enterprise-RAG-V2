# Enterprise-RAG: Multi-Tenant Knowledge Ingestion & Inference Console

A production-grade, highly scalable, multi-tenant Retrieval-Augmented Generation (RAG) platform. The system enforces strict logical tenant partitioning inside a shared vector database, extracts visually complex borderless financial tables, and routes conversational inference dynamically to specialized model weights or cloud endpoints.

---

## 🎯 Objective
Enterprise-RAG resolves the three primary constraints of standard RAG architectures in corporate networks:
1.  **Cross-Tenant Data Leakage**: Enforces strict departmental data boundaries (Finance, HR, Legal) inside a single vector pool using Qdrant HNSW payload keyword constraints, avoiding the administrative overhead of managing thousands of database collections.
2.  **Structural Document Destruction**: Employs layout-aware visual element extraction (`pdfplumber`) to isolate grid balance sheets and borderless data blocks, mapping them into native Markdown grids before chunking.
3.  **Credential & Configuration Exposure**: Integrates FIPS-compliant symmetric key encryption to store connection endpoints and API keys securely on disk, enabling on-the-fly model swapping.

---

## 🛠️ The Tech Stack
*   **Frontend & Ops Dashboard**: [Streamlit](https://streamlit.io/) (1.35+)
*   **Vector Database**: [Qdrant](https://qdrant.tech/) (1.9+)
*   **Embeddings Compute Server**: HuggingFace [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) (`BAAI/bge-large-en-v1.5`)
*   **Structural Parsing**: [pdfplumber](https://github.com/jasonmc/pdfplumber) (layout element extraction)
*   **Text Splitters**: [LangChain Experimental](https://github.com/langchain-ai/langchain) (Semantic Chunker)
*   **Symmetric Encryption**: Python `cryptography` (FIPS-compliant Fernet AES-128 encryption)
*   **Data Layout Parsing**: `pandas` & `openpyxl` (Excel spreadsheets indexing)

---

## 🏛️ Features

### 1. SandyGPT Conversational Workspace
*   A ChatGPT-style conversational chat console.
*   Enforces direct model dialogue (non-RAG) with adjustable parameters (temperature set to `0.7` for fluent dialog).
*   Maintains conversation history isolated per tenant.

### 2. Grounded Query Playground
*   Retrieval-scoped sandbox workspace utilizing tenant payload restrictions.
*   Instructs the LLM under strict system grounding guidelines (prevents background weight hallucinations; answers only from context).
*   Renders clickable citation audit cards linking directly to the source file, page number, and vector similarity ranking score.

### 3. Document Processing Panel
*   Layout-aware visual document ingestion supporting PDFs and Excel Workbooks (`.xlsx`, `.xls`).
*   Extracts borderless spreadsheet matrices and formats them to Markdown grids, appending them to target semantic chunks.
*   Has content hash checks to skip duplicate ingestion and lineage checks to purge old versions.

### 4. Tenant Administration & Model Profiles Directory
*   **On-The-Fly Swapping**: Allows registration of named connection profiles (local vLLM endpoint vs Cloud Gemini API) and saves them in encrypted format (`model_profiles.enc`) using a secure local key file (`.enc_key`).
*   **Metadata Registry**: Persists active tenant maps in `tenant_registry.json` so tenant spaces are preserved across restarts.

---

## 🚀 How to Reproduce & Run the Application

Follow these steps to deploy and run the entire environment:

### Step 1: Clone and Set Up Environment Variables
Copy `.env.example` to create `.env` and fill in your connection variables:
```bash
cp .env.example .env
```
*Example `.env` configuration for Cloud Google Gemini:*
```ini
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=tenant_knowledge_base
TEI_ENDPOINT=http://localhost:8080/embed
CLIENT_BATCH_LIMIT=32

LLM_DEPLOYMENT_MODE=CLOUD
LLM_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_API_KEY=YOUR_GEMINI_API_KEY
DEFAULT_MODEL_ID=gemini-1.5-flash
```

---

### Step 2: Spin Up the Infrastructure Containers
Launch the Vector Database (Qdrant) and Embedding Server (TEI) using Docker Compose:
```bash
docker compose up -d
```
Verify that the services are online:
- **Qdrant**: `http://localhost:6333`
- **TEI Core**: `http://localhost:8080/info`

---

### Step 3: Run the Streamlit Application UI

#### Option A: Running Locally (Recommended for Development)
1.  **Initialize Python Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install System Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Streamlit Dashboard**:
    ```bash
    streamlit run src/main.py
    ```
    Open your browser to `http://localhost:8501`.

#### Option B: Running via Docker
Build and start the application container:
```bash
docker build -t enterprise-rag-app .
docker run -d -p 8501:8501 \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/.enc_key:/app/.enc_key \
  -v $(pwd)/model_profiles.enc:/app/model_profiles.enc \
  -v $(pwd)/tenant_registry.json:/app/tenant_registry.json \
  --name rag-console enterprise-rag-app
```
*(Mounting files as volumes preserves your encryption keys and tenant settings on your host machine across container updates).*
