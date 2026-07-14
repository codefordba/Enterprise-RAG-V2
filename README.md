# Enterprise-RAG-V2: Decoupled Multi-Tenant Knowledge Ingestion & Assessment Console

A production-grade, highly scalable, multi-tenant Retrieval-Augmented Generation (RAG) platform. The system enforces strict logical tenant partitioning inside a shared vector database, extracts visually complex borderless financial tables, and routes conversational inference dynamically to specialized model weights, on-prem services, or cloud endpoints.

The application uses a modern **decoupled React SPA frontend** and a **FastAPI backend**, containerized as a single multi-stage build image.

---

## 🎯 Key Objectives

1.  **Strict Logical Tenant Partitioning**: Enforces strict departmental data boundaries (Finance, HR, Legal) inside a single vector pool using Qdrant payload keyword constraints, avoiding the administrative overhead of managing thousands of database collections.
2.  **Structural Document Extraction**: Employs layout-aware visual element extraction (`pdfplumber`) to isolate grid balance sheets and borderless data blocks, mapping them into native Markdown grids before chunking.
3.  **Document Lineage & Version Control**: Automatically marks active document revisions as `"is_latest": true` while deprecating older chunks to `"is_latest": false`. Deterministic UUIDv5 generation overrides duplication and guarantees clean versioning.
4.  **Credential & Configuration Isolation**: Integrates Fernet symmetric AES-128 encryption to secure connection endpoints and credentials on disk.

---

## 🛠️ The Tech Stack

*   **Frontend**: React SPA (TypeScript + Vite)
*   **Backend**: FastAPI (Python 3.11 + Uvicorn)
*   **Vector Database**: Qdrant (1.9+)
*   **Embeddings Compute**: Text Embeddings Inference (TEI CPU/GPU - `BAAI/bge-large-en-v1.5`)
*   **Reranker**: TEI Cross-Encoder Reranker (`BAAI/bge-reranker-large`)
*   **Structural Parsing**: pdfplumber
*   **Text Splitters**: LangChain Experimental Semantic Chunker
*   **Symmetric Encryption**: Cryptography Fernet
*   **Evaluation Framework**: RAGAS (0.4.x) with custom parser sanitizers

---

## 🏛️ New Production-Grade Features & Updates

### 1. Decoupled SSE Streaming Pipeline
*   **Server-Sent Events (SSE)**: The FastAPI server streams real-time token outputs using SSE (`text/event-stream`). The React frontend parses chunks chunk-by-chunk using Fetch stream readers for zero-lag rendering.
*   **Timing Telemetry**: Displays Time-to-First-Token (TTFT) metrics, generation velocity (tokens/sec), and token budget distributions in a collapsible query trace split-pane.

### 2. Infrastructure Diagnostics & Health Checks
*   **One-Click Diagnostics**: An interactive telemetry grid validates active connectivity states to LLMs (completes a dummy chat call), Embeddings, Reranker, and Qdrant DB collection pools.
*   **Diagnostics SLA Warning Flags**: Tints latency metrics amber when execution limits exceed preset SLAs (e.g. Reranker latency > 300ms, DB query > 100ms, TTFT > 500ms).

### 3. Dynamic vLLM Fallback Routing
*   **Adaptive Target Resolution**: Inspects currently loaded models on the live vLLM `/models` endpoint. If a tenant's registered LoRA adapter weight matrix is missing, the query is routed to the active connection profile's `DEFAULT_MODEL_ID` to prevent `404 Not Found` API crashes.
*   **Synthetic QA Test Case Fallback**: Prevents synthesis loops from breaking on changed server environments by falling back to the active base LLM model.

### 4. Interactive IP Address Masking
*   **Secure Telemetry Presentation**: Masking algorithms automatically replace raw IPv4 target addresses in metrics cards with `***.***.***.***`. Clicking on the masked address toggles it to reveal the real IP instantly, preventing sensitive configuration leakage in screenshots.

### 5. Weighted Ragas Status Grading Matrix
*   **Production Quality Gating**: Run metrics are evaluated against a balanced multi-tier quality gate to determine the `PASSED`, `MARGINAL`, or `FAILED` run status:
    *   **Tier 1 (Hard Gates)**: Faithfulness `> 0.80` and Answer Relevancy `> 0.80`. Failing either forces a `FAILED` result.
    *   **Tier 2 (Health Gate)**: If Tier 1 passes and Context Recall `>= 0.80`, the run is marked `PASSED`.
    *   **Fallback**: Otherwise, the run is tagged as `MARGINAL`.

### 6. Vendor-Neutral Prompt Engineering Layer
*   **Agile Prompt Optimization**: Prompts are optimized to support any instruction-tuned open-source model (Mistral, Llama, Qwen, DeepSeek) without vendor-specific tags. Enforces zero prior knowledge, strict context-only grounding, and exact disclaimers for missing documents.

---

## 🐳 Containerized Quick Start (Docker Compose)

Deploy the entire microservice ecosystem—including the FastAPI React UI, Qdrant, TEI Embeddings, and GPU CUDA Reranker—in a single command.

### 1. Configure the Environment
Copy the example environment file and secure your local workspace files:
```bash
cp .env.example .env
mkdir -p qdrant_storage embedding_cache reranker_cache
touch .enc_key model_profiles.enc tenant_registry.json eval_runs.json
```

### 2. Build & Launch Containers
Run the Docker Compose suite:
```bash
docker compose up -d --build
```
*   **FastAPI & React UI**: Exposed at `http://localhost:8000` (or the mapped host port).
*   **Qdrant Console**: Exposed at `http://localhost:6333`.
*   **Embedding Server**: Exposed at `http://localhost:8090`.
*   **Cross-Encoder Reranker**: Exposed at `http://localhost:8081`.

### 3. Verify Container Status
```bash
docker compose ps
```

---

## 🛠️ Direct Docker Container Deployment (Manual)

If you prefer building and running the combined frontend/backend application image independently on a custom bridge network:

### 1. Create a Shared Docker Network
```bash
docker network create --driver bridge llm-infra-net
```

### 2. Run Database & Inference Services
```bash
# Qdrant Database
docker run -d \
  --name qdrant-server \
  --network llm-infra-net \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant:latest

# BGE Embeddings compute node (CPU-only version)
docker run -d \
  --name embedding-server \
  --network llm-infra-net \
  -p 8090:80 \
  -v $(pwd)/embedding_cache:/data \
  --restart unless-stopped \
  ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-latest \
  --model-id BAAI/bge-large-en-v1.5
```

### 3. Build the Decoupled Image
```bash
docker build -t enterprise-rag-v2:latest .
```

### 4. Run the Decoupled Application Container
```bash
docker run -d \
  --name enterprise-rag-app \
  --network llm-infra-net \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/.enc_key:/app/.enc_key \
  -v $(pwd)/model_profiles.enc:/app/model_profiles.enc \
  -v $(pwd)/tenant_registry.json:/app/tenant_registry.json \
  -v $(pwd)/eval_runs.json:/app/eval_runs.json \
  -e QDRANT_HOST=qdrant-server \
  -e QDRANT_PORT=6333 \
  -e TEI_ENDPOINT=http://embedding-server:80/embed \
  --restart unless-stopped \
  enterprise-rag-v2:latest
```
Access the application directly at `http://localhost:8000`.
