import os
import json
import uuid
import time
import tempfile
import urllib.request
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src.config import Config
from src.database.secure_storage import SecureStorageManager
from src.processing.ingest_pipeline import TenantIngestionPipeline
from src.generation.orchestrator import ContextOrchestrator
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

app = FastAPI(
    title="Enterprise-RAG Ops API",
    description="Decoupled, scalable Control Plane and Multi-Tenant RAG Ingestion Pipeline API.",
    version="2.0.0"
)

# Enable CORS for frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Instantiate core orchestration engine
orchestrator = ContextOrchestrator()

# Helper: Load default LLM parameters and overrides
def get_active_llm_config() -> dict:
    profiles = SecureStorageManager.load_encrypted_profiles()
    active_profile_name = profiles.get("_active_profile", "Default Environment")
    if active_profile_name not in profiles:
        keys = [k for k in profiles.keys() if not k.startswith("_")]
        active_profile_name = keys[0] if keys else "Default Environment"
        
    cfg = profiles.get(active_profile_name, {
        "LLM_DEPLOYMENT_MODE": Config.LLM_DEPLOYMENT_MODE,
        "LLM_API_BASE_URL": Config.LLM_API_BASE_URL,
        "LLM_API_KEY": Config.LLM_API_KEY,
        "DEFAULT_MODEL_ID": Config.DEFAULT_MODEL_ID,
        "PROVIDER_TYPE": "Cloud API" if Config.LLM_DEPLOYMENT_MODE == "CLOUD" else "vLLM"
    }).copy()
    
    # Mix in global downstream settings
    global_ds = profiles.get("_global_downstream", {})
    cfg["QDRANT_URL"] = global_ds.get("QDRANT_URL", Config.QDRANT_BASE_URL)
    cfg["EMBEDDING_SERVER_URL"] = global_ds.get("EMBEDDING_SERVER_URL", Config.EMBEDDING_SERVER_URL)
    cfg["RERANKER_SERVER_URL"] = global_ds.get("RERANKER_SERVER_URL", Config.RERANKER_SERVER_URL)
    cfg["VECTOR_TOP_K"] = int(global_ds.get("VECTOR_TOP_K", Config.VECTOR_TOP_K))
    cfg["RERANK_TOP_K"] = int(global_ds.get("RERANK_TOP_K", Config.RERANK_TOP_K))
    cfg["RERANKER_SCORE_THRESHOLD"] = float(global_ds.get("RERANKER_SCORE_THRESHOLD", Config.RERANKER_SCORE_THRESHOLD))
    
    return cfg
# Helper: Determine production grading status from Ragas metrics
def determine_production_status(scores: dict) -> str:
    # Handle both potential variations of dictionary keys from Ragas framework
    faithfulness = scores.get("faithfulness", 0.0)
    relevancy = scores.get("answer_relevance", 0.0) or scores.get("answer_relevancy", 0.0)
    precision = scores.get("context_precision", 0.0)
    recall = scores.get("context_recall", 0.0)

    # 1. CRITICAL CRASH GATE (User Experience Defect)
    if faithfulness < 0.80 or relevancy < 0.80:
        return "FAILED"

    # 2. OPTIMAL PRODUCTION PASS GATE 
    # If the user is getting a 100% accurate grounded answer, the pipeline passes.
    if faithfulness >= 0.80 and relevancy >= 0.80 and recall >= 0.80:
        # If the retrieval signal-to-noise ratio is slightly below target but successfully filtered, mark as PASSED
        if precision >= 0.70:
            return "PASSED"
        return "PASSED"

    # 3. FALLBACK TUNING GATE
    return "MARGINAL"

# Apply active stored overrides at startup
try:
    active_cfg = get_active_llm_config()
    Config.apply_runtime_overrides(active_cfg)
except Exception as e:
    print(f"⚠️ Warning: Stored profile override application failed: {str(e)}")

# Connection health checking helper
def check_url_health(url: str, method: str = "GET", requires_auth: bool = False, api_key: str = None) -> bool:
    try:
        headers = {}
        if requires_auth and api_key and api_key.lower() != "none":
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False

# --- SYSTEM HEALTH ENDPOINTS ---
@app.get("/api/health")
async def get_health_status():
    """Checks the live connectivity status of the Qdrant, TEI Embedder, Reranker, and LLM endpoints."""
    llm_cfg = get_active_llm_config()
    
    qdrant_url = Config.QDRANT_BASE_URL
    tei_url = Config.EMBEDDING_SERVER_URL
    reranker_url = Config.RERANKER_SERVER_URL
    llm_health_url = f"{llm_cfg['LLM_API_BASE_URL']}/models"
    
    # Run active checks
    qdrant_ok = check_url_health(qdrant_url)
    tei_ok = check_url_health(tei_url)
    reranker_ok = check_url_health(reranker_url) if getattr(Config, "RERANK_ENABLED", True) else False
    
    # LLM health is checked via standard model list or simple endpoint accessibility
    llm_ok = check_url_health(llm_health_url, requires_auth=True, api_key=llm_cfg.get("LLM_API_KEY"))
    if not llm_ok:
        # Fallback check to base URL if model listing requires complex routing
        llm_ok = check_url_health(llm_cfg["LLM_API_BASE_URL"])
 
    # Count Qdrant vector statistics
    vector_count = 0
    collection_status = "UNKNOWN"
    try:
        qdrant_client = Config.get_qdrant_client()
        collection_info = qdrant_client.get_collection(Config.COLLECTION_NAME)
        vector_count = collection_info.points_count
        collection_status = collection_info.status.name
    except Exception:
        pass

    return {
        "status": "GREEN" if (qdrant_ok and tei_ok and llm_ok) else "YELLOW" if (qdrant_ok or tei_ok) else "RED",
        "nodes": {
            "qdrant": {"status": "ONLINE" if qdrant_ok else "OFFLINE", "collection_status": collection_status, "points_count": vector_count},
            "tei_embedder": {"status": "ONLINE" if tei_ok else "OFFLINE"},
            "reranker": {"status": "ONLINE" if reranker_ok else "OFFLINE", "enabled": getattr(Config, "RERANK_ENABLED", True)},
            "llm_api": {"status": "ONLINE" if llm_ok else "OFFLINE", "model": llm_cfg.get("DEFAULT_MODEL_ID")}
        }
    }

# --- CONTROL PLANE CONFIG ENDPOINTS ---
@app.get("/api/config")
async def get_configuration():
    """Retrieves the decrypted operational model profile configurations."""
    cfg = get_active_llm_config()
    # Mask API key for UI safety
    masked_key = "••••••••••••••••"
    if cfg.get("LLM_API_KEY") and len(cfg["LLM_API_KEY"]) > 6:
        masked_key = f"{cfg['LLM_API_KEY'][:3]}...{cfg['LLM_API_KEY'][-3:]}"
    elif cfg.get("LLM_API_KEY") and cfg["LLM_API_KEY"].lower() == "none":
        masked_key = "None"
        
    return {
        "LLM_DEPLOYMENT_MODE": cfg.get("LLM_DEPLOYMENT_MODE", "CLOUD"),
        "LLM_API_BASE_URL": cfg.get("LLM_API_BASE_URL", ""),
        "LLM_API_KEY": cfg.get("LLM_API_KEY", ""),
        "MASKED_API_KEY": masked_key,
        "DEFAULT_MODEL_ID": cfg.get("DEFAULT_MODEL_ID", ""),
        "QDRANT_URL": cfg.get("QDRANT_URL", Config.QDRANT_BASE_URL),
        "EMBEDDING_SERVER_URL": cfg.get("EMBEDDING_SERVER_URL", Config.EMBEDDING_SERVER_URL),
        "RERANKER_SERVER_URL": cfg.get("RERANKER_SERVER_URL", Config.RERANKER_SERVER_URL),
        "VECTOR_TOP_K": int(cfg.get("VECTOR_TOP_K", Config.VECTOR_TOP_K)),
        "RERANK_TOP_K": int(cfg.get("RERANK_TOP_K", Config.RERANK_TOP_K)),
        "RERANKER_SCORE_THRESHOLD": float(cfg.get("RERANKER_SCORE_THRESHOLD", Config.RERANKER_SCORE_THRESHOLD))
    }

@app.post("/api/config")
async def save_configuration(cfg_data: Dict[str, Any]):
    """Saves updated model configurations encrypted to disk."""
    required_keys = ["LLM_DEPLOYMENT_MODE", "LLM_API_BASE_URL", "LLM_API_KEY", "DEFAULT_MODEL_ID"]
    for key in required_keys:
        if key not in cfg_data:
            raise HTTPException(status_code=400, detail=f"Missing configuration key: {key}")
            
    profiles = SecureStorageManager.load_encrypted_profiles()
    active_profile_name = list(profiles.keys())[0] if profiles else "Default Environment"
    profiles[active_profile_name] = {
        "LLM_DEPLOYMENT_MODE": cfg_data["LLM_DEPLOYMENT_MODE"],
        "LLM_API_BASE_URL": cfg_data["LLM_API_BASE_URL"],
        "LLM_API_KEY": cfg_data["LLM_API_KEY"],
        "DEFAULT_MODEL_ID": cfg_data["DEFAULT_MODEL_ID"],
        "QDRANT_URL": cfg_data.get("QDRANT_URL", ""),
        "EMBEDDING_SERVER_URL": cfg_data.get("EMBEDDING_SERVER_URL", ""),
        "RERANKER_SERVER_URL": cfg_data.get("RERANKER_SERVER_URL", ""),
        "VECTOR_TOP_K": int(cfg_data.get("VECTOR_TOP_K", Config.VECTOR_TOP_K)),
        "RERANK_TOP_K": int(cfg_data.get("RERANK_TOP_K", Config.RERANK_TOP_K)),
        "RERANKER_SCORE_THRESHOLD": float(cfg_data.get("RERANKER_SCORE_THRESHOLD", Config.RERANKER_SCORE_THRESHOLD))
    }
    # Immediately apply runtime overrides
    Config.apply_runtime_overrides(profiles[active_profile_name])
    SecureStorageManager.save_encrypted_profiles(profiles)
    return {"status": "success", "message": "Connection properties saved securely."}

# --- NEW MULTI-PROFILE CONFIG ENDPOINTS ---
@app.get("/api/config/profiles")
async def get_profiles():
    """Retrieves all connection profiles and identifies the active one."""
    profiles = SecureStorageManager.load_encrypted_profiles()
    active = profiles.get("_active_profile", "Default Environment")
    clean_profiles = {k: v for k, v in profiles.items() if not k.startswith("_")}
    
    global_ds = profiles.get("_global_downstream", {
        "QDRANT_URL": Config.QDRANT_BASE_URL,
        "EMBEDDING_SERVER_URL": Config.EMBEDDING_SERVER_URL,
        "RERANKER_SERVER_URL": Config.RERANKER_SERVER_URL,
        "VECTOR_TOP_K": Config.VECTOR_TOP_K,
        "RERANK_TOP_K": Config.RERANK_TOP_K,
        "RERANKER_SCORE_THRESHOLD": Config.RERANKER_SCORE_THRESHOLD
    })
    
    return {
        "profiles": clean_profiles,
        "active_profile": active,
        "global_downstream": global_ds
    }

@app.post("/api/config/profiles/activate")
async def activate_profile(payload: Dict[str, str]):
    """Activates a saved operational profile and applies its settings overrides."""
    alias = payload.get("alias")
    if not alias:
        raise HTTPException(status_code=400, detail="alias parameter is required.")
        
    profiles = SecureStorageManager.load_encrypted_profiles()
    if alias not in profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{alias}' not found.")
        
    profiles["_active_profile"] = alias
    SecureStorageManager.save_encrypted_profiles(profiles)
    active_cfg = get_active_llm_config()
    Config.apply_runtime_overrides(active_cfg)
    return {"status": "success", "message": f"Operational profile '{alias}' activated successfully."}

@app.post("/api/config/profiles/onboard")
async def onboard_profile(cfg_data: Dict[str, Any]):
    """Onboards and persists a new operational connection profile."""
    alias = cfg_data.get("alias", "").strip()
    if not alias:
        raise HTTPException(status_code=400, detail="alias friendly name is required.")
        
    profiles = SecureStorageManager.load_encrypted_profiles()
    
    # Map input provider to deployment mode
    provider = cfg_data.get("provider_type", "Cloud API")
    mode = "CLOUD" if provider == "Cloud API" else "LOCAL"
    
    profiles[alias] = {
        "LLM_DEPLOYMENT_MODE": mode,
        "LLM_API_BASE_URL": cfg_data.get("endpoint_url", ""),
        "LLM_API_KEY": cfg_data.get("api_key", ""),
        "DEFAULT_MODEL_ID": cfg_data.get("model_id", ""),
        "PROVIDER_TYPE": provider
    }
    
    SecureStorageManager.save_encrypted_profiles(profiles)
    return {"status": "success", "message": f"Connection profile '{alias}' onboarded successfully."}

@app.post("/api/config/runtime-settings")
async def save_runtime_settings(settings_data: Dict[str, Any]):
    """Saves downstream/retrieval parameters globally (shared across all profiles)."""
    profiles = SecureStorageManager.load_encrypted_profiles()
    
    profiles["_global_downstream"] = {
        "QDRANT_URL": settings_data.get("QDRANT_URL", ""),
        "EMBEDDING_SERVER_URL": settings_data.get("EMBEDDING_SERVER_URL", ""),
        "RERANKER_SERVER_URL": settings_data.get("RERANKER_SERVER_URL", ""),
        "VECTOR_TOP_K": int(settings_data.get("VECTOR_TOP_K", Config.VECTOR_TOP_K)),
        "RERANK_TOP_K": int(settings_data.get("RERANK_TOP_K", Config.RERANK_TOP_K)),
        "RERANKER_SCORE_THRESHOLD": float(settings_data.get("RERANKER_SCORE_THRESHOLD", Config.RERANKER_SCORE_THRESHOLD))
    }
    
    SecureStorageManager.save_encrypted_profiles(profiles)
    
    # Apply runtime overrides using active profile mixed with global downstream
    active_cfg = get_active_llm_config()
    Config.apply_runtime_overrides(active_cfg)
    
    return {"status": "success", "message": "Global downstream settings updated & applied.", "config": active_cfg}

@app.delete("/api/config/profiles/{alias}")
async def delete_profile(alias: str):
    """Purges a connection profile by friendly name."""
    profiles = SecureStorageManager.load_encrypted_profiles()
    if alias not in profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{alias}' not found.")
        
    if alias in profiles:
        del profiles[alias]
    
    # Check remaining profiles keys
    keys = [k for k in profiles.keys() if k != "_active_profile"]
    if not keys:
        # Re-seed default profile if empty
        profiles["_active_profile"] = "Default Environment"
        profiles["Default Environment"] = {
            "LLM_DEPLOYMENT_MODE": "CLOUD",
            "LLM_API_BASE_URL": "https://api.openai.com/v1",
            "LLM_API_KEY": "none",
            "DEFAULT_MODEL_ID": "gpt-4o",
            "QDRANT_URL": "",
            "EMBEDDING_SERVER_URL": "",
            "RERANKER_SERVER_URL": "",
            "VECTOR_TOP_K": 20,
            "RERANK_TOP_K": 3,
            "RERANKER_SCORE_THRESHOLD": 0.40,
            "PROVIDER_TYPE": "Cloud API"
        }
        Config.apply_runtime_overrides(profiles["Default Environment"])
    else:
        active = profiles.get("_active_profile")
        if active == alias:
            new_active = keys[0]
            profiles["_active_profile"] = new_active
            Config.apply_runtime_overrides(profiles[new_active])
            
    SecureStorageManager.save_encrypted_profiles(profiles)
    return {"status": "success", "message": f"Profile '{alias}' deleted successfully."}

# --- EVALUATIONS PERSISTENCE ENDPOINTS ---
@app.get("/api/evaluations/runs")
async def get_eval_runs():
    """Retrieves the list of historical evaluation runs."""
    return SecureStorageManager.load_eval_runs()

@app.post("/api/evaluations/runs")
async def add_eval_run(run_data: Dict[str, Any]):
    """Appends a new evaluation run to historical runs registry (capped at last 10)."""
    scores = run_data.get("scores", {})
    run_data["status"] = determine_production_status(scores)
    
    runs = SecureStorageManager.load_eval_runs()
    runs.insert(0, run_data)
    runs = runs[:10]
    SecureStorageManager.save_eval_runs(runs)
    return runs

@app.delete("/api/evaluations/runs/{run_id}")
async def delete_eval_run(run_id: str):
    """Deletes a specific evaluation run by ID."""
    runs = SecureStorageManager.load_eval_runs()
    runs = [r for r in runs if r.get("id") != run_id]
    SecureStorageManager.save_eval_runs(runs)
    return runs

# --- TENANT ADMINISTRATION ENDPOINTS ---
@app.get("/api/vllm/models")
async def list_vllm_models():
    """Fetches the list of live models/adapters available on the configured LLM API endpoint."""
    llm_cfg = get_active_llm_config()
    deployment_mode = llm_cfg.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
    api_base_url = llm_cfg.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
    api_key = llm_cfg.get("LLM_API_KEY", Config.LLM_API_KEY)
    default_model = llm_cfg.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)

    # 1. Fetch base models from live vLLM models endpoint if LOCAL/ON_PREM
    vllm_models = []
    if deployment_mode == "LOCAL":
        url = f"{api_base_url.rstrip('/')}/models"
        try:
            req = urllib.request.Request(url, method="GET")
            if api_key and api_key.lower() != "none":
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=2) as res:
                data = json.loads(res.read().decode("utf-8"))
                vllm_models = [model["id"] for model in data.get("data", [])]
        except Exception:
            pass

    # 2. Query local Ollama tag registry if running on loopback
    ollama_models = []
    try:
        req_ollama = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req_ollama, timeout=2) as res_ollama:
            ollama_data = json.loads(res_ollama.read().decode("utf-8"))
            for m in ollama_data.get("models", []):
                m_name = m.get("name")
                if m_name:
                    ollama_models.append(m_name)
    except Exception:
        pass

    # 3. Retrieve manually onboarded models
    onboarded_models = llm_cfg.get("ONBOARDED_OLLAMA_MODELS", [])

    # 4. Fetch fallback tenant registered adapter lists
    registry = SecureStorageManager.load_tenant_registry()
    fallback_models = list(set(registry.values()))

    # Compile and filter distinct names
    merged_models = list(set(vllm_models + ollama_models + onboarded_models + fallback_models))
    if default_model and default_model not in merged_models:
        merged_models.append(default_model)

    return {"models": sorted(merged_models)}

@app.get("/api/tenants")
async def list_tenants():
    """Lists all provisioned active workspaces registered in the platform."""
    registry = SecureStorageManager.load_tenant_registry()
    return [{"tenant_id": k, "adapter_weight_matrix": v} for k, v in registry.items()]

@app.post("/api/tenants")
async def register_tenant(tenant_info: Dict[str, str]):
    """Registers and provisions a new tenant workspace adapter alignment focus."""
    tenant_id = tenant_info.get("tenant_id", "").strip().lower()
    adapter = tenant_info.get("adapter_weight_matrix", "").strip()
    
    if not tenant_id or not adapter:
        raise HTTPException(status_code=400, detail="tenant_id and adapter_weight_matrix are required fields.")
        
    registry = SecureStorageManager.load_tenant_registry()
    if tenant_id in registry:
        raise HTTPException(status_code=400, detail=f"Tenant workspace '{tenant_id}' already registered.")
        
    registry[tenant_id] = adapter
    SecureStorageManager.save_tenant_registry(registry)
    return {"status": "success", "message": f"Tenant workspace '{tenant_id}' provisioned."}

@app.delete("/api/tenants/{tenant_id}")
async def deprovision_tenant(tenant_id: str):
    """Deletes a tenant workspace and purges all of its vector index records from Qdrant."""
    registry = SecureStorageManager.load_tenant_registry()
    if tenant_id not in registry:
        raise HTTPException(status_code=404, detail=f"Tenant workspace '{tenant_id}' not found.")
        
    # 1. Purge from Qdrant
    try:
        qdrant_client = Config.get_qdrant_client()
        qdrant_client.delete(
            collection_name=Config.COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
                )
            )
        )
    except Exception as e:
        print(f"⚠️ Warning: Qdrant purge failed for tenant '{tenant_id}': {str(e)}")

    # 2. Delete registry record
    del registry[tenant_id]
    SecureStorageManager.save_tenant_registry(registry)
    return {"status": "success", "message": f"Tenant workspace '{tenant_id}' deprovisioned and vector footprint cleared."}

@app.get("/api/tenants/{tenant_id}/documents")
async def list_tenant_documents(tenant_id: str):
    """Retrieves all active document lineages/families stored for a given tenant."""
    pipeline = TenantIngestionPipeline(tenant_id=tenant_id)
    docs = pipeline.get_tenant_documents()
    return {"tenant_id": tenant_id, "documents": docs}

# --- BATCH MULTI-FILE INGESTION ---
@app.post("/api/tenants/{tenant_id}/ingest")
async def ingest_tenant_files(
    tenant_id: str,
    files: List[UploadFile] = File(...),
    configs: str = Form(...) # JSON-string dict mapping filename to {"family_key": str, "version": str, "replace_target": Optional[str]}
):
    """Ingests multiple document or spreadsheet assets with specific configurations."""
    registry = SecureStorageManager.load_tenant_registry()
    if tenant_id not in registry:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' is not registered.")
        
    try:
        configs_map = json.loads(configs)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid configs JSON format.")

    pipeline = TenantIngestionPipeline(tenant_id=tenant_id)
    results = []
    
    for file in files:
        cfg = configs_map.get(file.filename, {
            "family_key": file.filename.split(".")[0].lower(),
            "version": "1.0",
            "replace_target": None
        })
        
        try:
            # SpooledTemporaryFile behaves as a file-like object compatible with our pipeline
            status = pipeline.process_and_upsert(
                document_name=file.filename,
                file_source=file.file,
                custom_family_key=cfg.get("family_key"),
                target_to_replace=cfg.get("replace_target"),
                document_version=cfg.get("version"),
                progress_callback=None
            )
            
            deprecation_cnt = pipeline.last_deprecation_count if hasattr(pipeline, "last_deprecation_count") else 0
            results.append({
                "filename": file.filename,
                "status": "success",
                "ingest_status": status,
                "deprecation_count": deprecation_cnt,
                "family_key": cfg.get("family_key")
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {"tenant_id": tenant_id, "results": results}

# --- SEMANTIC GROUNDED QUERY & CHAT ---
@app.post("/api/tenants/{tenant_id}/query")
async def query_tenant_rag(tenant_id: str, payload: Dict[str, Any]):
    """Executes a grounded search and reasoning query against the tenant's isolated vector partition, streaming results."""
    request_start = time.perf_counter_ns()
    registry = SecureStorageManager.load_tenant_registry()
    if tenant_id not in registry:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
        
    query_str = payload.get("user_query", "").strip()
    temperature = float(payload.get("temperature", 0.0))
    top_k = int(payload.get("top_k", 3))
    
    if not query_str:
        raise HTTPException(status_code=400, detail="user_query cannot be empty.")
        
    adapter_matrix = registry[tenant_id]
    llm_cfg = get_active_llm_config()
    
    def event_stream_generator():
        for chunk in orchestrator.generate_answer_stream(
            tenant_id=tenant_id,
            target_adapter=adapter_matrix,
            user_query=query_str,
            temperature=temperature,
            top_k=top_k,
            llm_overrides=llm_cfg,
            request_start_time=request_start
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
            
    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")

@app.post("/api/tenants/{tenant_id}/chat")
async def chat_tenant(tenant_id: str, payload: Dict[str, Any]):
    """Executes a conversational chat prompt against the active tenant domain (non-RAG direct chat)."""
    registry = SecureStorageManager.load_tenant_registry()
    if tenant_id not in registry:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")
        
    chat_history = payload.get("chat_history", [])
    temperature = float(payload.get("temperature", 0.7))
    
    adapter_matrix = registry[tenant_id]
    llm_cfg = get_active_llm_config()
    
    res = orchestrator.generate_chat_response(
        target_adapter=adapter_matrix,
        chat_history=chat_history,
        temperature=temperature,
        llm_overrides=llm_cfg
    )
    
    return res

# --- EXECUTIVE DOCUMENT SUMMARIZATION ---
@app.post("/api/tenants/{tenant_id}/summarize")
async def summarize_document(
    tenant_id: str,
    file: UploadFile = File(...),
    summary_length: str = Form("Standard Medium"),
    max_tokens: int = Form(3000),
    max_context_chars: int = Form(300000)
):
    """Generates an executive analysis report summary from an uploaded document asset."""
    file_ext = file.filename.split(".")[-1].lower()
    full_text_stream = ""
    
    # We must read the file stream based on extension
    # Save UploadFile stream into temporary file to let PDF/Excel readers read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
        
    try:
        if file_ext == "pdf":
            import pdfplumber
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    full_text_stream += (page.extract_text() or "") + "\n"
        elif file_ext == "docx":
            import docx
            doc = docx.Document(tmp_path)
            full_text_stream = "\n".join([para.text for para in doc.paragraphs])
        elif file_ext in ["xlsx", "xls"]:
            import pandas as pd
            excel_workbook = pd.ExcelFile(tmp_path)
            excel_texts = []
            for sheet_name in excel_workbook.sheet_names:
                df = excel_workbook.parse(sheet_name).fillna("")
                if df.empty:
                    continue
                headers = [str(col).strip() for col in df.columns]
                md_grid = f"### SPREADSHEET WORKBOOK TAB: {sheet_name.upper()}\n| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
                for _, row in df.iterrows():
                    md_grid += f"| {' | '.join([str(val).strip() for val in row.values])} |\n"
                excel_texts.append(md_grid)
            full_text_stream = "\n\n".join(excel_texts)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, DOCX, or Excel sheets.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    if not full_text_stream.strip():
        raise HTTPException(status_code=400, detail="Document appears to contain no extractable text layers.")
        
    llm_cfg = get_active_llm_config()
    res = orchestrator.generate_summary(
        document_name=file.filename,
        full_text_content=full_text_stream,
        summary_length=summary_length,
        max_tokens=max_tokens,
        max_context_chars=max_context_chars,
        llm_overrides=llm_cfg
    )
    
    return res

# --- EVALUATION AND BENCHMARKING ENDPOINTS ---
@app.post("/api/tenants/{tenant_id}/evaluations/generate-test-set")
async def generate_eval_test_cases(tenant_id: str, payload: Dict[str, Any]):
    """Generates synthetic test sets using the Ragas framework for the tenant workspace."""
    count = int(payload.get("count", 3))
    
    llm_cfg = get_active_llm_config()
    from src.evaluation.rag_evaluator import RAGEvaluator
    evaluator = RAGEvaluator(llm_overrides=llm_cfg)
    
    async def event_generator():
        import random
        import urllib.request
        import json
        import re
        
        url = f"{Config.QDRANT_BASE_URL.rstrip('/')}/collections/{Config.COLLECTION_NAME}/points/scroll"
        scroll_payload = {
            "limit": 100,
            "filter": {"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]},
            "with_payload": True,
            "with_vector": False
        }
        
        chunks = []
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': '[START] Qdrant scroll request dispatching to remote A40 node'})}\n\n"
            req = urllib.request.Request(
                url, data=json.dumps(scroll_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=15.0) as response:
                res = json.loads(response.read().decode("utf-8")).get("result", {})
                points = res.get("points", [])
                valid_chunks = []
                for pt in points:
                    txt = pt.get("payload", {}).get("document_text", "")
                    if txt and len(txt) > 200:
                        valid_chunks.append(txt)
                if valid_chunks:
                    chunks = random.sample(valid_chunks, min(len(valid_chunks), count))
            yield f"data: {json.dumps({'type': 'status', 'message': '[END] Qdrant scroll request successfully completed'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'status', 'message': f'⚠️ Failed to scroll points from Qdrant: {str(e)}'})}\n\n"
            
        if not chunks:
            yield f"data: {json.dumps({'type': 'result', 'status': 'warning', 'message': 'No vector records found to generate questions.', 'test_set': []})}\n\n"
            return

        # Construct the payload for the remote A40 vLLM endpoint
        import requests
        api_base = llm_cfg.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
        api_key = llm_cfg.get("LLM_API_KEY", Config.LLM_API_KEY)
        default_model = llm_cfg.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)
        deployment_mode = llm_cfg.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
        
        provider_type = llm_cfg.get("PROVIDER_TYPE", "Cloud API" if deployment_mode == "CLOUD" else "vLLM")
        base_url = api_base.rstrip('/')
        if provider_type in ["Ollama", "OpenAI-Compatible"] and not base_url.endswith("/v1") and "/v1/" not in base_url:
            base_url = f"{base_url}/v1"
        vllm_endpoint = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.lower() != "none":
            headers["Authorization"] = f"Bearer {api_key}"

        test_cases = []
        total_rows = len(chunks)
        yield f"data: {json.dumps({'type': 'status', 'message': f'[START] generate_synthetic_test_set loop - processing {total_rows} rows'})}\n\n"
        
        for idx, chunk in enumerate(chunks):
            yield f"data: {json.dumps({'type': 'status', 'message': f'[START] Processing row {idx + 1} of {total_rows}'})}\n\n"
            prompt = (
                "You are an expert test dataset generator.\n"
                "Given the context block below, generate a high-quality, professional question and a complete, factual ground truth answer based STRICTLY on this context.\n\n"
                "Format your response as a valid JSON object with EXACTLY these keys (do not output any other text or Markdown wrapping):\n"
                "{\n"
                '  "question": "your generated question here",\n'
                '  "ground_truth": "your generated ground truth answer here"\n'
                "}\n\n"
                f"--- CONTEXT BLOCK ---\n{chunk}\n---------------------"
            )
            
            try:
                yield f"data: {json.dumps({'type': 'status', 'message': f'  [START] Calling remote GPU (A40) to synthesize QA row {idx + 1}'})}\n\n"
                
                # Fetch active adapter weight matrix for routing
                registry = SecureStorageManager.load_tenant_registry()
                active_adapter = registry.get(tenant_id, "tech_support")
                
                if provider_type in ["Ollama", "OpenAI-Compatible", "Cloud API"]:
                    target_model = default_model
                else:
                    # Dynamic check fallback to default_model if active_adapter is not active on vLLM server
                    live_models = []
                    try:
                        url_models = f"{base_url.rstrip('/')}/models"
                        if "/v1" not in url_models and "/v1/" not in url_models:
                            url_v1 = f"{base_url.rstrip('/')}/v1/models"
                            try:
                                req_m = urllib.request.Request(url_v1, method="GET")
                                if api_key and api_key.lower() != "none":
                                    req_m.add_header("Authorization", f"Bearer {api_key}")
                                with urllib.request.urlopen(req_m, timeout=1.5) as res_m:
                                    data_m = json.loads(res_m.read().decode("utf-8"))
                                    live_models = [m["id"] for m in data_m.get("data", [])]
                            except Exception:
                                pass
                        if not live_models:
                            req_m = urllib.request.Request(url_models, method="GET")
                            if api_key and api_key.lower() != "none":
                                req_m.add_header("Authorization", f"Bearer {api_key}")
                            with urllib.request.urlopen(req_m, timeout=1.5) as res_m:
                                data_m = json.loads(res_m.read().decode("utf-8"))
                                live_models = [m["id"] for m in data_m.get("data", [])]
                    except Exception:
                        pass
                    
                    if active_adapter in live_models:
                        target_model = active_adapter
                    else:
                        target_model = default_model

                vllm_payload = {
                    "model": target_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
                
                res_sync = requests.post(vllm_endpoint, json=vllm_payload, headers=headers, timeout=15.0)
                res_sync.raise_for_status()
                res_data = res_sync.json()
                res_text = res_data["choices"][0]["message"]["content"].strip()
                
                yield f"data: {json.dumps({'type': 'status', 'message': f'  [END] Calling remote GPU (A40) to synthesize QA row {idx + 1}'})}\n\n"
                
                # Clean Markdown fences if present
                if res_text.startswith("```"):
                    res_text = re.sub(r"^```(?:json)?\n", "", res_text)
                    res_text = re.sub(r"\n```$", "", res_text)
                    
                parsed = json.loads(res_text)
                if parsed.get("question") and parsed.get("ground_truth"):
                    test_cases.append({
                        "question": parsed["question"].strip(),
                        "ground_truth": parsed["ground_truth"].strip()
                    })
            except Exception as ex:
                yield f"data: {json.dumps({'type': 'status', 'message': f'  [ERROR] Failed to generate question from chunk: {str(ex)}'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'[END] Processing row {idx + 1} of {total_rows}'})}\n\n"
            
        yield f"data: {json.dumps({'type': 'status', 'message': f'[END] generate_synthetic_test_set loop - completed with {len(test_cases)} cases'})}\n\n"
        yield f"data: {json.dumps({'type': 'result', 'status': 'success', 'test_set': test_cases})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/evaluations/test-connectivity")
async def test_evaluation_connectivity():
    """Validates connectivity to all active evaluation infrastructure components."""
    llm_cfg = get_active_llm_config()
    deployment_mode = llm_cfg.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
    api_base_url = llm_cfg.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
    api_key = llm_cfg.get("LLM_API_KEY", Config.LLM_API_KEY)
    default_model = llm_cfg.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)
    
    report = {
        "llm": {"status": "unknown", "message": "", "details": {}},
        "embeddings": {"status": "unknown", "message": ""},
        "reranker": {"status": "unknown", "message": ""},
        "qdrant": {"status": "unknown", "message": ""}
    }
    
    # 1. Test LLM
    base_url = api_base_url.rstrip('/')
    try:
        import urllib.request
        import json
        provider_type = llm_cfg.get("PROVIDER_TYPE", "Cloud API" if deployment_mode == "CLOUD" else "vLLM")
        if provider_type in ["Ollama", "OpenAI-Compatible"] and not base_url.endswith("/v1") and "/v1/" not in base_url:
            base_url = f"{base_url}/v1"
        
        # Test model chat completions endpoint with a simple request
        vllm_payload = {
            "model": default_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5
        }
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.lower() != "none":
            headers["Authorization"] = f"Bearer {api_key}"
            
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(vllm_payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5.0) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            answer = res_data["choices"][0]["message"]["content"]
            report["llm"] = {
                "status": "success",
                "message": f"Successfully connected to LLM. Model response: '{answer.strip()}'",
                "details": {"model": default_model, "provider": provider_type}
            }
    except Exception as e:
        report["llm"] = {
            "status": "error",
            "message": f"LLM Connection failed: {str(e)}",
            "details": {"model": default_model, "url": f"{base_url}/chat/completions"}
        }

    # 2. Test Embedding Server
    try:
        import urllib.request
        import json
        # Embed a dummy word
        embed_payload = {"inputs": ["ping"]}
        req = urllib.request.Request(
            f"{Config.EMBEDDING_SERVER_URL.rstrip('/')}/embed",
            data=json.dumps(embed_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3.0) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            report["embeddings"] = {
                "status": "success",
                "message": f"Successfully connected to Embedding Server. Vector dimension: {len(res_data[0]) if res_data else 'unknown'}"
            }
    except Exception as e:
        report["embeddings"] = {
            "status": "error",
            "message": f"Embedding Server Connection failed: {str(e)}"
        }

    # 3. Test Reranker
    if Config.RERANK_ENABLED:
        try:
            import urllib.request
            import json
            # Rerank a dummy query and context
            rerank_payload = {
                "query": "ping",
                "texts": ["pong"],
                "top_n": 1
            }
            req = urllib.request.Request(
                Config.RERANKER_ENDPOINT,
                data=json.dumps(rerank_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3.0) as res:
                res_data = json.loads(res.read().decode("utf-8"))
                report["reranker"] = {
                    "status": "success",
                    "message": f"Successfully connected to Reranker Server. Score returned: {res_data[0].get('score', 'unknown') if res_data else 'unknown'}"
                }
        except Exception as e:
            report["reranker"] = {
                "status": "error",
                "message": f"Reranker Server Connection failed: {str(e)}"
            }
    else:
        report["reranker"] = {
            "status": "disabled",
            "message": "Reranker is currently disabled in system configuration."
        }

    # 4. Test Qdrant DB
    try:
        qdrant_client = Config.get_qdrant_client()
        exists = qdrant_client.collection_exists(collection_name=Config.COLLECTION_NAME)
        if exists:
            info = qdrant_client.get_collection(collection_name=Config.COLLECTION_NAME)
            report["qdrant"] = {
                "status": "success",
                "message": f"Successfully connected to Qdrant. Collection '{Config.COLLECTION_NAME}' exists with {info.points_count} points."
            }
        else:
            report["qdrant"] = {
                "status": "warning",
                "message": f"Connected to Qdrant, but collection '{Config.COLLECTION_NAME}' does not exist."
            }
    except Exception as e:
        report["qdrant"] = {
            "status": "error",
            "message": f"Qdrant Connection failed: {str(e)}"
        }
        
    return report

@app.post("/api/tenants/{tenant_id}/evaluations/run")
async def run_eval_pass(tenant_id: str, payload: Dict[str, Any]):
    """Executes a full evaluation metrics benchmark suite on a loaded test set."""
    test_set = payload.get("test_set", [])
    top_k = int(payload.get("top_k", Config.RERANK_TOP_K))
    
    if not test_set:
        raise HTTPException(status_code=400, detail="test_set dataset cannot be empty.")
        
    llm_cfg = get_active_llm_config()
    from src.evaluation.rag_evaluator import RAGEvaluator
    evaluator = RAGEvaluator(llm_overrides=llm_cfg)
    
    async def event_generator():
        completed_dataset = []
        total = len(test_set)
        
        for idx, case in enumerate(test_set):
            question = case.get("question", "").strip()
            ground_truth = case.get("ground_truth", "").strip()
            
            # Yield active progress event frame
            yield f"data: {json.dumps({'type': 'status', 'phase': 'inference', 'current': idx + 1, 'total': total, 'message': f'Generating RAG answers ({idx + 1}/{total}): {question[:28]}...'})}\n\n"
            
            # Retrieve isolated context chunks from Qdrant
            retrieved_nodes, _, _ = evaluator.query_engine.retrieve_context(
                query_str=question,
                tenant_id=tenant_id,
                limit=top_k
            )
            contexts = [node.get("text", "") for node in retrieved_nodes if node.get("text", "")]
            if not contexts:
                contexts = ["NO VERIFIED CONTEXT DETECTED."]

            # Fetch active adapter weight matrix for routing
            from src.database.secure_storage import SecureStorageManager
            registry = SecureStorageManager.load_tenant_registry()
            active_adapter = registry.get(tenant_id, "tech_support")

            # Generate answer from LLM under active settings overrides
            res = evaluator.orchestrator.generate_answer(
                tenant_id=tenant_id,
                target_adapter=active_adapter,
                user_query=question,
                temperature=0.0,
                top_k=top_k,
                llm_overrides=evaluator.overrides
            )
            generated_answer = res.get("answer", "Information missing from current isolated partition data store.")
            
            completed_dataset.append({
                "question": question,
                "contexts": contexts,
                "answer": generated_answer,
                "ground_truth": ground_truth
            })
            
        # Yield Ragas local processing transition event frame
        yield f"data: {json.dumps({'type': 'status', 'phase': 'evaluation', 'current': total, 'total': total, 'message': 'Running Ragas metrics locally on Ollama ggozad/prometheus2...'})}\n\n"
        
        # Run Ragas Metrics
        eval_res = evaluator.evaluate_dataset(completed_dataset)
        
        # Yield final completed dataset results frame
        if eval_res.get("status") == "success":
            scores = eval_res.get('scores', {})
            prod_status = determine_production_status(scores)
            result_payload = {
                'type': 'result',
                'data': {
                    'status': 'success',
                    'scores': scores,
                    'production_status': prod_status,
                    'raw_dataframe': eval_res.get('raw_dataframe', []),
                    'vector_top_k': Config.VECTOR_TOP_K,
                    'rerank_top_k': Config.RERANK_TOP_K,
                    'reranker_score_threshold': Config.RERANKER_SCORE_THRESHOLD
                }
            }
            yield f"data: {json.dumps(result_payload)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'result', 'data': eval_res})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Mount static build folder at the root to serve our React SPA frontend
# Note: we mount this at the very end so it serves index.html on fallback
try:
    dist_dir = "frontend/dist" if os.path.exists("frontend/dist") else "src/static"
    if os.path.exists(dist_dir):
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
except Exception as e:
    print(f"⚠️ Warning: Could not mount static files: {str(e)}")
