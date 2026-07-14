import os

class Config:
    """Central System Configuration Blueprint."""
    # Read the local .env file if it exists and load variables into the environment
    _env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "r") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("#") and "=" in clean_line:
                    key, val = clean_line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

    # --- VECTOR DATABASE CONFIG ---
    QDRANT_URL = os.getenv("QDRANT_URL", None)
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_BASE_URL = QDRANT_URL or f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tenant_knowledge_base")
    
    # --- TEI EMBEDDING CONFIG ---
    EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8090")
    TEI_ENDPOINT = f"{EMBEDDING_SERVER_URL.rstrip('/')}/embed"
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))
    TEI_TIMEOUT = int(os.getenv("TEI_TIMEOUT", "60"))

    # --- RERANKER CONFIG ---
    RERANKER_SERVER_URL = os.getenv("RERANKER_SERVER_URL", "http://localhost:8081")
    RERANKER_ENDPOINT = f"{RERANKER_SERVER_URL.rstrip('/')}/rerank"
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "True").lower() == "true"
    RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "60"))
    RERANKER_SCORE_THRESHOLD = float(os.getenv("RERANKER_SCORE_THRESHOLD", "0.40"))
    VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

    # --- MODULAR BACKEND ENGINE PARAMETERS ---
    LLM_DEPLOYMENT_MODE = os.getenv("LLM_DEPLOYMENT_MODE", "ON_PREM").upper()
    LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "none")
    DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "gemini-1.5-pro")

    # --- PARAMETRIC RECOVERY RULES ---
    CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "1200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    NOISE_THRESHOLD_GATE = int(os.getenv("NOISE_THRESHOLD_GATE", "60"))
    CLIENT_BATCH_LIMIT = int(os.getenv("CLIENT_BATCH_LIMIT", "32"))

    @classmethod
    def get_qdrant_client(cls):
        from qdrant_client import QdrantClient
        if cls.QDRANT_URL:
            return QdrantClient(url=cls.QDRANT_URL, timeout=60.0)
        return QdrantClient(host=cls.QDRANT_HOST, port=cls.QDRANT_PORT, timeout=60.0)

    @classmethod
    def apply_runtime_overrides(cls, overrides: dict):
        if "QDRANT_URL" in overrides and overrides["QDRANT_URL"]:
            cls.QDRANT_URL = overrides["QDRANT_URL"]
            cls.QDRANT_BASE_URL = overrides["QDRANT_URL"]
        if "EMBEDDING_SERVER_URL" in overrides and overrides["EMBEDDING_SERVER_URL"]:
            cls.EMBEDDING_SERVER_URL = overrides["EMBEDDING_SERVER_URL"]
            cls.TEI_ENDPOINT = f"{cls.EMBEDDING_SERVER_URL.rstrip('/')}/embed"
            settings.EMBEDDING_API_URL = cls.EMBEDDING_SERVER_URL
        if "RERANKER_SERVER_URL" in overrides and overrides["RERANKER_SERVER_URL"]:
            cls.RERANKER_SERVER_URL = overrides["RERANKER_SERVER_URL"]
            cls.RERANKER_ENDPOINT = f"{cls.RERANKER_SERVER_URL.rstrip('/')}/rerank"
        if "VECTOR_TOP_K" in overrides:
            try:
                cls.VECTOR_TOP_K = int(overrides["VECTOR_TOP_K"])
            except ValueError:
                pass
        if "RERANK_TOP_K" in overrides:
            try:
                cls.RERANK_TOP_K = int(overrides["RERANK_TOP_K"])
            except ValueError:
                pass
        if "RERANKER_SCORE_THRESHOLD" in overrides:
            try:
                cls.RERANKER_SCORE_THRESHOLD = float(overrides["RERANKER_SCORE_THRESHOLD"])
            except ValueError:
                pass

# Create settings alias for compatibility with modules
settings = Config
settings.EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", Config.EMBEDDING_SERVER_URL)