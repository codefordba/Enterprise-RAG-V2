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
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tenant_knowledge_base")
    
    # --- TEI EMBEDDING CONFIG ---
    TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://localhost:8080/embed")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))
    TEI_TIMEOUT = int(os.getenv("TEI_TIMEOUT", "60"))

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

# Create settings alias for compatibility with modules
settings = Config
# TEIEmbeddingClient expects the base URL to construct '/embed' path
settings.EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8080")