import os
import json
from cryptography.fernet import Fernet
from src.config import Config

KEY_FILE = ".enc_key"
DATA_FILE = "model_profiles.enc"
REGISTRY_FILE = "tenant_registry.json"

class SecureStorageManager:
    @staticmethod
    def _get_or_create_key() -> bytes:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            # Set restrictive file permissions (read/write by owner only)
            with open(KEY_FILE, "wb") as f:
                f.write(key)
            try:
                os.chmod(KEY_FILE, 0o600)
            except Exception:
                pass
            return key

    @classmethod
    def save_encrypted_profiles(cls, profiles: dict):
        key = cls._get_or_create_key()
        fernet = Fernet(key)
        serialized = json.dumps(profiles).encode("utf-8")
        encrypted = fernet.encrypt(serialized)
        with open(DATA_FILE, "wb") as f:
            f.write(encrypted)

    @classmethod
    def load_encrypted_profiles(cls) -> dict:
        if not os.path.exists(DATA_FILE):
            # Seed with default config from env variables as the first profile
            default_profile = {
                "_active_profile": "Default Environment",
                "Default Environment": {
                    "LLM_DEPLOYMENT_MODE": Config.LLM_DEPLOYMENT_MODE,
                    "LLM_API_BASE_URL": Config.LLM_API_BASE_URL,
                    "LLM_API_KEY": Config.LLM_API_KEY,
                    "DEFAULT_MODEL_ID": Config.DEFAULT_MODEL_ID,
                    "QDRANT_URL": Config.QDRANT_BASE_URL or "",
                    "EMBEDDING_SERVER_URL": Config.EMBEDDING_SERVER_URL or "",
                    "RERANKER_SERVER_URL": Config.RERANKER_SERVER_URL or "",
                    "VECTOR_TOP_K": Config.VECTOR_TOP_K,
                    "RERANK_TOP_K": Config.RERANK_TOP_K,
                    "RERANKER_SCORE_THRESHOLD": Config.RERANKER_SCORE_THRESHOLD,
                    "PROVIDER_TYPE": "Cloud API" if Config.LLM_DEPLOYMENT_MODE == "CLOUD" else "vLLM"
                }
            }
            cls.save_encrypted_profiles(default_profile)
            return default_profile
        try:
            key = cls._get_or_create_key()
            fernet = Fernet(key)
            with open(DATA_FILE, "rb") as f:
                encrypted = f.read()
            decrypted = fernet.decrypt(encrypted)
            return json.loads(decrypted.decode("utf-8"))
        except Exception:
            return {}

    @classmethod
    def save_tenant_registry(cls, registry: dict):
        with open(REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=4)

    @classmethod
    def load_tenant_registry(cls) -> dict:
        if not os.path.exists(REGISTRY_FILE):
            # Seed and save default settings
            default_registry = {
                "finance_reasoning": "finance_reasoning",
                "tech_support": "tech_support"
            }
            cls.save_tenant_registry(default_registry)
            return default_registry
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "finance_reasoning": "finance_reasoning",
                "tech_support": "tech_support"
            }

    EVAL_RUNS_FILE = "eval_runs.json"

    @classmethod
    def save_eval_runs(cls, runs: list):
        with open(cls.EVAL_RUNS_FILE, "w") as f:
            json.dump(runs, f, indent=4)

    @classmethod
    def load_eval_runs(cls) -> list:
        if not os.path.exists(cls.EVAL_RUNS_FILE):
            return []
        try:
            with open(cls.EVAL_RUNS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
