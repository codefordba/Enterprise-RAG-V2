import json
import urllib.request
import urllib.error
from qdrant_client import QdrantClient
from src.config import Config

class QdrantManager:
    def __init__(self):
        self.base_url = f"{Config.QDRANT_BASE_URL.rstrip('/')}/collections/{Config.COLLECTION_NAME}"
        self.client = Config.get_qdrant_client()

    def initialize_schema(self):
        self.initialize_collection()

    def initialize_collection(self):
        print(f"🛠️ Configuring schema collection framework for: '{Config.COLLECTION_NAME}'...")
        payload = {
            "vectors": {"size": Config.VECTOR_DIMENSION, "distance": "Cosine"},
            "hnsw_config": {"m": 0, "payload_m": 16}
        }
        try:
            req = urllib.request.Request(
                self.base_url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="PUT"
            )
            with urllib.request.urlopen(req) as response:
                res = json.loads(response.read().decode("utf-8"))
                print(f"✅ Collection initialized: {res.get('status')}")
            self._create_tenant_payload_index()
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8")
            if "already exists" in error_msg:
                print(f"ℹ️ Collection '{Config.COLLECTION_NAME}' already exist in cluster layout.")
            else:
                print(f"❌ Schema execution error: {error_msg}")

    def _create_tenant_payload_index(self):
        try:
            res = self.client.create_payload_index(
                collection_name=Config.COLLECTION_NAME,
                field_name="tenant_id",
                field_schema={"type": "keyword", "is_tenant": True}
            )
            print(f"✅ HNSW tenant segmentation schema mapped: {res.status}")
        except Exception as e:
            print(f"❌ Index configuration fault: {str(e)}")

    def upsert_chunks(self, points):
        self.client.upsert(
            collection_name=Config.COLLECTION_NAME,
            points=points
        )

if __name__ == "__main__":
    manager = QdrantManager()
    manager.initialize_schema()