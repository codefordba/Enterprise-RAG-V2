import json
import uuid
import hashlib
import urllib.request
from typing import List, Dict, Any
from src.config import Config

class TenantIngestionPipeline:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    def _get_tei_embeddings(self, texts: List[str]) -> List[List[float]]:
        payload = {"inputs": texts}
        try:
            req = urllib.request.Request(
                Config.TEI_ENDPOINT, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=Config.TEI_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Embedding compute core connection timed out: {str(e)}")

    def _check_content_hash_exists(self, content_hash: str) -> bool:
        url = f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}/collections/{Config.COLLECTION_NAME}/points/scroll"
        payload = {
            "limit": 1,
            "filter": {
                "must": [
                    {"key": "tenant_id", "match": {"value": self.tenant_id}},
                    {"key": "content_hash", "match": {"value": content_hash}}
                ]
            }
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req) as res:
                points = json.loads(res.read().decode("utf-8")).get("result", {}).get("points", [])
                return len(points) > 0
        except Exception:
            return False

    def _purge_by_lineage(self, document_family: str, source_file: str):
        url = f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}/collections/{Config.COLLECTION_NAME}/points/delete"
        payload = {
            "filter": {
                "should": [
                    {
                        "must": [
                            {"key": "tenant_id", "match": {"value": self.tenant_id}},
                            {"key": "document_family", "match": {"value": document_family}}
                        ]
                    },
                    {
                        "must": [
                            {"key": "tenant_id", "match": {"value": self.tenant_id}},
                            {"key": "source_file", "match": {"value": source_file}}
                        ]
                    }
                ]
            }
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req) as response:
                json.loads(response.read().decode("utf-8"))
        except Exception:
            pass

    def process_and_upsert(self, document_name: str, raw_pages: List[Dict[str, Any]], custom_family_key: str = None):
        full_text_stream = "".join([p.get("text", "") + p.get("table_markdown", "") for p in raw_pages])
        doc_hash = hashlib.sha256(full_text_stream.encode("utf-8")).hexdigest()
        
        if self._check_content_hash_exists(doc_hash):
            return "skipped_duplicate"

        if custom_family_key:
            doc_family = custom_family_key.strip().lower().replace(" ", "_")
        else:
            first_page_text = raw_pages[0].get("text", "").strip() if raw_pages else ""
            first_line = first_page_text.split("\n")[0] if first_page_text else "unassigned_family"
            doc_family = "".join([c for c in first_line if c.isalnum() or c.isspace()]).strip().lower().replace(" ", "_")[:32]
            if not doc_family:
                doc_family = "unassigned_family"

        self._purge_by_lineage(document_family=doc_family, source_file=document_name)

        max_size = Config.CHUNK_MAX_SIZE
        overlap = Config.CHUNK_OVERLAP
        qdrant_points = []

        for page in raw_pages:
            p_num = page.get("page_number", 0)
            text_pool = page.get("text", "").strip()
            table_pool = page.get("table_markdown", "").strip()

            if table_pool:
                text_pool += f"\n\n### STRUCTURAL DATA RECOVERY MATRIX:\n{table_pool}\n"
            if len(text_pool) < Config.NOISE_THRESHOLD_GATE:
                continue

            start_pointer = 0
            while start_pointer < len(text_pool):
                end_pointer = start_pointer + max_size
                chunk_slice = text_pool[start_pointer:end_pointer]
                
                vector = self._get_tei_embeddings([chunk_slice])[0]
                
                qdrant_points.append({
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "tenant_id": self.tenant_id,
                        "source_file": document_name,
                        "document_family": doc_family,
                        "content_hash": doc_hash,
                        "page_number": p_num,
                        "document_text": chunk_slice
                    }
                })
                start_pointer += (max_size - overlap)

        if not qdrant_points:
            return "no_valid_content"

        upsert_url = f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}/collections/{Config.COLLECTION_NAME}/points?wait=true"
        req = urllib.request.Request(
            upsert_url, data=json.dumps({"points": qdrant_points}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="PUT"
        )
        with urllib.request.urlopen(req) as response:
            json.loads(response.read().decode("utf-8"))
        
        return "ingested_successfully"