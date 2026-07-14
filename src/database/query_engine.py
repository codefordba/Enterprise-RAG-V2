import logging
import time
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config import settings
from src.processing.semantic_splitter import TEIEmbeddingClient

logger = logging.getLogger("QueryEngine")

class MultiTenantQueryEngine:
    def __init__(self) -> None:
        """Initializes the decoupled embedding client and the Qdrant cluster connection."""
        self.collection_name = settings.COLLECTION_NAME

    @property
    def qdrant(self):
        return settings.get_qdrant_client()

    @property
    def embedder(self):
        return TEIEmbeddingClient(settings.EMBEDDING_API_URL)

    def retrieve_context(self, query_str: str, tenant_id: str, limit: int = 3) -> tuple:
        """Vectorizes a search query and returns isolated semantic matches for the specified tenant alongside query telemetry."""
        retrieval_telemetry = {
            "embedding_ms": 0.0,
            "qdrant_ms": 0.0,
            "rerank_ms": 0.0
        }
        try:
            # 1. Convert the natural language string into a 1024-dimension float vector via TEI
            embed_start = time.perf_counter_ns()
            query_vector = self.embedder.embed_query(query_str)
            retrieval_telemetry["embedding_ms"] = round((time.perf_counter_ns() - embed_start) / 1_000_000.0, 2)
            
            # 2. Query a larger set of candidates for reranking
            qdrant_limit = getattr(settings, "VECTOR_TOP_K", max(limit * 5, 20))
            qdrant_start = time.perf_counter_ns()
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="tenant_id", 
                            match=MatchValue(value=tenant_id)
                        ),
                        FieldCondition(
                            key="is_latest", 
                            match=MatchValue(value=True)
                        )
                    ]
                ),
                limit=qdrant_limit,
                with_payload=True
            )
            retrieval_telemetry["qdrant_ms"] = round((time.perf_counter_ns() - qdrant_start) / 1_000_000.0, 2)
            
            # 3. Parse out structural data records (parent blocks if available) and deduplicate by parent_id
            retrieved_contexts = []
            seen_parent_ids = set()
            
            for point in response.points:
                parent_id = point.payload.get("parent_id")
                parent_text = point.payload.get("parent_text")
                
                if parent_id and parent_text:
                    # Parent-child structure detected
                    if parent_id in seen_parent_ids:
                        continue
                    seen_parent_ids.add(parent_id)
                    text_content = parent_text
                else:
                    # Fallback for standard legacy documents
                    text_content = point.payload.get("document_text", "")
                    
                if text_content:
                    retrieved_contexts.append({
                        "score": point.score,
                        "text": text_content,
                        "chunk_type": point.payload.get("chunk_type", "prose"),
                        "page_number": point.payload.get("page_number", 0),
                        "source_file": point.payload.get("source_file", "unknown")
                    })
            
            # 4. If Reranker is enabled and reachable, perform Cross-Encoder reranking
            reranker_success = False
            if getattr(settings, "RERANK_ENABLED", True) and retrieved_contexts:
                rerank_start = time.perf_counter_ns()
                try:
                    import requests
                    rerank_payload = {
                        "query": query_str,
                        "texts": [c["text"] for c in retrieved_contexts]
                    }
                    rerank_url = getattr(settings, "RERANKER_ENDPOINT", "http://localhost:8081/rerank")
                    timeout = getattr(settings, "RERANK_TIMEOUT", 60)
                    
                    res = requests.post(rerank_url, json=rerank_payload, timeout=timeout)
                    if res.status_code == 200:
                        rerank_results = res.json()
                        # Map reranker scores back to candidate blocks
                        for r_item in rerank_results:
                            idx = r_item["index"]
                            if idx < len(retrieved_contexts):
                                retrieved_contexts[idx]["score"] = r_item["score"]
                        
                        # Sort by cross-encoder score descending
                        retrieved_contexts.sort(key=lambda x: x["score"], reverse=True)
                        reranker_success = True
                        logger.info("Successfully re-ranked candidate context blocks using Cross-Encoder.")
                    else:
                        logger.warning(f"Reranker backend returned error code {res.status_code}. Falling back to standard vector search ranking.")
                except Exception as rerank_err:
                    logger.warning(f"Cross-Encoder reranking failed: {str(rerank_err)}. Falling back to standard vector search ranking.")
                finally:
                    retrieval_telemetry["rerank_ms"] = round((time.perf_counter_ns() - rerank_start) / 1_000_000.0, 2)

            # 5. Apply score threshold cutoff and slice down to the requested limit size
            if reranker_success:
                threshold = getattr(settings, "RERANKER_SCORE_THRESHOLD", 0.35)
                above_threshold = [c for c in retrieved_contexts if c.get("score", 0.0) >= threshold]
                below_threshold = [c for c in retrieved_contexts if c.get("score", 0.0) < threshold]
                
                final_limit = getattr(settings, "RERANK_TOP_K", limit)
                accepted_contexts = above_threshold[:final_limit]
                discarded_contexts = above_threshold[final_limit:] + below_threshold
            else:
                final_limit = getattr(settings, "RERANK_TOP_K", limit)
                accepted_contexts = retrieved_contexts[:final_limit]
                discarded_contexts = retrieved_contexts[final_limit:]
            
            logger.info(f"Successfully retrieved {len(accepted_contexts)} deduplicated context nodes for tenant '{tenant_id}'.")
            return accepted_contexts, retrieval_telemetry, discarded_contexts

        except Exception as e:
            logger.error(f"Critical failure during multi-tenant vector retrieval pass: {str(e)}")
            raise RuntimeError("Context retrieval failure encountered.")
