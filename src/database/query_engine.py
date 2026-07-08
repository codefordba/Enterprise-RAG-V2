import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config import settings
from src.processing.semantic_splitter import TEIEmbeddingClient

logger = logging.getLogger("QueryEngine")

class MultiTenantQueryEngine:
    def __init__(self) -> None:
        """Initializes the decoupled embedding client and the Qdrant cluster connection."""
        self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.embedder = TEIEmbeddingClient(settings.EMBEDDING_API_URL)
        self.collection_name = settings.COLLECTION_NAME

    def retrieve_context(self, query_str: str, tenant_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Vectorizes a search query and returns isolated semantic matches for the specified tenant."""
        try:
            # 1. Convert the natural language string into a 1024-dimension float vector via TEI
            query_vector = self.embedder.embed_query(query_str)
            
            # 2. PRODUCTION FIX: Employing the modern query_points method instead of deprecated .search()
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
                limit=limit,
                with_payload=True
            )
            
            # 3. Parse out structural data records out of the returned points array
            retrieved_contexts = []
            for point in response.points:
                retrieved_contexts.append({
                    "score": point.score,
                    "text": point.payload.get("document_text", ""),
                    "chunk_type": point.payload.get("chunk_type", "prose"),
                    "page_number": point.payload.get("page_number", 0),
                    "source_file": point.payload.get("source_file", "unknown")
                })
                
            logger.info(f"Successfully retrieved {len(retrieved_contexts)} context nodes for tenant '{tenant_id}'.")
            return retrieved_contexts

        except Exception as e:
            logger.error(f"Critical failure during multi-tenant vector retrieval pass: {str(e)}")
            raise RuntimeError("Context retrieval failure encountered.")
