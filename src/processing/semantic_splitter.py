# src/processing/semantic_splitter.py
import uuid
import logging
import requests
from typing import List, Dict, Any
from langchain_experimental.text_splitter import SemanticChunker
from qdrant_client.models import PointStruct
from src.config import settings

logger = logging.getLogger("SemanticSplitter")

class TEIEmbeddingClient:
    """Production-ready client wrapper targeting the decoupled Rust Inference container."""
    def __init__(self, api_url: str):
        if api_url.endswith("/embed"):
            self.api_url = api_url
        else:
            self.api_url = f"{api_url.rstrip('/')}/embed"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Sends a text array to the TEI container, safely batched to avoid server limits."""
        # DEFENSIVE FIX: Restrict client batches to 32 elements to comply with TEI safety valves
        MAX_SERVER_BATCH = 32
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), MAX_SERVER_BATCH):
                sub_batch = texts[i : i + MAX_SERVER_BATCH]
                response = requests.post(self.api_url, json={"inputs": sub_batch}, timeout=30)
                response.raise_for_status()
                all_embeddings.extend(response.json())
                
            return all_embeddings
        except Exception as e:
            logger.error(f"Network error routing vectors to TEI microservice: {str(e)}")
            raise RuntimeError("Embedding backend container unreachable or payload invalid.")

    def embed_query(self, text: str) -> List[float]:
        """Vectorizes standalone incoming search prompts."""
        return self.embed_documents([text])[0]


class SemanticProcessingEngine:
    def __init__(self) -> None:
        logger.info(f"Connecting to standalone embedding container at: {settings.EMBEDDING_API_URL}")
        self.embedder = TEIEmbeddingClient(settings.EMBEDDING_API_URL)
        self.splitter = SemanticChunker(self.embedder, min_chunk_size=100)

    def generate_points(self, raw_elements: List[Dict[str, Any]], tenant_id: str, filename: str) -> List[PointStruct]:
        """Slices structural elements and fetches high-density vector profiles via REST."""
        processed_points = []
        
        for element in raw_elements:
            chunks = self.splitter.split_text(element["content"])
            if not chunks:
                continue
                
            # Safely request vectors via our batch-protected client wrapper
            embeddings = self.embedder.embed_documents(chunks)
            
            for idx, chunk_str in enumerate(chunks):
                processed_points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embeddings[idx],
                        payload={
                            "tenant_id": tenant_id,
                            "document_text": chunk_str,
                            "chunk_type": element["type"],
                            "page_number": element["page"],
                            "source_file": filename
                        }
                    )
                )
        return processed_points