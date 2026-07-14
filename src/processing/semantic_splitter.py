# src/processing/semantic_splitter.py
import uuid
import logging
import requests
import tiktoken
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        # Restrict client batches to 16 elements to match TEI's CPU backend capacity
        MAX_SERVER_BATCH = getattr(settings, "CLIENT_BATCH_LIMIT", 16)
        if MAX_SERVER_BATCH > 16:
            MAX_SERVER_BATCH = 16
            
        timeout = getattr(settings, "TEI_TIMEOUT", 120)
        all_embeddings = []
        
        try:
            import time
            for i in range(0, len(texts), MAX_SERVER_BATCH):
                sub_batch = texts[i : i + MAX_SERVER_BATCH]
                
                max_retries = 3
                backoff = 1.0
                batch_response = None
                for attempt in range(max_retries):
                    try:
                        response = requests.post(self.api_url, json={"inputs": sub_batch}, timeout=timeout)
                        if response.status_code == 200:
                            batch_response = response.json()
                            break
                        elif response.status_code in [429, 500, 502, 503, 504]:
                            time.sleep(backoff * (attempt + 1))
                        else:
                            response.raise_for_status()
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        time.sleep(backoff * (attempt + 1))
                        
                if batch_response is None:
                    raise RuntimeError("Embedding backend container returned invalid status code.")
                all_embeddings.extend(batch_response)
                
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
        
    def _count_tokens(self, text: str) -> int:
        """Counts exact tokens in a text using the cl100k_base tokenizer (GPT-4 / Gemini standard)."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback estimation if tiktoken fails
            return len(text.split()) * 4 // 3

    def _split_into_parent_blocks(self, raw_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compiles elements, splits them by Markdown headings (#, ##), and returns parent context blocks."""
        # 1. Compile the elements into a single Markdown document with embedded page break comments
        full_doc_md = ""
        for el in raw_elements:
            content = el["content"]
            page = el.get("page", 1)
            
            if el["type"] == "page_markdown":
                page_break_marker = f"\n\n---\n<!-- PAGE BREAK Page {page} -->\n\n"
                if full_doc_md:
                    full_doc_md += page_break_marker + content
                else:
                    full_doc_md += content
            else:
                # Direct tables or sheet tabs from Excel/xlsx
                table_marker = f"\n\n---\n<!-- PAGE BREAK Page {page} -->\n\n"
                if full_doc_md:
                    full_doc_md += table_marker + content
                else:
                    full_doc_md += content
                    
        # 2. Parse lines and split into logical sections on Heading `#` and `##` boundaries
        lines = full_doc_md.split("\n")
        sections = []
        current_section_lines = []
        current_page = 1
        
        for line in lines:
            # Update current page tracking
            if "<!-- PAGE BREAK Page" in line:
                try:
                    parts = line.split("Page")
                    page_num_str = parts[1].split("-->")[0].strip()
                    current_page = int(page_num_str)
                except Exception:
                    pass
                current_section_lines.append(line)
                continue
                
            # Split on # and ##
            if line.startswith(("# ", "## ")):
                if current_section_lines:
                    sections.append({
                        "content": "\n".join(current_section_lines).strip(),
                        "page": current_page
                    })
                    current_section_lines = []
                current_section_lines.append(line)
            else:
                current_section_lines.append(line)
                
        # Flush the final section
        if current_section_lines:
            sections.append({
                "content": "\n".join(current_section_lines).strip(),
                "page": current_page
            })
            
        # 3. Refine section blocks into size-capped Parent Blocks (1500 to 2048 tokens)
        parent_blocks = []
        # Target: ~1500 to 2048 tokens limit.
        # Tables must never be broken to keep their logical matrix rows intact.
        for sec in sections:
            content = sec["content"]
            page = sec["page"]
            
            # Check if it contains a table (Markdown tables contain "|---")
            is_table = "|---" in content
            token_count = self._count_tokens(content)
            
            if token_count <= 2048 or is_table:
                parent_blocks.append({
                    "content": content,
                    "page": page,
                    "type": "table" if is_table else "prose"
                })
            else:
                # Split large prose parent block recursively (2048 tokens is ~8000 characters)
                # We target chunks of size ~1500 tokens (~6000 chars) with ~200 tokens overlap (~800 chars)
                prose_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=6000,
                    chunk_overlap=800,
                    length_function=len
                )
                sub_chunks = prose_splitter.split_text(content)
                for chunk in sub_chunks:
                    parent_blocks.append({
                        "content": chunk,
                        "page": page,
                        "type": "prose"
                    })
                    
        return parent_blocks

    def _slice_parent_to_child_chunks(self, parent_text: str) -> List[str]:
        """Slices a parent text block into overlapping child windows of ~256 to 512 tokens."""
        # 256 tokens is roughly 1000 characters. 512 is 2000.
        # We configure RecursiveCharacterTextSplitter to produce chunks around 1024 characters (~256 tokens)
        # with an overlap of 256 characters (~64 tokens) to ensure search target coverage.
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            length_function=len
        )
        return child_splitter.split_text(parent_text)

    def generate_points(self, raw_elements: List[Dict[str, Any]], tenant_id: str, filename: str, document_family: str = None, content_hash: str = None, document_version: str = None) -> List[PointStruct]:
        """Slices structural elements into parent-child blocks and fetches embeddings for children via TEI."""
        processed_points = []
        
        # 1. Structural splitting into logical Parent blocks
        parent_blocks = self._split_into_parent_blocks(raw_elements)
        
        # 2. Slice Parent blocks into Child chunks and organize indexing queues
        flat_child_chunks = []
        child_metadata = []
        
        for parent_idx, parent in enumerate(parent_blocks):
            parent_text = parent["content"]
            parent_page = parent["page"]
            parent_type = parent["type"]
            parent_id = str(uuid.uuid4())
            
            # Slice parent block into overlapping child blocks
            child_chunks = self._slice_parent_to_child_chunks(parent_text)
            
            for child_text in child_chunks:
                flat_child_chunks.append(child_text)
                child_metadata.append({
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "child_text": child_text,
                    "page": parent_page,
                    "type": parent_type
                })
                
        if not flat_child_chunks:
            return []
            
        # 3. Safely request vector embeddings for all Child chunks in batches
        logger.info(f"Generating embeddings for {len(flat_child_chunks)} child search targets...")
        embeddings = self.embedder.embed_documents(flat_child_chunks)
        
        # 4. Construct PointStruct payloads for Qdrant
        for idx, meta in enumerate(child_metadata):
            # Deterministic ID generation to support upserts and prevent conflicts
            seed_str = f"{tenant_id}_{document_family or 'unassigned_family'}_{document_version or '1.0'}_{idx}_{meta['child_text'][:30]}"
            deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_str))
            
            payload = {
                "tenant_id": tenant_id,
                "parent_id": meta["parent_id"],
                "parent_text": meta["parent_text"],
                "page_content": meta["child_text"],             # Standard child target text
                "document_text": meta["child_text"],           # Compatibility key
                "chunk_type": meta["type"],
                "page_number": meta["page"],
                "source_file": filename,
                "document_family": document_family or "unassigned_family",
                "document_version": document_version or "1.0",
                "content_hash": content_hash or "",
                "is_latest": True
            }
            
            processed_points.append(
                PointStruct(
                    id=deterministic_id,
                    vector=embeddings[idx],
                    payload=payload
                )
            )
            
        logger.info(f"Generated {len(processed_points)} vector points for Qdrant ingestion.")
        return processed_points