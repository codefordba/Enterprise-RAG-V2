import hashlib
import uuid
import pandas as pd
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
from src.config import Config
from src.processing.layout_parser import LayoutAwareParser
from src.processing.semantic_splitter import SemanticProcessingEngine

class TenantIngestionPipeline:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        self.semantic_engine = SemanticProcessingEngine()

    def _check_content_hash_exists(self, content_hash: str) -> bool:
        """Uses the Qdrant SDK to check if a document with the same content hash already exists for this tenant."""
        try:
            points, _ = self.client.scroll(
                collection_name=Config.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=self.tenant_id)),
                        FieldCondition(key="content_hash", match=MatchValue(value=content_hash))
                    ]
                ),
                limit=1,
                with_payload=False
            )
            return len(points) > 0
        except Exception:
            return False

    def _check_filename_exists(self, filename: str) -> bool:
        """Checks if a document with the same filename already exists for this tenant in Qdrant."""
        try:
            points, _ = self.client.scroll(
                collection_name=Config.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=self.tenant_id)),
                        FieldCondition(key="source_file", match=MatchValue(value=filename))
                    ]
                ),
                limit=1,
                with_payload=False
            )
            return len(points) > 0
        except Exception:
            return False

    def get_tenant_documents(self) -> List[str]:
        """Retrieves a list of all unique active document families ingested for this tenant in Qdrant."""
        try:
            points, _ = self.client.scroll(
                collection_name=Config.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=self.tenant_id)),
                        FieldCondition(key="is_latest", match=MatchValue(value=True))
                    ]
                ),
                limit=1000,
                with_payload=["document_family"],
                with_vector=False
            )
            unique_families = set()
            for pt in points:
                fam = pt.payload.get("document_family")
                if fam:
                    unique_families.add(fam)
            return sorted(list(unique_families))
        except Exception as e:
            print(f"⚠️ Warning: Failed to retrieve tenant document families: {str(e)}")
            return []

    def _deprecate_existing_versions(self, document_family: str):
        """Finds all active chunks for this family and tenant, and flips their is_latest flag to False."""
        try:
            points, _ = self.client.scroll(
                collection_name=Config.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=self.tenant_id)),
                        FieldCondition(key="document_family", match=MatchValue(value=document_family)),
                        FieldCondition(key="is_latest", match=MatchValue(value=True))
                    ]
                ),
                limit=1000,
                with_payload=False
            )
            
            if points:
                point_ids = [pt.id for pt in points]
                self.client.set_payload(
                    collection_name=Config.COLLECTION_NAME,
                    payload={"is_latest": False},
                    points=point_ids
                )
                print(f"🔄 Deprecated {len(point_ids)} previous chunks for family '{document_family}'.")
        except Exception as e:
            print(f"⚠️ Warning: Failed to deprecate previous document versions: {str(e)}")

    def process_and_upsert(self, document_name: str, file_source: Any, custom_family_key: str = None, target_to_replace: str = None, document_version: str = None, progress_callback: Any = None) -> str:
        """Parses layout elements, chunks them semantically, and upserts them into Qdrant with lineage and state flags."""
        file_ext = document_name.split(".")[-1].lower()
        elements = []

        try:
            if progress_callback:
                progress_callback("Initializing parser and extracting text/tables...", 0.1)

            # 1. Parse document into structured text/table elements
            if file_ext == "pdf":
                elements = LayoutAwareParser.extract_elements(file_source)
            elif file_ext in ["xlsx", "xls"]:
                excel_workbook = pd.ExcelFile(file_source)
                for sheet_idx, sheet_name in enumerate(excel_workbook.sheet_names, 1):
                    df = excel_workbook.parse(sheet_name).fillna("")
                    if df.empty:
                        continue
                    headers = [str(col).strip() for col in df.columns]
                    m_str = f"### SPREADSHEET WORKBOOK TAB: {sheet_name.upper()}\n| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
                    for _, row in df.iterrows():
                        m_str += f"| {' | '.join([str(val).strip() for val in row.values])} |\n"
                    elements.append({
                        "content": m_str,
                        "type": "table",
                        "page": sheet_idx
                    })
            else:
                return "unsupported_format"

            if not elements:
                return "no_valid_content"

            if progress_callback:
                progress_callback("Calculating content hash and checking for duplicates...", 0.3)

            # 2. Compute content hash over all extracted elements for deduplication
            full_text_stream = "".join([e["content"] for e in elements])
            doc_hash = hashlib.sha256(full_text_stream.encode("utf-8")).hexdigest()

            # Normalize inputs
            final_version = (document_version or doc_hash[:8]).strip()
            
            # Determine document family lineage key
            if custom_family_key:
                doc_family = custom_family_key.strip().lower().replace(" ", "_")
            else:
                # Use target_to_replace if available, else fall back to parsing text
                if target_to_replace and target_to_replace != "New Document":
                    doc_family = target_to_replace
                else:
                    prose_elements = [e for e in elements if e["type"] == "prose"]
                    first_prose_text = prose_elements[0]["content"].strip() if prose_elements else ""
                    first_line = first_prose_text.split("\n")[0] if first_prose_text else "unassigned_family"
                    doc_family = "".join([c for c in first_line if c.isalnum() or c.isspace()]).strip().lower().replace(" ", "_")[:32]
                    if not doc_family:
                        doc_family = "unassigned_family"

            # 3. Check for existence in the database before uploading
            filename_exists = self._check_filename_exists(document_name)
            family_exists = doc_family in self.get_tenant_documents()
            hash_exists = self._check_content_hash_exists(doc_hash)

            if hash_exists:
                # Content already exists in database (exact duplicate, regardless of name)
                return "skipped_duplicate"

            # This is an update if the filename or family already exists
            is_update = filename_exists or family_exists or (target_to_replace is not None and target_to_replace != "New Document")
            target_family = target_to_replace if (target_to_replace and target_to_replace != "New Document") else doc_family

            if progress_callback:
                progress_callback(f"Phase 1: Querying database to deprecate old versions of family '{target_family}'...", 0.5)

            # 4. Phase 1: Atomically deprecate active versions of this family
            self._deprecate_existing_versions(document_family=target_family)

            if progress_callback:
                progress_callback("Phase 2: Slicing text semantically and fetching vector embeddings...", 0.7)

            # 5. Phase 2: Segment elements semantically and fetch embeddings with uuid.uuid5 deterministic IDs
            qdrant_points = self.semantic_engine.generate_points(
                raw_elements=elements,
                tenant_id=self.tenant_id,
                filename=document_name,
                document_family=target_family,
                content_hash=doc_hash,
                document_version=final_version
            )

            if not qdrant_points:
                return "no_valid_content"

            if progress_callback:
                progress_callback("Finalizing payload structures and upserting vector points to Qdrant...", 0.9)

            # 6. Upsert semantic points to Qdrant (which will now default to is_latest: True)
            self.client.upsert(
                collection_name=Config.COLLECTION_NAME,
                points=qdrant_points
            )

            return "updated_version" if is_update else "ingested_successfully"

        except Exception as e:
            raise RuntimeError(f"Ingestion pipeline critical fault: {str(e)}")