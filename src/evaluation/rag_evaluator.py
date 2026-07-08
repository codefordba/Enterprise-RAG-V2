# src/evaluation/rag_evaluator.py
import re
import json
import time
import urllib.request
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from src.processing.semantic_splitter import TEIEmbeddingClient
from src.database.query_engine import MultiTenantQueryEngine
from src.generation.orchestrator import ContextOrchestrator
from src.config import settings

class RagasTEIEmbeddings(Embeddings):
    """Langchain adapter for the custom TEIEmbeddingClient to be compatible with Ragas."""
    def __init__(self, tei_client: TEIEmbeddingClient):
        self.tei_client = tei_client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.tei_client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.tei_client.embed_query(text)


class RAGEvaluator:
    def __init__(self, llm_overrides: Dict[str, Any] = None):
        self.overrides = llm_overrides or {}
        self.deployment_mode = self.overrides.get("LLM_DEPLOYMENT_MODE", settings.LLM_DEPLOYMENT_MODE)
        self.api_base_url = self.overrides.get("LLM_API_BASE_URL", settings.LLM_API_BASE_URL)
        self.api_key = self.overrides.get("LLM_API_KEY", settings.LLM_API_KEY)
        self.default_model = self.overrides.get("DEFAULT_MODEL_ID", settings.DEFAULT_MODEL_ID)

        # Use active API Key or a dummy fallback to satisfy internal validators if using local mode
        eval_api_key = self.api_key if (self.api_key and self.api_key.lower() != "none") else "dummy_key"

        # Initialize the LangChain ChatOpenAI client wrapper pointing to our active endpoint
        self.eval_llm = ChatOpenAI(
            model=self.default_model,
            openai_api_key=eval_api_key,
            openai_api_base=self.api_base_url,
            temperature=0.0
        )

        # Initialize the custom Embedding client adapter
        tei_client = TEIEmbeddingClient(settings.EMBEDDING_API_URL)
        self.eval_embeddings = RagasTEIEmbeddings(tei_client)

        # Initialize core pipeline components
        self.query_engine = MultiTenantQueryEngine()
        self.orchestrator = ContextOrchestrator()

    def run_rag_inference(self, tenant_id: str, test_cases: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Runs the RAG pipeline over a set of input test cases to generate contexts and answers.
        """
        completed_dataset = []
        for case in test_cases:
            question = case.get("question", "").strip()
            ground_truth = case.get("ground_truth", "").strip()
            
            if not question:
                continue

            # 1. Retrieve isolated context chunks from Qdrant
            retrieved_nodes = self.query_engine.retrieve_context(
                query_str=question,
                tenant_id=tenant_id,
                limit=top_k
            )
            contexts = [node.get("text", "") for node in retrieved_nodes if node.get("text", "")]
            
            if not contexts:
                contexts = ["NO VERIFIED CONTEXT DETECTED."]

            # 2. Generate answer from LLM under active settings overrides
            res = self.orchestrator.generate_answer(
                tenant_id=tenant_id,
                target_adapter="tech_support",
                user_query=question,
                temperature=0.0,
                top_k=top_k,
                llm_overrides=self.overrides
            )
            
            generated_answer = res.get("answer", "Information missing from current isolated partition data store.")
            
            completed_dataset.append({
                "question": question,
                "contexts": contexts,
                "answer": generated_answer,
                "ground_truth": ground_truth
            })
            
        return completed_dataset

    def evaluate_dataset(self, rag_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes evaluation via Ragas metrics on pre-compiled generation data.
        """
        if not rag_data:
            return {"status": "error", "message": "No test cases provided for evaluation."}

        try:
            formatted_data = {
                "question": [item["question"] for item in rag_data],
                "contexts": [item["contexts"] for item in rag_data],
                "answer": [item["answer"] for item in rag_data],
                "ground_truth": [item["ground_truth"] for item in rag_data]
            }

            dataset = Dataset.from_dict(formatted_data)
            metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall]
            
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=metrics_list,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings
            )
            
            df_results = evaluation_result.to_pandas()
            
            # Clean up NaN values in raw dataframe to avoid JSON serialization faults
            df_results_cleaned = df_results.fillna(0.0)

            # Safely extract average scores from the pandas DataFrame to handle failed metrics gracefully
            scores = {
                "faithfulness": float(df_results["faithfulness"].mean()) if "faithfulness" in df_results.columns else 0.0,
                "answer_relevance": float(df_results["answer_relevancy"].mean()) if "answer_relevancy" in df_results.columns else 0.0,
                "context_precision": float(df_results["context_precision"].mean()) if "context_precision" in df_results.columns else 0.0,
                "context_recall": float(df_results["context_recall"].mean()) if "context_recall" in df_results.columns else 0.0
            }
            
            # Handle NaN values specifically in scores
            for key in scores:
                import math
                if math.isnan(scores[key]):
                    scores[key] = 0.0

            return {
                "status": "success",
                "scores": scores,
                "raw_dataframe": df_results_cleaned.to_dict(orient="records")
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Ragas evaluation engine fault: {str(e)}"}

    def generate_synthetic_test_set(self, tenant_id: str, count: int = 5) -> List[Dict[str, str]]:
        """
        Retrieves raw chunks from the tenant's isolated vector store and synthesizes QA test cases using the LLM.
        """
        url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/collections/{settings.COLLECTION_NAME}/points/scroll"
        payload = {
            "limit": count * 2,
            "filter": {"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]},
            "with_payload": True,
            "with_vector": False
        }
        
        chunks = []
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                res = json.loads(response.read().decode("utf-8")).get("result", {})
                points = res.get("points", [])
                for pt in points:
                    txt = pt.get("payload", {}).get("document_text", "")
                    if txt and len(txt) > 200:
                        chunks.append(txt)
        except Exception as e:
            print(f"⚠️ Failed to scroll points from Qdrant: {str(e)}")
            
        if not chunks:
            return []

        test_cases = []
        for chunk in chunks[:count]:
            prompt = (
                "You are an expert test dataset generator.\n"
                "Given the context block below, generate a high-quality, professional question and a complete, factual ground truth answer based STRICTLY on this context.\n\n"
                "Format your response as a valid JSON object with EXACTLY these keys (do not output any other text or Markdown wrapping):\n"
                "{\n"
                '  "question": "your generated question here",\n'
                '  "ground_truth": "your generated ground truth answer here"\n'
                "}\n\n"
                f"--- CONTEXT BLOCK ---\n{chunk}\n---------------------"
            )
            
            try:
                response = self.eval_llm.invoke(prompt)
                res_text = response.content.strip()
                
                # Clean Markdown fences if present
                if res_text.startswith("```"):
                    res_text = re.sub(r"^```(?:json)?\n", "", res_text)
                    res_text = re.sub(r"\n```$", "", res_text)
                    
                parsed = json.loads(res_text)
                if parsed.get("question") and parsed.get("ground_truth"):
                    test_cases.append({
                        "question": parsed["question"].strip(),
                        "ground_truth": parsed["ground_truth"].strip()
                    })
            except Exception as ex:
                print(f"⚠️ Failed to generate question from chunk: {str(ex)}")
                
        return test_cases
