import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List
from src.config import Config

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ContextOrchestrator:
    def __init__(self):
        pass

    def _get_query_vector(self, query_text: str) -> List[float]:
        payload = {"inputs": [query_text]}
        try:
            req = urllib.request.Request(
                Config.TEI_ENDPOINT, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))[0]
        except Exception:
            return [0.0] * Config.VECTOR_DIMENSION

    def _retrieve_context_from_qdrant(self, tenant_id: str, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        url = f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}/collections/{Config.COLLECTION_NAME}/points/search"
        query_vector = self._get_query_vector(query_text)

        payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
            "filter": {"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]}
        }
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8")).get("result", [])
        except Exception:
            return []

    def generate_answer(self, tenant_id: str, target_adapter: str, user_query: str, temperature: float, top_k: int, llm_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        points = self._retrieve_context_from_qdrant(tenant_id, user_query, top_k)
        
        context_chunks = []
        citations = []
        for pt in points:
            score = pt.get("score", 0.0)
            payload_data = pt.get("payload", {})
            text_content = payload_data.get("document_text", "")
            if text_content:
                context_chunks.append(text_content)
                citations.append({
                    "source": payload_data.get("source_file", "unknown_source.pdf"),
                    "page": payload_data.get("page_number", "N/A"),
                    "score": round(score, 4)
                })

        flat_context = "\n---\n".join(context_chunks) if context_chunks else "NO VERIFIED CONTEXT DETECTED."
        
        system_instruction = (
            "You are an elite enterprise core engine agent operating within a secure multi-tenant architecture.\n"
            f"Your current operational domain context isolation group is: [TENANT: {tenant_id.upper()}]\n\n"
            "STRICT GROUNDING REGIME:\n"
            "1. Synthesize your answer using ONLY the explicit text context blocks provided below.\n"
            "2. If context is insufficient, state: 'Information missing from current isolated partition data store.'\n"
            "3. Do NOT utilize background parametric weights to extrapolate claims.\n\n"
            f"--- ISOLATED CONTEXT PAYLOAD ---\n{flat_context}\n--------------------------------"
        )

        overrides = llm_overrides or {}
        deployment_mode = overrides.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
        api_base_url = overrides.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
        api_key = overrides.get("LLM_API_KEY", Config.LLM_API_KEY)
        default_model = overrides.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)

        vllm_endpoint = f"{api_base_url}/chat/completions"
        target_model = default_model if deployment_mode == "CLOUD" else target_adapter

        vllm_payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            "temperature": temperature,
            "max_tokens": 1024
        }

        request_headers = {"Content-Type": "application/json"}
        if api_key and api_key.lower() != "none":
            request_headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(vllm_endpoint, data=json.dumps(vllm_payload).encode("utf-8"), headers=request_headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as response:
                vllm_res = json.loads(response.read().decode("utf-8"))
                return {
                    "status": "success",
                    "answer": vllm_res["choices"][0]["message"]["content"],
                    "citations": citations,
                    "latency_seconds": round(time.time() - start_time, 3),
                    "token_metrics": vllm_res.get("usage", {})
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                parsed_error = json.loads(error_body)
                error_msg = parsed_error.get("error", {}).get("message", error_body)
            except Exception:
                error_msg = error_body
            return {"status": "error", "message": f"Backend Engine Rejection [{e.code}]: {error_msg}"}
        except Exception as e:
            return {"status": "error", "message": f"Modular Inference routing path failure: {str(e)}"}

    def generate_summary(self, document_name: str, full_text_content: str, summary_length: str, llm_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        system_instruction = (
            "You are an expert corporate analyst.\n"
            "Your task is to analyze the provided document and generate a highly professional executive summary.\n"
            "Structure your output using clean Markdown formatting with the following sections:\n"
            "1. 📌 **Executive Overview** (A high-level 3-sentence summary of the document's purpose)\n"
            "2. 🔑 **Key Takeaways & Core Pillars** (Bullet points outlining the most critical rules, numbers, or terms)\n"
            "3. ⚠️ **Critical Warnings / Action Items** (Any deadlines, penalties, or constraints mentioned)\n\n"
            f"Enforce a target length constraint of: {summary_length} details."
        )

        overrides = llm_overrides or {}
        deployment_mode = overrides.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
        api_base_url = overrides.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
        api_key = overrides.get("LLM_API_KEY", Config.LLM_API_KEY)
        default_model = overrides.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)

        vllm_endpoint = f"{api_base_url}/chat/completions"
        target_model = default_model if deployment_mode == "CLOUD" else "tech_support"

        vllm_payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Document Title: {document_name}\n\nFull Document Content:\n{full_text_content}"}
            ],
            "temperature": 0.2,
            "max_tokens": 1500
        }

        request_headers = {"Content-Type": "application/json"}
        if api_key and api_key.lower() != "none":
            request_headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(vllm_endpoint, data=json.dumps(vllm_payload).encode("utf-8"), headers=request_headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as response:
                vllm_res = json.loads(response.read().decode("utf-8"))
                return {
                    "status": "success",
                    "summary": vllm_res["choices"][0]["message"]["content"],
                    "latency_seconds": round(time.time() - start_time, 3),
                    "token_metrics": vllm_res.get("usage", {})
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            return {"status": "error", "message": f"Summarization Engine Rejection [{e.code}]: {error_body}"}
        except Exception as e:
            return {"status": "error", "message": f"Summarization pipeline failure: {str(e)}"}

    def generate_chat_response(self, target_adapter: str, chat_history: List[Dict[str, str]], temperature: float, llm_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        system_instruction = "You are SandyGPT, a helpful corporate assistant operating in the Enterprise-RAG environment."

        overrides = llm_overrides or {}
        deployment_mode = overrides.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
        api_base_url = overrides.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
        api_key = overrides.get("LLM_API_KEY", Config.LLM_API_KEY)
        default_model = overrides.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)

        vllm_endpoint = f"{api_base_url}/chat/completions"
        target_model = default_model if deployment_mode == "CLOUD" else target_adapter

        vllm_payload = {
            "model": target_model,
            "messages": [{"role": "system", "content": system_instruction}] + chat_history,
            "temperature": temperature,
            "max_tokens": 1024
        }

        request_headers = {"Content-Type": "application/json"}
        if api_key and api_key.lower() != "none":
            request_headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(vllm_endpoint, data=json.dumps(vllm_payload).encode("utf-8"), headers=request_headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as response:
                vllm_res = json.loads(response.read().decode("utf-8"))
                return {
                    "status": "success",
                    "answer": vllm_res["choices"][0]["message"]["content"],
                    "latency_seconds": round(time.time() - start_time, 3),
                    "token_metrics": vllm_res.get("usage", {})
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                parsed_error = json.loads(error_body)
                error_msg = parsed_error.get("error", {}).get("message", error_body)
            except Exception:
                error_msg = error_body
            return {"status": "error", "message": f"Backend Engine Rejection [{e.code}]: {error_msg}"}
        except Exception as e:
            return {"status": "error", "message": f"Modular Inference routing path failure: {str(e)}"}