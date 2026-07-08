import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List
from src.config import Config
from src.database.query_engine import MultiTenantQueryEngine

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ContextOrchestrator:
    def __init__(self):
        self.query_engine = MultiTenantQueryEngine()

    def generate_answer(self, tenant_id: str, target_adapter: str, user_query: str, temperature: float, top_k: int, llm_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        points = self.query_engine.retrieve_context(
            query_str=user_query,
            tenant_id=tenant_id,
            limit=top_k
        )
        
        context_chunks = []
        citations = []
        for pt in points:
            score = pt.get("score", 0.0)
            text_content = pt.get("text", "")
            if text_content:
                context_chunks.append(text_content)
                citations.append({
                    "source": pt.get("source_file", "unknown_source.pdf"),
                    "page": pt.get("page_number", "N/A"),
                    "score": round(score, 4)
                })

        flat_context = "\n---\n".join(context_chunks) if context_chunks else "NO VERIFIED CONTEXT DETECTED."
        
        system_instruction = (
            "You are a strict QA engine. Your current tenant domain is: " f"[{tenant_id.upper()}]\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the query directly, concisely, and factually, relying ONLY on the isolated context blocks below.\n"
            "2. Do NOT use conversational preambles (e.g., 'Based on the context...', 'According to the document...').\n"
            "3. Do NOT extrapolate or assume any facts not directly written in the context.\n"
            "4. If the provided context is insufficient to answer the query, respond ONLY with: "
            "'Information missing from current isolated partition data store.'\n\n"
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

    def generate_summary(self, document_name: str, full_text_content: str, summary_length: str, max_tokens: int = 3000, max_context_chars: int = 300000, llm_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        truncated_content = full_text_content[:max_context_chars]
        if len(full_text_content) > max_context_chars:
            print(f"⚠️ Truncating document context from {len(full_text_content)} to {max_context_chars} characters.")

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
                {"role": "user", "content": f"Document Title: {document_name}\n\nFull Document Content:\n{truncated_content}"}
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens
        }

        request_headers = {"Content-Type": "application/json"}
        if api_key and api_key.lower() != "none":
            request_headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(vllm_endpoint, data=json.dumps(vllm_payload).encode("utf-8"), headers=request_headers, method="POST")
            with urllib.request.urlopen(req, timeout=300) as response:
                vllm_res = json.loads(response.read().decode("utf-8"))
                summary_text = vllm_res["choices"][0]["message"]["content"]
                if len(full_text_content) > max_context_chars:
                    summary_text += f"\n\n---\n*⚠️ Note: The document was truncated to the first {max_context_chars:,} characters to comply with the model's context window size limit.*"
                return {
                    "status": "success",
                    "summary": summary_text,
                    "latency_seconds": round(time.time() - start_time, 3),
                    "token_metrics": vllm_res.get("usage", {})
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            
            # Smart Auto-Recovery for Context Length Errors
            try:
                error_json = json.loads(error_body)
                err_msg = error_json.get("error", {}).get("message", "")
                
                if "maximum context length" in err_msg.lower() or "context length" in err_msg.lower() or "too long" in err_msg.lower():
                    import re
                    max_ctx_match = re.search(r"maximum context length is (\d+)", err_msg)
                    input_tok_match = re.search(r"(?:contains at least|resulted in|prompt contains) (\d+)", err_msg)
                    
                    if max_ctx_match and input_tok_match:
                        max_ctx = int(max_ctx_match.group(1))
                        input_tok = int(input_tok_match.group(1))
                        # Use 150 tokens safety margin for chat template/formatting overhead
                        safe_max_tokens = max_ctx - input_tok - 150
                        
                        if safe_max_tokens >= 256:
                            print(f"🔄 Auto-Recovery: Prompt tokens ({input_tok}) + requested output ({max_tokens}) exceeds max context ({max_ctx}). Retrying with adjusted max_tokens={safe_max_tokens}...")
                            vllm_payload["max_tokens"] = safe_max_tokens
                            
                            try:
                                req_retry = urllib.request.Request(vllm_endpoint, data=json.dumps(vllm_payload).encode("utf-8"), headers=request_headers, method="POST")
                                with urllib.request.urlopen(req_retry, timeout=300) as response_retry:
                                    vllm_res_retry = json.loads(response_retry.read().decode("utf-8"))
                                    summary_text_retry = vllm_res_retry["choices"][0]["message"]["content"]
                                    if len(full_text_content) > max_context_chars:
                                        summary_text_retry += f"\n\n---\n*⚠️ Note: The document was truncated to the first {max_context_chars:,} characters to comply with the model's context window size limit.*"
                                    return {
                                        "status": "success",
                                        "summary": summary_text_retry,
                                        "latency_seconds": round(time.time() - start_time, 3),
                                        "token_metrics": vllm_res_retry.get("usage", {})
                                    }
                            except Exception as retry_err:
                                print(f"⚠️ Retry with adjusted max_tokens failed: {str(retry_err)}. Falling back to context shrinking.")
                    
                    # Fallback recursive shrink if numbers couldn't be parsed, if input tokens alone exceed limit, or if the token-adjusted retry failed
                    shrunk_chars = int(max_context_chars * 0.6)
                    print(f"🔄 Auto-Recovery: Input context too large or retry failed. Retrying with reduced input limit {shrunk_chars} chars...")
                    return self.generate_summary(
                        document_name=document_name,
                        full_text_content=full_text_content,
                        summary_length=summary_length,
                        max_tokens=max_tokens,
                        max_context_chars=shrunk_chars,
                        llm_overrides=llm_overrides
                    )
            except Exception as recovery_error:
                print(f"⚠️ Auto-recovery attempt failed: {str(recovery_error)}")

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