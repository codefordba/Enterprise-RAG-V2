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

    def generate_answer(self, tenant_id: str, target_adapter: str, user_query: str, temperature: float, top_k: int, llm_overrides: Dict[str, Any] = None, request_start_time: float = None) -> Dict[str, Any]:
        pre_start = request_start_time or time.perf_counter_ns()
        pre_processing_ms = round((time.perf_counter_ns() - pre_start) / 1_000_000.0, 2)

        try:
            points, retrieval_telemetry, discarded_points = self.query_engine.retrieve_context(
                query_str=user_query,
                tenant_id=tenant_id,
                limit=top_k
            )
            
            context_chunks = []
            citations = []
            context_nodes = []
            
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
                    context_nodes.append({
                        "source": pt.get("source_file", "unknown_source.pdf"),
                        "page": pt.get("page_number", "N/A"),
                        "score": round(score, 4),
                        "text": text_content,
                        "selected": True
                    })

            for pt in discarded_points:
                score = pt.get("score", 0.0)
                text_content = pt.get("text", "")
                if text_content:
                    context_nodes.append({
                        "source": pt.get("source_file", "unknown_source.pdf"),
                        "page": pt.get("page_number", "N/A"),
                        "score": round(score, 4),
                        "text": text_content,
                        "selected": False
                    })

            flat_context = "\n---\n".join(context_chunks) if context_chunks else "NO VERIFIED CONTEXT DETECTED."

            ref_docs_str = ""
            if points:
                for idx, pt in enumerate(points):
                    title = pt.get("source_file", "unknown_source.pdf")
                    source = pt.get("source_file", "unknown_source.pdf")
                    content = pt.get("text", "").strip()
                    ref_docs_str += (
                        f"Document {idx + 1}\n\n"
                        f"Title:\n{title}\n\n"
                        f"Source:\n{source}\n\n"
                        f"Content:\n{content}\n\n"
                        "--------------------------------\n\n"
                    )
            else:
                ref_docs_str = "No verified context detected.\n\n--------------------------------\n\n"

            system_instruction = (
                "### Role\n"
                "You are a professional HR Support Analyst operating inside an isolated enterprise data partition.\n"
                f"Your current tenant domain scope is explicitly restricted to: [{tenant_id.upper()}]\n"
                "Your objective is to answer user queries accurately, directly, and concisely using the provided reference documents.\n\n"
                "### Strict Grounding Boundaries\n"
                "- State only the raw facts explicitly mentioned or directly answered in the reference documents.\n"
                "- Do NOT use logical transition words, summaries, or deductive connectors (e.g., do not say 'Therefore', 'However', 'Consequently', or 'As a result').\n"
                "- Completely eliminate explanatory bridges. Write only direct, independent factual declarations.\n"
                "- Output ONLY the pure extracted facts. Never include conversational preambles or comment on prompt structure.\n"
                "- Do NOT print document indices, source labels, file names, or citation markers (e.g., do not append '(Source: Document X)').\n"
                "- If the documents do not provide an answer, reply exactly with: "
                "\"The requested information is not available in the provided documents.\"\n\n"
                "### Response Formatting\n"
                "- State the direct answer clearly in your very first sentence.\n"
                "- Present multi-step rules or structured guidelines using clean Markdown bullet points."
            )

            user_payload = (
                "### REFERENCE DOCUMENTS\n\n"
                f"{ref_docs_str.strip()}\n\n"
                "### QUESTION\n\n"
                f"{user_query.strip()}\n\n"
                "### GROUNDING INSTRUCTIONS\n\n"
                "Answer the question using ONLY the reference documents above.\n"
                "Before producing the final response, internally verify that every statement is directly supported by the text.\n"
                "Remove any unsupported claims. If the information is missing from the documents, reply exactly with:\n"
                "\"The requested information is not available in the provided documents.\"\n\n"
                "### FINAL ANSWER\n"
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ]

            overrides = llm_overrides or {}
            deployment_mode = overrides.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
            api_base_url = overrides.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
            api_key = overrides.get("LLM_API_KEY", Config.LLM_API_KEY)
            default_model = overrides.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)

            provider_type = overrides.get("PROVIDER_TYPE", "Cloud API" if deployment_mode == "CLOUD" else "vLLM")
            base_url = api_base_url.rstrip('/')
            if provider_type in ["Ollama", "OpenAI-Compatible"] and not base_url.endswith("/v1") and "/v1/" not in base_url:
                base_url = f"{base_url}/v1"
            vllm_endpoint = f"{base_url}/chat/completions"
            if provider_type in ["Ollama", "OpenAI-Compatible", "Cloud API"]:
                target_model = default_model
            else:
                live_models = []
                try:
                    url = f"{base_url.rstrip('/')}/models"
                    if "/v1" not in url and "/v1/" not in url:
                        url_v1 = f"{base_url.rstrip('/')}/v1/models"
                        try:
                            req = urllib.request.Request(url_v1, method="GET")
                            if api_key and api_key.lower() != "none":
                                req.add_header("Authorization", f"Bearer {api_key}")
                            with urllib.request.urlopen(req, timeout=1.5) as res:
                                data = json.loads(res.read().decode("utf-8"))
                                live_models = [m["id"] for m in data.get("data", [])]
                        except Exception:
                            pass
                    if not live_models:
                        req = urllib.request.Request(url, method="GET")
                        if api_key and api_key.lower() != "none":
                            req.add_header("Authorization", f"Bearer {api_key}")
                        with urllib.request.urlopen(req, timeout=1.5) as res:
                            data = json.loads(res.read().decode("utf-8"))
                            live_models = [m["id"] for m in data.get("data", [])]
                except Exception:
                    pass
                
                if target_adapter in live_models:
                    target_model = target_adapter
                else:
                    target_model = default_model

            vllm_payload = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1024
            }
            
            # Enforce strict local structural bounds to prevent token drift
            if provider_type not in ["Cloud API"]:
                vllm_payload["repetition_penalty"] = 1.05

            request_headers = {"Content-Type": "application/json"}
            if api_key and api_key.lower() != "none":
                request_headers["Authorization"] = f"Bearer {api_key}"

            ttft_ms = 0.0
            generation_ms = 0.0
            response_text = ""
            token_metrics = {}

            try:
                import httpx
                vllm_payload["stream"] = True
                llm_start = time.perf_counter_ns()
                first_chunk_received = False
                
                with httpx.stream("POST", vllm_endpoint, json=vllm_payload, headers=request_headers, timeout=15.0) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not first_chunk_received:
                            ttft_ms = round((time.perf_counter_ns() - llm_start) / 1_000_000.0, 2)
                            first_chunk_received = True
                        
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    response_text += delta["content"]
                            except Exception:
                                pass
                generation_ms = round((time.perf_counter_ns() - llm_start) / 1_000_000.0, 2)
                
                prompt_tokens = len(system_instruction.split()) + len(user_payload.split())
                completion_tokens = len(response_text.split())
                token_metrics = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            except Exception as stream_err:
                import requests
                vllm_payload["stream"] = False
                llm_start = time.perf_counter_ns()
                res_sync = requests.post(vllm_endpoint, json=vllm_payload, headers=request_headers, timeout=15.0)
                res_sync.raise_for_status()
                res_data = res_sync.json()
                response_text = res_data["choices"][0]["message"]["content"]
                generation_ms = round((time.perf_counter_ns() - llm_start) / 1_000_000.0, 2)
                ttft_ms = round(generation_ms * 0.15, 2)
                token_metrics = res_data.get("usage", {})

            telemetry = {
                "pre_processing_ms": pre_processing_ms,
                "embedding_ms": retrieval_telemetry["embedding_ms"],
                "qdrant_ms": retrieval_telemetry["qdrant_ms"],
                "rerank_ms": retrieval_telemetry["rerank_ms"],
                "ttft_ms": ttft_ms,
                "generation_ms": generation_ms
            }

            return {
                "status": "success",
                "answer": response_text.strip(),
                "citations": citations,
                "latency_seconds": round((time.perf_counter_ns() - pre_start) / 1_000_000_000.0, 3),
                "token_metrics": token_metrics,
                "telemetry": telemetry,
                "raw_context_block": flat_context,
                "context_nodes": context_nodes
            }
        except Exception as e:
            return {"status": "error", "message": f"Modular Inference routing path failure: {str(e)}"}

    def generate_answer_stream(self, tenant_id: str, target_adapter: str, user_query: str, temperature: float, top_k: int, llm_overrides: Dict[str, Any] = None, request_start_time: float = None):
        pre_start = request_start_time or time.perf_counter_ns()
        pre_processing_ms = round((time.perf_counter_ns() - pre_start) / 1_000_000.0, 2)

        try:
            points, retrieval_telemetry, discarded_points = self.query_engine.retrieve_context(
                query_str=user_query,
                tenant_id=tenant_id,
                limit=top_k
            )
            
            context_chunks = []
            citations = []
            context_nodes = []
            
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
                    context_nodes.append({
                        "source": pt.get("source_file", "unknown_source.pdf"),
                        "page": pt.get("page_number", "N/A"),
                        "score": round(score, 4),
                        "text": text_content,
                        "selected": True
                    })

            for pt in discarded_points:
                score = pt.get("score", 0.0)
                text_content = pt.get("text", "")
                if text_content:
                    context_nodes.append({
                        "source": pt.get("source_file", "unknown_source.pdf"),
                        "page": pt.get("page_number", "N/A"),
                        "score": round(score, 4),
                        "text": text_content,
                        "selected": False
                    })

            flat_context = "\n---\n".join(context_chunks) if context_chunks else "NO VERIFIED CONTEXT DETECTED."

            ref_docs_str = ""
            if points:
                for idx, pt in enumerate(points):
                    title = pt.get("source_file", "unknown_source.pdf")
                    source = pt.get("source_file", "unknown_source.pdf")
                    content = pt.get("text", "").strip()
                    ref_docs_str += (
                        f"Document {idx + 1}\n\n"
                        f"Title:\n{title}\n\n"
                        f"Source:\n{source}\n\n"
                        f"Content:\n{content}\n\n"
                        "--------------------------------\n\n"
                    )
            else:
                ref_docs_str = "No verified context detected.\n\n--------------------------------\n\n"

            # Perfectly aligned streaming system prompt tracking instructions matching generate_answer
            system_instruction = (
                "### Role\n"
                "You are a professional HR Support Analyst operating inside an isolated enterprise data partition.\n"
                f"Your current tenant domain scope is explicitly restricted to: [{tenant_id.upper()}]\n"
                "Your objective is to answer user queries accurately, directly, and concisely using the provided reference documents.\n\n"
                "### Strict Grounding Boundaries\n"
                "- State only the raw facts explicitly mentioned or directly answered in the reference documents.\n"
                "- Do NOT use logical transition words, summaries, or deductive connectors (e.g., do not say 'Therefore', 'However', 'Consequently', or 'As a result').\n"
                "- Completely eliminate explanatory bridges. Write only direct, independent factual declarations.\n"
                "- Output ONLY the pure extracted facts. Never include conversational preambles or comment on prompt structure.\n"
                "- Do NOT print document indices, source labels, file names, or citation markers (e.g., do not append '(Source: Document X)').\n"
                "- If the documents do not provide an answer, reply exactly with: "
                "\"The requested information is not available in the provided documents.\"\n\n"
                "### Response Formatting\n"
                "- State the direct answer clearly in your very first sentence.\n"
                "- Present multi-step rules or structured guidelines using clean Markdown bullet points."
            )

            user_payload = (
                "### REFERENCE DOCUMENTS\n\n"
                f"{ref_docs_str.strip()}\n\n"
                "### QUESTION\n\n"
                f"{user_query.strip()}\n\n"
                "### GROUNDING INSTRUCTIONS\n\n"
                "Answer the question using ONLY the reference documents above.\n"
                "Before producing the final response, internally verify that every statement is directly supported by the text.\n"
                "Remove any unsupported claims. If the information is missing from the documents, reply exactly with:\n"
                "\"The requested information is not available in the provided documents.\"\n\n"
                "### FINAL ANSWER\n"
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ]

            overrides = llm_overrides or {}
            deployment_mode = overrides.get("LLM_DEPLOYMENT_MODE", Config.LLM_DEPLOYMENT_MODE)
            api_base_url = overrides.get("LLM_API_BASE_URL", Config.LLM_API_BASE_URL)
            api_key = overrides.get("LLM_API_KEY", Config.LLM_API_KEY)
            default_model = overrides.get("DEFAULT_MODEL_ID", Config.DEFAULT_MODEL_ID)

            provider_type = overrides.get("PROVIDER_TYPE", "Cloud API" if deployment_mode == "CLOUD" else "vLLM")
            base_url = api_base_url.rstrip('/')
            if provider_type in ["Ollama", "OpenAI-Compatible"] and not base_url.endswith("/v1") and "/v1/" not in base_url:
                base_url = f"{base_url}/v1"
            vllm_endpoint = f"{base_url}/chat/completions"
            if provider_type in ["Ollama", "OpenAI-Compatible", "Cloud API"]:
                target_model = default_model
            else:
                live_models = []
                try:
                    url = f"{base_url.rstrip('/')}/models"
                    if "/v1" not in url and "/v1/" not in url:
                        url_v1 = f"{base_url.rstrip('/')}/v1/models"
                        try:
                            req = urllib.request.Request(url_v1, method="GET")
                            if api_key and api_key.lower() != "none":
                                req.add_header("Authorization", f"Bearer {api_key}")
                            with urllib.request.urlopen(req, timeout=1.5) as res:
                                data = json.loads(res.read().decode("utf-8"))
                                live_models = [m["id"] for m in data.get("data", [])]
                        except Exception:
                            pass
                    if not live_models:
                        req = urllib.request.Request(url, method="GET")
                        if api_key and api_key.lower() != "none":
                            req.add_header("Authorization", f"Bearer {api_key}")
                        with urllib.request.urlopen(req, timeout=1.5) as res:
                            data = json.loads(res.read().decode("utf-8"))
                            live_models = [m["id"] for m in data.get("data", [])]
                except Exception:
                    pass
                
                if target_adapter in live_models:
                    target_model = target_adapter
                else:
                    target_model = default_model

            vllm_payload = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1024
            }
            
            if provider_type not in ["Cloud API"]:
                vllm_payload["repetition_penalty"] = 1.05

            request_headers = {"Content-Type": "application/json"}
            if api_key and api_key.lower() != "none":
                request_headers["Authorization"] = f"Bearer {api_key}"

            ttft_ms = 0.0
            generation_ms = 0.0
            response_text = ""
            token_metrics = {}

            try:
                import httpx
                vllm_payload["stream"] = True
                llm_start = time.perf_counter_ns()
                first_chunk_received = False
                
                with httpx.stream("POST", vllm_endpoint, json=vllm_payload, headers=request_headers, timeout=15.0) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not first_chunk_received:
                            ttft_ms = round((time.perf_counter_ns() - llm_start) / 1_000_000.0, 2)
                            first_chunk_received = True
                        
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    chunk_text = delta["content"]
                                    response_text += chunk_text
                                    yield {"type": "token", "content": chunk_text}
                            except Exception:
                                pass
                generation_ms = round((time.perf_counter_ns() - llm_start) / 1_000_000.0, 2)
                
                prompt_tokens = len(system_instruction.split()) + len(user_payload.split())
                completion_tokens = len(response_text.split())
                token_metrics = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            except Exception as stream_err:
                import requests
                vllm_payload["stream"] = False
                llm_start = time.perf_counter_ns()
                res_sync = requests.post(vllm_endpoint, json=vllm_payload, headers=request_headers, timeout=15.0)
                res_sync.raise_for_status()
                res_data = res_sync.json()
                response_text = res_data["choices"][0]["message"]["content"]
                generation_ms = round((time.perf_counter_ns() - llm_start) / 1_000_000.0, 2)
                ttft_ms = round(generation_ms * 0.15, 2)
                token_metrics = res_data.get("usage", {})
                yield {"type": "token", "content": response_text}

            telemetry = {
                "pre_processing_ms": pre_processing_ms,
                "embedding_ms": retrieval_telemetry["embedding_ms"],
                "qdrant_ms": retrieval_telemetry["qdrant_ms"],
                "rerank_ms": retrieval_telemetry["rerank_ms"],
                "ttft_ms": ttft_ms,
                "generation_ms": generation_ms
            }

            yield {
                "type": "final",
                "status": "success",
                "answer": response_text.strip(),
                "citations": citations,
                "latency_seconds": round((time.perf_counter_ns() - pre_start) / 1_000_000_000.0, 3),
                "token_metrics": token_metrics,
                "telemetry": telemetry,
                "raw_context_block": flat_context,
                "context_nodes": context_nodes
            }
        except Exception as e:
            yield {"type": "error", "status": "error", "message": f"Modular Inference routing path failure: {str(e)}"}

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

        provider_type = overrides.get("PROVIDER_TYPE", "Cloud API" if deployment_mode == "CLOUD" else "vLLM")
        base_url = api_base_url.rstrip('/')
        if provider_type in ["Ollama", "OpenAI-Compatible"] and not base_url.endswith("/v1") and "/v1/" not in base_url:
            base_url = f"{base_url}/v1"
        vllm_endpoint = f"{base_url}/chat/completions"
        if provider_type in ["Ollama", "OpenAI-Compatible", "Cloud API"]:
            target_model = default_model
        else:
            target_model = "tech_support"

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

        provider_type = overrides.get("PROVIDER_TYPE", "Cloud API" if deployment_mode == "CLOUD" else "vLLM")
        base_url = api_base_url.rstrip('/')
        if provider_type in ["Ollama", "OpenAI-Compatible"] and not base_url.endswith("/v1") and "/v1/" not in base_url:
            base_url = f"{base_url}/v1"
        vllm_endpoint = f"{base_url}/chat/completions"
        if provider_type in ["Ollama", "OpenAI-Compatible", "Cloud API"]:
            target_model = default_model
        else:
            live_models = []
            try:
                url = f"{base_url.rstrip('/')}/models"
                if "/v1" not in url and "/v1/" not in url:
                    url_v1 = f"{base_url.rstrip('/')}/v1/models"
                    try:
                        req = urllib.request.Request(url_v1, method="GET")
                        if api_key and api_key.lower() != "none":
                            req.add_header("Authorization", f"Bearer {api_key}")
                        with urllib.request.urlopen(req, timeout=1.5) as res:
                            data = json.loads(res.read().decode("utf-8"))
                            live_models = [m["id"] for m in data.get("data", [])]
                    except Exception:
                        pass
                if not live_models:
                    req = urllib.request.Request(url, method="GET")
                    if api_key and api_key.lower() != "none":
                        req.add_header("Authorization", f"Bearer {api_key}")
                    with urllib.request.urlopen(req, timeout=1.5) as res:
                        data = json.loads(res.read().decode("utf-8"))
                        live_models = [m["id"] for m in data.get("data", [])]
            except Exception:
                pass
            
            if target_adapter in live_models:
                target_model = target_adapter
            else:
                target_model = default_model

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