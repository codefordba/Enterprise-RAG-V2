import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import urllib.request
import json
import ssl
from typing import Dict, Any, List
from src.config import Config
from src.generation.orchestrator import ContextOrchestrator
from src.processing.ingest_pipeline import TenantIngestionPipeline
from src.database.secure_storage import SecureStorageManager

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="Enterprise-RAG Ops Console", page_icon="🏛️", layout="wide")

# Load persistent tenant map and fallback defaults
if "tenant_adapter_map" not in st.session_state:
    st.session_state.tenant_adapter_map = SecureStorageManager.load_tenant_registry()

# Load encrypted model config profiles and fallback defaults
if "llm_overrides" not in st.session_state:
    profiles = SecureStorageManager.load_encrypted_profiles()
    active_profile_name = list(profiles.keys())[0] if profiles else "Default Environment"
    active_profile = profiles.get(active_profile_name, {
        "LLM_DEPLOYMENT_MODE": Config.LLM_DEPLOYMENT_MODE,
        "LLM_API_BASE_URL": Config.LLM_API_BASE_URL,
        "LLM_API_KEY": Config.LLM_API_KEY,
        "DEFAULT_MODEL_ID": Config.DEFAULT_MODEL_ID
    })
    st.session_state.llm_overrides = {
        "LLM_DEPLOYMENT_MODE": active_profile.get("LLM_DEPLOYMENT_MODE", "CLOUD"),
        "LLM_API_BASE_URL": active_profile.get("LLM_API_BASE_URL", ""),
        "LLM_API_KEY": active_profile.get("LLM_API_KEY", ""),
        "DEFAULT_MODEL_ID": active_profile.get("DEFAULT_MODEL_ID", "")
    }

if "sandygpt_messages" not in st.session_state:
    st.session_state.sandygpt_messages = {}

class CoreOperationsCenterApp:
    def __init__(self):
        self.orchestrator = ContextOrchestrator()

    def _inject_custom_css(self):
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
            <style>
                /* Typography & Reset */
                html, body, [class*="css"], .stApp {
                    font-family: 'Outfit', sans-serif;
                }
                
                /* Title Header with Gradient Styling */
                .app-title-container {
                    text-align: center;
                    margin-bottom: 2rem;
                    padding-top: 1rem;
                }
                .app-header {
                    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2.8rem;
                    font-weight: 800;
                    margin-bottom: 0.1rem;
                    letter-spacing: -0.05rem;
                }
                .app-subheader {
                    font-size: 1.1rem;
                    color: #9ca3af;
                    margin-top: 0px;
                }

                /* Metric Telemetry Card Grid System */
                .telemetry-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1.2rem;
                    margin-bottom: 2.2rem;
                }
                .telemetry-card {
                    background: rgba(30, 41, 59, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 12px;
                    padding: 1.2rem;
                    backdrop-filter: blur(8px);
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                    transition: transform 0.2s, border-color 0.2s;
                }
                .telemetry-card:hover {
                    transform: translateY(-2px);
                    border-color: rgba(99, 102, 241, 0.4);
                }
                .telemetry-card .card-title {
                    font-size: 0.85rem;
                    color: #9ca3af;
                    text-transform: uppercase;
                    letter-spacing: 0.05rem;
                    margin-bottom: 0.4rem;
                }
                .telemetry-card .card-value {
                    font-size: 1.6rem;
                    font-weight: 600;
                    color: #f3f4f6;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                /* Glowing Telemetry Lights */
                .status-pulse {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    display: inline-block;
                }
                .pulse-green {
                    background: #10b981;
                    box-shadow: 0 0 10px #10b981;
                    animation: pulseGreen 1.8s infinite alternate;
                }
                .pulse-red {
                    background: #ef4444;
                    box-shadow: 0 0 10px #ef4444;
                    animation: pulseRed 1.8s infinite alternate;
                }
                
                @keyframes pulseGreen {
                    0% { transform: scale(0.9); box-shadow: 0 0 5px rgba(16, 185, 129, 0.5); }
                    100% { transform: scale(1.15); box-shadow: 0 0 15px rgba(16, 185, 129, 1); }
                }
                @keyframes pulseRed {
                    0% { transform: scale(0.9); box-shadow: 0 0 5px rgba(239, 68, 68, 0.5); }
                    100% { transform: scale(1.15); box-shadow: 0 0 15px rgba(239, 68, 68, 1); }
                }

                /* Sidebar Section styling */
                .sidebar-section {
                    background: rgba(30, 41, 59, 0.5);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 10px;
                    padding: 0.8rem 1rem;
                    margin-bottom: 0.8rem;
                }
                .sidebar-section-title {
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: #a5b4fc;
                    text-transform: uppercase;
                    letter-spacing: 0.04rem;
                }

                /* Custom Tabs Restyling */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 1.5rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                    margin-bottom: 1.5rem;
                }
                .stTabs [data-baseweb="tab"] {
                    font-size: 1.05rem;
                    font-weight: 600;
                    color: #9ca3af;
                    padding-bottom: 0.8rem;
                    transition: color 0.2s;
                }
                .stTabs [data-baseweb="tab"]:hover {
                    color: #e0e7ff;
                }
                .stTabs [data-baseweb="tab"][aria-selected="true"] {
                    color: #6366f1;
                    border-bottom: 2px solid #6366f1;
                }

                /* Premium Card Border overrides */
                div[data-testid="stForm"] {
                    border: 1px solid rgba(255, 255, 255, 0.08) !important;
                    background: rgba(30, 41, 59, 0.25) !important;
                    border-radius: 12px !important;
                    padding: 1.8rem !important;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
                }
                
                .citation-card {
                    background: rgba(30, 41, 59, 0.45);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 10px;
                    padding: 0.9rem 1.2rem;
                    margin-bottom: 0.8rem;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                    transition: border-color 0.2s;
                }
                .citation-card:hover {
                    border-color: rgba(99, 102, 241, 0.3);
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    def _check_node_health(self, url: str, method: str = "GET", requires_auth: bool = False) -> bool:
        try:
            headers = {}
            api_key = st.session_state.llm_overrides["LLM_API_KEY"]
            if requires_auth and api_key and api_key.lower() != "none":
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def _fetch_collection_stats(self) -> dict:
        url = f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}/collections/{Config.COLLECTION_NAME}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                data = json.loads(res.read().decode("utf-8"))
                result = data.get("result", {})
                return {
                    "status": result.get("status", "unknown").upper(),
                    "points_count": result.get("points_count", 0),
                    "vectors_count": result.get("vectors_count", 0),
                    "segments_count": result.get("segments_count", 0)
                }
        except Exception:
            return {
                "status": "OFFLINE",
                "points_count": 0,
                "vectors_count": 0,
                "segments_count": 0
            }

    def _fetch_live_vllm_adapters(self) -> list:
        if st.session_state.llm_overrides["LLM_DEPLOYMENT_MODE"] == "CLOUD":
            return [st.session_state.llm_overrides["DEFAULT_MODEL_ID"]]
        url = f"{st.session_state.llm_overrides['LLM_API_BASE_URL']}/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                data = json.loads(res.read().decode("utf-8"))
                return [model["id"] for model in data.get("data", [])]
        except Exception:
            return list(set(st.session_state.tenant_adapter_map.values()))

    def _clean_markdown_fences(self, text: str) -> str:
        clean_text = text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
        return clean_text

    def render_sidebar_status(self):
        with st.sidebar:
            st.markdown('<div class="app-header" style="font-size: 1.8rem; text-align: left; margin-bottom: 0.2rem;">🏛️ Ops Center</div>', unsafe_allow_html=True)
            st.markdown('<p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 1.5rem; margin-top: 0px;">Enterprise-RAG Partition Panel</p>', unsafe_allow_html=True)
            
            st.markdown(
                """
                <div class="sidebar-section">
                    <div class="sidebar-section-title">🔑 Active Tenant Scope</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.session_state.current_tenant = st.selectbox(
                "Select active workspace focus:",
                options=list(st.session_state.tenant_adapter_map.keys()),
                format_func=lambda x: f"🏢 {x.upper()}",
                label_visibility="collapsed"
            )
            
            active_adapter = st.session_state.tenant_adapter_map[st.session_state.current_tenant]
            if st.session_state.llm_overrides["LLM_DEPLOYMENT_MODE"] == "CLOUD":
                st.caption(f"Routed Model: `{st.session_state.llm_overrides['DEFAULT_MODEL_ID']}`")
            else:
                st.caption(f"Routed Weight Matrix: `{active_adapter}`")
            
            st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
            if st.button("🔄 Trigger Hard Refresh", use_container_width=True):
                st.rerun()

    def render_telemetry_dashboard(self):
        q_health = self._check_node_health(f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}/readyz")
        t_health = self._check_node_health(Config.TEI_ENDPOINT.split("/embed")[0], method="HEAD")
        
        api_base = st.session_state.llm_overrides["LLM_API_BASE_URL"]
        if st.session_state.llm_overrides["LLM_DEPLOYMENT_MODE"] == "CLOUD":
            v_health = self._check_node_health(f"{api_base}/models", method="GET", requires_auth=True)
            llm_label = "Cloud API"
        else:
            v_health = self._check_node_health(f"{api_base}/models")
            llm_label = "On-Prem Cluster"
            
        q_stats = self._fetch_collection_stats()
        
        q_dot = "pulse-green" if q_health else "pulse-red"
        t_dot = "pulse-green" if t_health else "pulse-red"
        v_dot = "pulse-green" if v_health else "pulse-red"
        
        html_content = f"""
        <div class="telemetry-grid">
            <div class="telemetry-card">
                <div class="card-title">📡 Vector Node Status</div>
                <div class="card-value">
                    <span class="status-pulse {q_dot}"></span>
                    {q_stats.get("status", "OFFLINE")}
                </div>
            </div>
            <div class="telemetry-card">
                <div class="card-title">💾 Total Vector Footprint</div>
                <div class="card-value">{q_stats.get("points_count", 0):,} <span style="font-size: 0.8rem; color: #9ca3af; font-weight: normal; margin-left: 0.2rem;">vectors</span></div>
            </div>
            <div class="telemetry-card">
                <div class="card-title">⚡ Embedding Core</div>
                <div class="card-value">
                    <span class="status-pulse {t_dot}"></span>
                    {":8080" if t_health else "OFFLINE"}
                </div>
            </div>
            <div class="telemetry-card">
                <div class="card-title">🧠 Inference Hub</div>
                <div class="card-value">
                    <span class="status-pulse {v_dot}"></span>
                    {llm_label if v_health else "OFFLINE"}
                </div>
            </div>
            <div class="telemetry-card">
                <div class="card-title">🏢 Active Tenants</div>
                <div class="card-value">{len(st.session_state.tenant_adapter_map)} <span style="font-size: 0.8rem; color: #9ca3af; font-weight: normal; margin-left: 0.2rem;">domains</span></div>
            </div>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)

    def render_tenant_admin_tab(self):
        st.header("🧱 Dynamic Tenant Provisioning & LoRA Matrix Setup")
        add_col, del_col = st.columns(2)
        with add_col:
            st.subheader("✨ Provision New Corporate Tenant")
            with st.form("create_tenant_form", clear_on_submit=True):
                new_id = st.text_input("New Tenant Registry Key Unique Name:", placeholder="e.g., legal_compliance").strip()
                live_hardware_adapters = self._fetch_live_vllm_adapters()
                new_lora = st.selectbox("Select Target Inference Weight Matrix Identifier:", options=live_hardware_adapters)
                
                if st.form_submit_button("🔨 Initialize Tenant Domain Key"):
                    if new_id and new_lora:
                        st.session_state.tenant_adapter_map[new_id] = new_lora
                        SecureStorageManager.save_tenant_registry(st.session_state.tenant_adapter_map)
                        st.success(f"🎉 Tenant registry configured successfully! Domain Key [{new_id.upper()}] active.")
                        st.rerun()
                        
        with del_col:
            st.subheader("🗑️ Deprovision Isolated Tenant Matrix")
            target_del = st.selectbox("Select Target Tenant to Purge:", options=list(st.session_state.tenant_adapter_map.keys()))
            if st.button("❌ Terminate Tenant Workspace", type="primary", use_container_width=True):
                purge_payload = {"filter": {"must": [{"key": "tenant_id", "match": {"value": target_del}}]}}
                try:
                    urllib.request.urlopen(urllib.request.Request(
                        f"http://{Config.QDRANT_HOST}:{Config.QDRANT_PORT}/collections/{Config.COLLECTION_NAME}/points/delete",
                        data=json.dumps(purge_payload).encode("utf-8"), headers={"Content-Type":"application/json"}, method="POST"
                    ))
                except Exception:
                    pass
                del st.session_state.tenant_adapter_map[target_del]
                SecureStorageManager.save_tenant_registry(st.session_state.tenant_adapter_map)
                st.warning(f"💥 Workspace and all underlying vector footprints for [{target_del.upper()}] destroyed.")
                st.rerun()

    def render_data_feed_tab(self):
        st.header("📥 Document & Spreadsheet Processing Panel")
        st.write(f"Target Cluster Workspace Boundary: **[{st.session_state.current_tenant.upper()}]**")
        uploaded_file = st.file_uploader("Upload Target Document / Spreadsheet Assets:", type=["pdf", "xlsx", "xls"])
        
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            st.info(f"Asset Loaded: `{uploaded_file.name}` ({uploaded_file.size / 1024:.2f} KB)")
            suggested_family = uploaded_file.name.split(".")[0].split("_v")[0].split("202")[0].strip("_").lower()
            ui_family_key = st.text_input("📋 Document Lineage Identifier / Family Tracking Key:", value=suggested_family)
            
            if st.button("⚡ Start Layout-Aware Vector Ingestion", type="primary"):
                raw_pages = []
                if file_ext == "pdf":
                    with st.spinner("Extracting multi-page text and tabular layers..."):
                        import pdfplumber
                        with pdfplumber.open(uploaded_file) as pdf:
                            for idx, page in enumerate(pdf.pages, 1):
                                text_body = page.extract_text() or ""
                                table_md_accumulate = []
                                tables = page.extract_tables() or []
                                for tbl in tables:
                                    if not tbl: continue
                                    cleaned_rows = [[str(cell).strip() if cell is not None else "" for cell in r] for r in tbl]
                                    if len(cleaned_rows) > 1:
                                        headers = cleaned_rows[0]
                                        m_str = f"| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
                                        for row in cleaned_rows[1:]:
                                            m_str += f"| {' | '.join(row)} |\n"
                                        table_md_accumulate.append(m_str)
                                raw_pages.append({"page_number": idx, "text": text_body, "table_markdown": "\n\n".join(table_md_accumulate)})

                elif file_ext in ["xlsx", "xls"]:
                    with st.spinner("Parsing workbook layers and layout grids..."):
                        import pandas as pd
                        excel_workbook = pd.ExcelFile(uploaded_file)
                        for sheet_idx, sheet_name in enumerate(excel_workbook.sheet_names, 1):
                            df = excel_workbook.parse(sheet_name).fillna("")
                            if df.empty: continue
                            headers = [str(col).strip() for col in df.columns]
                            md_grid = f"### SPREADSHEET WORKBOOK TAB: {sheet_name.upper()}\n| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
                            for _, row in df.iterrows():
                                md_grid += f"| {' | '.join([str(val).strip() for val in row.values])} |\n"
                            raw_pages.append({"page_number": sheet_idx, "text": f"Workbook worksheet tab: {sheet_name}.", "table_markdown": md_grid})

                if not raw_pages:
                    st.warning("No usable records discovered inside the uploaded asset.")
                    return

                with st.spinner("⚙️ Evaluating lineage indexes and executing atomic updates..."):
                    pipeline = TenantIngestionPipeline(tenant_id=st.session_state.current_tenant)
                    status = pipeline.process_and_upsert(document_name=uploaded_file.name, raw_pages=raw_pages, custom_family_key=ui_family_key)
                    if status == "skipped_duplicate":
                        st.warning("ℹ️ System Event: Content match detected. Ingestion bypassed.")
                    elif status == "ingested_successfully":
                        st.success(f"🎉 Success! Asset '{uploaded_file.name}' mapped into lineage tracking path [{ui_family_key.upper()}].")
                        st.rerun()

    def render_summarization_tab(self):
        st.header("📝 Executive Document Summarization Console")
        st.write("Generate a structured executive summary directly from your active model endpoint.")
        summary_length = st.select_slider("Target Summary Depth Level:", options=["Brief & Concise", "Standard Medium", "Deep & Highly Detailed"], value="Standard Medium")
        
        # Advanced window and generation length settings
        with st.expander("⚙️ Advanced Context Window & Generation Length Settings", expanded=False):
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                ui_max_context_chars = st.number_input(
                    "Max Input Context Limit (Characters):",
                    min_value=1000,
                    max_value=10000000,
                    value=300000,
                    step=10000,
                    help="Maximum characters of the document sent to the LLM to fit within the model's context window."
                )
            with param_col2:
                ui_max_output_tokens = st.slider(
                    "Max Summary Generation Length (Output Tokens):",
                    min_value=256,
                    max_value=8192,
                    value=3000,
                    step=128,
                    help="Maximum response length generated by the LLM. Increase this if the summary gets cut off."
                )

        uploaded_sum_file = st.file_uploader("Choose a Document Asset for Summary Analysis:", type=["pdf", "docx", "xlsx", "xls"])
        
        if uploaded_sum_file is not None:
            file_ext = uploaded_sum_file.name.split(".")[-1].lower()
            if st.button("⚡ Generate Structured Executive Summary", type="primary", use_container_width=True):
                full_text_stream = ""
                with st.spinner("Extracting complete document text streams..."):
                    if file_ext == "pdf":
                        import pdfplumber
                        with pdfplumber.open(uploaded_sum_file) as pdf:
                            for page in pdf.pages:
                                full_text_stream += (page.extract_text() or "") + "\n"
                    elif file_ext == "docx":
                        import docx
                        doc = docx.Document(uploaded_sum_file)
                        full_text_stream = "\n".join([para.text for para in doc.paragraphs])
                    elif file_ext in ["xlsx", "xls"]:
                        import pandas as pd
                        excel_workbook = pd.ExcelFile(uploaded_sum_file)
                        excel_texts = []
                        for sheet_name in excel_workbook.sheet_names:
                            df = excel_workbook.parse(sheet_name).fillna("")
                            if df.empty:
                                continue
                            headers = [str(col).strip() for col in df.columns]
                            md_grid = f"### SPREADSHEET WORKBOOK TAB: {sheet_name.upper()}\n| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
                            for _, row in df.iterrows():
                                md_grid += f"| {' | '.join([str(val).strip() for val in row.values])} |\n"
                            excel_texts.append(md_grid)
                        full_text_stream = "\n\n".join(excel_texts)

                if not full_text_stream.strip():
                    st.warning("The uploaded document appears to contain no extractable alphanumeric text layers.")
                    return

                with st.spinner("🧠 Processing text structures and running analytical generation loops..."):
                    res = self.orchestrator.generate_summary(
                        document_name=uploaded_sum_file.name,
                        full_text_content=full_text_stream,
                        summary_length=summary_length,
                        max_tokens=ui_max_output_tokens,
                        max_context_chars=ui_max_context_chars,
                        llm_overrides=st.session_state.llm_overrides
                    )
                    if res.get("status") == "error":
                        st.error(res.get("message"))
                        return
                    st.success("🎉 Document analysis complete!")
                    st.markdown("---")
                    st.markdown("### 📋 Generated Executive Summary")
                    
                    clean_summary = self._clean_markdown_fences(res.get("summary", ""))
                    with st.container(border=True):
                        st.markdown(clean_summary)
                    
                    st.caption(f"Latency: {res.get('latency_seconds')}s | Context Tokens: {res.get('token_metrics', {}).get('prompt_tokens', 0)}")

    def render_query_playground_tab(self):
        st.header("🔍 Isolated Inference Console & Prompt Sandbox")
        st.write(f"Context Protection Isolation Mode: **[{st.session_state.current_tenant.upper()}]**")
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            ui_temp = st.slider("Model Generation Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        with param_col2:
            ui_top_k = st.slider("Top-K Retrieved Database Context Blocks:", min_value=1, max_value=10, value=3, step=1)

        user_query = st.text_input("Execute Data Search Input Query String:", placeholder="Ask anything about data in this workspace...")
        if st.button("🚀 Run Grounded Query Pass", type="primary", use_container_width=True):
            if not user_query: return
            with st.spinner("Compiling contextual layers..."):
                active_lora = st.session_state.tenant_adapter_map[st.session_state.current_tenant]
                res = self.orchestrator.generate_answer(
                    tenant_id=st.session_state.current_tenant,
                    target_adapter=active_lora,
                    user_query=user_query,
                    temperature=ui_temp,
                    top_k=ui_top_k,
                    llm_overrides=st.session_state.llm_overrides
                )
                if res.get("status") == "error":
                    st.error(res.get("message"))
                    return
                st.markdown("### 🤖 Grounded Generation Response Output")
                
                clean_answer = self._clean_markdown_fences(res.get("answer", ""))
                with st.container(border=True):
                    st.markdown(clean_answer)
                
                metric_col, citation_col = st.columns([1, 2])
                with metric_col:
                    st.markdown("##### ⏱️ Lifecycle Metrics")
                    st.metric("Total Execution Time", f"{res.get('latency_seconds')} sec")
                    st.markdown(f"**Context Tokens:** `{res.get('token_metrics', {}).get('prompt_tokens', 0)}`")
                with citation_col:
                    st.markdown("##### 📌 Citations Audit Trail")
                    for idx, cit in enumerate(res.get("citations", []), 1):
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div style="font-weight: 600; color: #a5b4fc; font-size: 0.9rem;">[{idx}] {cit['source']} (Page: {cit['page']})</div>
                                <div style="color: #9ca3af; font-size: 0.8rem; margin-top: 0.2rem;">Similarity Ranking: {cit['score']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    def render_sandygpt_tab(self):
        st.header("💬 SandyGPT Conversational Workspace")
        st.write("SandyGPT is a direct GPT chat conversation interface (non-RAG mode).")
        
        tenant_id = st.session_state.current_tenant
        if tenant_id not in st.session_state.sandygpt_messages:
            st.session_state.sandygpt_messages[tenant_id] = []
            
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.sandygpt_messages[tenant_id]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                                
        if user_prompt := st.chat_input("Message SandyGPT..."):
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            st.session_state.sandygpt_messages[tenant_id].append({"role": "user", "content": user_prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("SandyGPT is typing..."):
                    active_lora = st.session_state.tenant_adapter_map[tenant_id]
                    formatted_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.sandygpt_messages[tenant_id]
                    ]
                    
                    res = self.orchestrator.generate_chat_response(
                        target_adapter=active_lora,
                        chat_history=formatted_history,
                        temperature=0.7,
                        llm_overrides=st.session_state.llm_overrides
                    )
                    
                    if res.get("status") == "error":
                        st.error(res.get("message"))
                        st.session_state.sandygpt_messages[tenant_id].pop()
                        return
                        
                    clean_ans = self._clean_markdown_fences(res.get("answer", ""))
                    st.markdown(clean_ans)
                                
                    st.session_state.sandygpt_messages[tenant_id].append({
                        "role": "assistant",
                        "content": clean_ans
                    })
                    
                    st.rerun()

    def run(self):
        self._inject_custom_css()
        
        # Dashboard title banner
        st.markdown(
            """
            <div class="app-title-container">
                <h1 class="app-header">🏛️ ENTERPRISE-RAG</h1>
                <p class="app-subheader">Multi-Tenant Knowledge Ingestion & Grounded Inference Platform</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Telemetry dashboard block
        self.render_telemetry_dashboard()
        
        # Runtime Configurations Overrides Expander
        with st.expander("⚙️ Connection Settings & LLM Engine Config overrides", expanded=False):
            st.markdown("Configure model profiles and connection details. Encrypted details are saved securely on disk.")
            
            # Load registered encrypted profiles
            profiles = SecureStorageManager.load_encrypted_profiles()
            profile_keys = list(profiles.keys())
            
            st.subheader("📁 Model Profiles Directory")
            
            select_col, action_col = st.columns([3, 1])
            with select_col:
                selected_profile = st.selectbox(
                    "Switch Active Connection Profile:",
                    options=profile_keys,
                    key="profile_select_box"
                )
            with action_col:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("⚡ Load Profile", use_container_width=True):
                    profile_data = profiles[selected_profile]
                    st.session_state.llm_overrides = {
                        "LLM_DEPLOYMENT_MODE": profile_data.get("LLM_DEPLOYMENT_MODE", "CLOUD"),
                        "LLM_API_BASE_URL": profile_data.get("LLM_API_BASE_URL", ""),
                        "LLM_API_KEY": profile_data.get("LLM_API_KEY", ""),
                        "DEFAULT_MODEL_ID": profile_data.get("DEFAULT_MODEL_ID", "")
                    }
                    st.success(f"🎉 Active config swapped to: '{selected_profile}'!")
                    st.rerun()
            
            st.markdown("---")
            st.subheader("📝 Edit Active Settings or Onboard New Profile")
            
            with st.form("llm_config_form"):
                mode_col, model_col = st.columns(2)
                with mode_col:
                    mode_selection = st.radio(
                        "LLM Engine Mode:",
                        options=["Local Model (vLLM)", "Cloud API"],
                        index=0 if st.session_state.llm_overrides["LLM_DEPLOYMENT_MODE"] == "ON_PREM" else 1,
                        horizontal=True
                    )
                with model_col:
                    if mode_selection == "Cloud API":
                        cloud_models = [
                            "gemini-1.5-flash",
                            "gemini-1.5-pro",
                            "gemini-2.0-flash",
                            "gemini-2.0-pro-exp",
                            "Custom Model..."
                        ]
                        current_model = st.session_state.llm_overrides["DEFAULT_MODEL_ID"]
                        if current_model in cloud_models:
                            default_idx = cloud_models.index(current_model)
                            is_custom = False
                        else:
                            default_idx = cloud_models.index("Custom Model...")
                            is_custom = True
                            
                        selected_model = st.selectbox(
                            "Select Cloud Model ID:",
                            options=cloud_models,
                            index=default_idx
                        )
                        if selected_model == "Custom Model...":
                            ui_model_id = st.text_input(
                                "Enter Custom Model ID:",
                                value=current_model if is_custom else ""
                            )
                        else:
                            ui_model_id = selected_model
                    else:
                        local_models = self._fetch_live_vllm_adapters()
                        if "Custom Model..." not in local_models:
                            local_models.append("Custom Model...")
                        current_model = st.session_state.llm_overrides["DEFAULT_MODEL_ID"]
                        if current_model in local_models:
                            default_idx = local_models.index(current_model)
                            is_custom = False
                        else:
                            default_idx = local_models.index("Custom Model...")
                            is_custom = True
                            
                        selected_model = st.selectbox(
                            "Select Local Model / LoRA Adapter:",
                            options=local_models,
                            index=default_idx
                        )
                        if selected_model == "Custom Model...":
                            ui_model_id = st.text_input(
                                "Enter Custom Model ID / Adapter Name:",
                                value=current_model if is_custom else ""
                            )
                        else:
                            ui_model_id = selected_model
                    
                url_col, key_col = st.columns(2)
                with url_col:
                    ui_base_url = st.text_input(
                        "API Base URL Path Endpoint:",
                        value=st.session_state.llm_overrides["LLM_API_BASE_URL"]
                    )
                with key_col:
                    ui_api_key = st.text_input(
                        "API Auth Token Key:",
                        value=st.session_state.llm_overrides["LLM_API_KEY"],
                        type="password"
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                profile_save_col, name_col = st.columns([1, 1])
                with name_col:
                    save_name = st.text_input(
                        "Onboard Profile Name (Leave blank to only apply to active session):",
                        value=selected_profile if selected_profile in profiles else "",
                        placeholder="e.g. My Custom Gemini Profile"
                    ).strip()
                
                with profile_save_col:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    delete_profile = st.checkbox("Delete selected profile on Save")
                
                if st.form_submit_button("💾 Save & Apply Connection Config"):
                    if delete_profile and save_name in profiles:
                        # Prevent deleting the last remaining profile to preserve sanity
                        if len(profiles) <= 1:
                            st.error("❌ Action Rejected: Cannot delete the last remaining configuration profile!")
                        else:
                            del profiles[save_name]
                            SecureStorageManager.save_encrypted_profiles(profiles)
                            st.warning(f"💥 Profile '{save_name}' removed from directory.")
                            st.rerun()
                    else:
                        st.session_state.llm_overrides["LLM_DEPLOYMENT_MODE"] = "ON_PREM" if mode_selection == "Local Model (vLLM)" else "CLOUD"
                        st.session_state.llm_overrides["DEFAULT_MODEL_ID"] = ui_model_id.strip()
                        st.session_state.llm_overrides["LLM_API_BASE_URL"] = ui_base_url.strip()
                        st.session_state.llm_overrides["LLM_API_KEY"] = ui_api_key.strip()
                        
                        if save_name:
                            profiles[save_name] = {
                                "LLM_DEPLOYMENT_MODE": st.session_state.llm_overrides["LLM_DEPLOYMENT_MODE"],
                                "LLM_API_BASE_URL": st.session_state.llm_overrides["LLM_API_BASE_URL"],
                                "LLM_API_KEY": st.session_state.llm_overrides["LLM_API_KEY"],
                                "DEFAULT_MODEL_ID": st.session_state.llm_overrides["DEFAULT_MODEL_ID"]
                            }
                            SecureStorageManager.save_encrypted_profiles(profiles)
                            st.success(f"🎉 Configuration settings encrypted and saved to profile '{save_name}'!")
                        else:
                            st.success("🎉 Configuration settings saved and applied to active session!")
                    
                    st.rerun()
        
        self.render_sidebar_status()
        t_chat, t_query, t_summary, t_feed, t_admin = st.tabs([
            "💬 SandyGPT Conversational Workspace",
            "🔍 Grounded Query Workspace Playground", 
            "📝 Document Summarization Console",
            "📥 Document Processing Feed Panel", 
            "📊 Tenant Space Administration"
        ])
        with t_chat: self.run_error_wrapper(self.render_sandygpt_tab)
        with t_query: self.run_error_wrapper(self.render_query_playground_tab)
        with t_summary: self.run_error_wrapper(self.render_summarization_tab)
        with t_feed: self.run_error_wrapper(self.render_data_feed_tab)
        with t_admin: self.run_error_wrapper(self.render_tenant_admin_tab)

    def run_error_wrapper(self, render_func):
        try:
            render_func()
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")

if __name__ == "__main__":
    app = CoreOperationsCenterApp()
    app.run()