import streamlit as st
import os
import json
import time
import threading
from orchestration.flow import RaggedyOrchestrator
from utils.logger import raggedy_logger
from utils.chat_manager import ChatManager
from utils.resource_monitor import ResourceMonitor

st.set_page_config(page_title="RAGGEDY TOOL", page_icon="🤖", layout="wide")

# Initialize orchestrator
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = RaggedyOrchestrator()

# Initialize Chat Manager
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

# Initialize Resource Monitor
if "monitor" not in st.session_state:
    st.session_state.monitor = ResourceMonitor()

# Initialize active chat
if st.session_state.get("active_chat_id") is None:
    chats = st.session_state.chat_manager.list_chats()
    if chats:
        st.session_state.active_chat_id = chats[0]["id"]
    else:
        st.session_state.active_chat_id = st.session_state.chat_manager.create_chat()

# Background Worker for async tasks
class BackgroundWorker:
    def __init__(self, orchestrator, chat_manager, monitor):
        self.orchestrator = orchestrator
        self.chat_manager = chat_manager
        self.monitor = monitor
        self.statuses = {} # chat_id -> str

    def start_task(self, chat_id, prompt, history, query_emb, chunks_map, methods, doc_ids):
        # Initial check
        if not self.orchestrator.llm.is_available():
            self.statuses[chat_id] = "🔴 LLM Offline"
            return

        self.statuses[chat_id] = "⏳ Waiting for Idle"
        thread = threading.Thread(
            target=self._task_wrapper,
            args=(chat_id, prompt, history, query_emb, chunks_map, methods, doc_ids)
        )
        thread.daemon = True
        thread.start()

    def _task_wrapper(self, chat_id, prompt, history, query_emb, chunks_map, methods, doc_ids):
        try:
            # Initial wait to let the main chat stream get a head start
            time.sleep(3)
            
            # Check resources before each heavy step
            stats = self.monitor.get_system_stats()
            # If we are using a lot of resources, wait a bit or bail
            # LLM generation is heavy, so we wait if percent is high
            wait_attempts = 0
            while stats.get("vram_available") and stats["vram_percent"] > 85 and wait_attempts < 10:
                self.statuses[chat_id] = "⏳ Busy - Waiting"
                time.sleep(5)
                stats = self.monitor.get_system_stats()
                wait_attempts += 1
            
            if stats.get("vram_available") and stats["vram_percent"] > 95:
                self.statuses[chat_id] = "❌ Low Resources - Skipped"
                return

            self.statuses[chat_id] = "🏃 Analyzing"
            # 1. Insights
            new_insights = self.orchestrator.llm.extract_insights(prompt, history)
            
            # Check again
            stats = self.monitor.get_system_stats()
            if stats.get("vram_available") and stats["vram_percent"] > 95:
                # Insights is usually enough if resources are tight
                self.statuses[chat_id] = "✅ Complete (Minimal)"
                chat_data = self.chat_manager.load_chat(chat_id)
                if chat_data:
                    self.chat_manager.save_chat(chat_id, chat_data.get("messages", []), insights=new_insights)
                return

            # 2. Data Pooling (Only if documents are involved)
            pooled_data = {}
            if doc_ids is None or len(doc_ids) > 0:
                self.statuses[chat_id] = "🏃 Pooling Data"
                pooled_data = self.orchestrator.pool_additional_data(
                    prompt, history, query_emb, chunks_map, methods, doc_ids
                )
            
            # 3. Proactive Thought
            self.statuses[chat_id] = "🏃 Reflecting"
            proactive_thought = self.orchestrator.evaluate_proactive_thought(
                prompt, history, list(chunks_map.values()), pooled_data.get("chunks", [])
            )
            
            # 5. Grade the Answer
            self.statuses[chat_id] = "🏃 Grading"
            chat_data_curr = self.chat_manager.load_chat(chat_id)
            grade = None
            if chat_data_curr and chat_data_curr.get("messages"):
                last_msg = chat_data_curr["messages"][-1]
                if last_msg["role"] == "assistant":
                    content_data = get_message_content(last_msg)
                    ans_text = content_data["content"]
                    ans_sources = content_data.get("sources", [])
                    context_chunks = [{"chunk_id": s["chunk_id"], "text": s["text"], "filename": s["filename"]} for s in ans_sources]
                    grade = self.orchestrator.llm.grade_answer(prompt, ans_text, context_chunks)

            # 6. Save
            chat_data = self.chat_manager.load_chat(chat_id)
            if chat_data:
                self.chat_manager.save_chat(
                    chat_id, 
                    chat_data.get("messages", []), 
                    insights=new_insights,
                    pooled_data=pooled_data,
                    proactive_thought=proactive_thought,
                    trace=pooled_data.get("trace"),
                    grade=grade
                )
            self.statuses[chat_id] = "✅ Complete"
        except Exception as e:
            raggedy_logger.error(f"Background worker failed: {e}")
            self.statuses[chat_id] = "❌ Resource Issue/Failed"

if "bg_worker" not in st.session_state:
    st.session_state.bg_worker = BackgroundWorker(st.session_state.orchestrator, st.session_state.chat_manager, st.session_state.monitor)

# Initialize copy box state
if "show_copy_idx" not in st.session_state:
    st.session_state.show_copy_idx = None

# Custom CSS for modern/sleek UI
st.markdown("""
    <style>
    /* Main layout adjustments */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
        min-width: 300px !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #fafafa;
    }

    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c9d1d9;
    }
    
    /* Global button styling - transparent and sleek */
    .stButton button {
        border-radius: 4px;
        transition: all 0.2s;
        border: none;
        background-color: transparent;
        color: #8b949e;
        padding: 0.25rem 0.5rem;
        font-size: 1.1rem;
    }
    
    .stButton button:hover {
        color: #58a6ff;
        background-color: #21262d;
        border: none;
    }

    .stButton button:active {
        background-color: #30363d;
        color: #58a6ff;
    }

    .stButton button:focus:not(:active) {
        border: none;
        box-shadow: none;
        color: #58a6ff;
    }
    
    /* Navigation icons/buttons in sidebar */
    .nav-container {
        display: flex;
        flex-direction: row;
        gap: 5px;
        margin-bottom: 20px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: none;
    }
    
    /* Action button grouping */
    .action-row {
        display: flex;
        gap: 2px;
        align-items: center;
        margin-top: 5px;
    }
    
    /* Version nav grouping */
    .version-nav {
        display: flex;
        gap: 8px;
        align-items: center;
        background: #161b22;
        padding: 2px 8px;
        border-radius: 4px;
        width: fit-content;
        border: 1px solid #30363d;
    }
    
    /* Small simple buttons for actions */
    .small-btn button {
        padding: 2px 4px !important;
        min-height: 28px !important;
        height: 28px !important;
        font-size: 1rem !important;
        background: transparent !important;
        border: none !important;
        color: #8b949e !important;
    }

    .small-btn button:hover {
        color: #58a6ff !important;
    }
    
    /* Tiny margin navigation style */
    .side-nav-icons {
        display: flex;
        flex-direction: column;
        gap: 15px;
        padding-right: 10px;
        border-right: 1px solid #30363d;
    }

    /* Fix for text area in dark mode */
    .stTextArea textarea {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }

    /* Code block styling */
    .stCodeBlock {
        border: 1px solid #30363d !important;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    
    /* Citation styling and Tooltip */
    .citation-link {
        color: #58a6ff !important;
        font-weight: bold !important;
        text-decoration: none !important;
        vertical-align: super;
        font-size: 0.8em;
        margin-left: 1px;
        position: relative;
        cursor: help;
        display: inline-block;
    }
    
    .citation-link:hover {
        color: #79c0ff !important;
    }

    /* Bubble Popup (Tooltip) */
    .citation-link .tooltip-content {
        visibility: hidden;
        width: 300px;
        background-color: #161b22;
        color: #c9d1d9;
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 150%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        font-size: 0.85rem;
        line-height: 1.4;
        font-weight: normal;
        pointer-events: none;
    }

    .citation-link .tooltip-content::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #30363d transparent transparent transparent;
    }

    .citation-link:hover .tooltip-content {
        visibility: visible;
        opacity: 1;
        pointer-events: auto;
    }

    .tooltip-header {
        font-weight: bold;
        color: #58a6ff;
        margin-bottom: 5px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 3px;
        display: flex;
        justify-content: space-between;
    }

    .tooltip-body {
        max-height: 150px;
        overflow-y: auto;
    }
    
    /* Source anchor styling */
    .source-anchor {
        scroll-margin-top: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load active chat messages
chat_data = st.session_state.chat_manager.load_chat(st.session_state.active_chat_id)
if chat_data:
    st.session_state.messages = chat_data.get("messages", [])
    st.session_state.chat_title = chat_data.get("title", "New Chat")
else:
    st.session_state.messages = []
    st.session_state.chat_title = "New Chat"

# Navigation
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"

def save_uploaded_file(uploaded_file):
    raw_dir = st.session_state.orchestrator.raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join(raw_dir, uploaded_file.name)

def get_message_content(message, version_idx=None):
    if "versions" in message:
        idx = version_idx if version_idx is not None else message.get("active_version", 0)
        return message["versions"][idx]
    # Compatibility with old format
    return {"content": message.get("content", ""), "sources": message.get("sources", [])}

def render_system_info(concise=False):
    stats = st.session_state.monitor.get_system_stats()
    
    if concise:
        cpu = stats["cpu_percent"]
        ram = stats["ram_percent"]
        vram = stats.get("vram_percent", "N/A")
        st.caption(f"C:{cpu}% | R:{ram}% | V:{vram}%")
        return

    with st.container(border=True):
        st.write("**System Resources**")
        
        # CPU
        st.write(f"CPU Usage: {stats['cpu_percent']}%")
        st.progress(stats['cpu_percent'] / 100)
        
        # RAM
        st.write(f"RAM: {stats['ram_used_gb']} / {stats['ram_total_gb']} GB ({stats['ram_percent']}%)")
        st.progress(stats['ram_percent'] / 100)
        
        # VRAM
        if stats.get("vram_available"):
            st.write(f"VRAM: {stats['vram_used_gb']} / {stats['vram_total_gb']} GB ({stats['vram_percent']}%)")
            st.progress(stats['vram_percent'] / 100)
            if stats['vram_percent'] > 90:
                st.warning("⚠️ High VRAM usage. Background tasks may be paused.")
        else:
            st.write("VRAM: Not Detected (Using CPU)")
        
        # LLM Status
        llm_avail = st.session_state.orchestrator.llm.is_available()
        status_color = "green" if llm_avail else "red"
        status_text = "Online" if llm_avail else "Offline"
        st.markdown(f"LLM Server: :{status_color}[{status_text}]")

def render_citations(text, sources):
    import re
    
    def replace_citation(match):
        try:
            idx = int(match.group(1))
            source = next((s for s in sources if s['index'] == idx), None)
            if source:
                filename = source['filename']
                # Escape HTML for tooltip to prevent breaking the UI
                snippet = source['text'].replace('<', '&lt;').replace('>', '&gt;')
                text_snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
                method = source['method'].upper()
                return f'''
                <span class="citation-link">
                    ⁽{idx}⁾
                    <div class="tooltip-content">
                        <div class="tooltip-header">
                            <span>Source {idx}: {filename}</span>
                            <span style="font-size: 0.8em; opacity: 0.8;">{method}</span>
                        </div>
                        <div class="tooltip-body">{text_snippet}</div>
                    </div>
                </span>
                '''
        except:
            pass
        return match.group(0)

    # Robust regex for [Source X] or [Source  X]
    return re.sub(r'\[Source\s*(\d+)\]', replace_citation, text)

def format_history_for_llm(messages):
    history = []
    for msg in messages:
        content_data = get_message_content(msg)
        history.append({
            "role": msg["role"],
            "content": content_data["content"]
        })
    return history

# Sidebar for Navigation and Global Status
with st.sidebar:
    st.markdown("### ⚙ RAGGEDY")
    
    # Tiny margin navigation layout
    nav_col, menu_col = st.columns([1, 4])
    
    with nav_col:
        st.markdown('<div class="side-nav-icons">', unsafe_allow_html=True)
        if st.button("◣", key="nav_chat", help="Chat", use_container_width=True):
            st.session_state.active_tab = "Chat"
            st.rerun()
        if st.button("▤", key="nav_files", help="Files", use_container_width=True):
            st.session_state.active_tab = "Files"
            st.rerun()
        if st.button("⚙", key="nav_models", help="Models", use_container_width=True):
            st.session_state.active_tab = "Models"
            st.rerun()
        if st.button("☰", key="nav_logs", help="Logs", use_container_width=True):
            st.session_state.active_tab = "Logs"
            st.rerun()
        if st.button("🌐", key="nav_map", help="Knowledge Map", use_container_width=True):
            st.session_state.active_tab = "Map"
            st.rerun()
        
        st.divider()
        render_system_info(concise=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with menu_col:
        if st.session_state.active_tab == "Chat":
            st.subheader("Conversations")
            if st.button("➕ New", key="new_chat_btn", use_container_width=True):
                new_id = st.session_state.chat_manager.create_chat()
                st.session_state.active_chat_id = new_id
                st.rerun()
            
            chats = st.session_state.chat_manager.list_chats()
            for c in chats:
                col1, col2 = st.columns([4, 1])
                label = f"{c['title']}"
                if c['id'] == st.session_state.active_chat_id:
                    label = f"**{c['title']}**"
                
                if col1.button(label, key=f"select_{c['id']}", use_container_width=True):
                    st.session_state.active_chat_id = c['id']
                    st.rerun()
                
                if col2.button("🗑️", key=f"delete_chat_{c['id']}"):
                    st.session_state.chat_manager.delete_chat(c['id'])
                    if st.session_state.active_chat_id == c['id']:
                        st.session_state.active_chat_id = None
                    st.rerun()
        
        elif st.session_state.active_tab == "Files":
            st.subheader("Data Ops")
            st.caption("Manage your knowledge pools and document indexing.")
            
        elif st.session_state.active_tab == "Models":
            st.subheader("Model Ops")
            st.caption("Discovery and serving controls.")
            
        elif st.session_state.active_tab == "Logs":
            st.subheader("System Logs")
            st.caption("Real-time telemetry.")

    st.divider()
    
    # LLM Status
    llm_available = st.session_state.orchestrator.llm.is_available()
    if llm_available:
        current_model = st.session_state.orchestrator.llm.current_model
        nickname = st.session_state.orchestrator.llm.manager.get_nickname(current_model) if current_model else "External"
        st.success(f"• {nickname}")
    else:
        st.error("• Offline")
    
    st.divider()
    if st.button("↺ Reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- TAB: Chat ---
if st.session_state.active_tab == "Chat":
    llm_available = st.session_state.orchestrator.llm.is_available()
    current_model = st.session_state.orchestrator.llm.current_model
    model_display = ""
    if llm_available and current_model:
        nickname = st.session_state.orchestrator.llm.manager.get_nickname(current_model)
        model_display = f" — `{nickname}`"
    
    st.title(f"💬 RAGGEDY Chat{model_display}")
    
    # Knowledge Base Selection for Chat
    with st.expander("▤ Selective Knowledge Base", expanded=False):
        ingested_docs = st.session_state.orchestrator.ingestor.get_ingested_documents()
        pool_manager = st.session_state.orchestrator.pool_manager
        pools = pool_manager.list_pools()
        
        selected_doc_ids = set()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**By Pool:**")
            for pool_name in pools:
                if st.checkbox(f"Pool: {pool_name}", key=f"chat_pool_{pool_name}"):
                    selected_doc_ids.update(pool_manager.get_pool_docs(pool_name))
        
        with col2:
            st.write("**Individual Files:**")
            if ingested_docs:
                for doc in ingested_docs:
                    doc_id = doc['doc_id']
                    # Default value is True if no pools selected, or if doc is in selected pools
                    is_in_pool = doc_id in selected_doc_ids
                    is_selected = st.checkbox(f"{doc['filename']}", value=is_in_pool, key=f"chat_sel_{doc_id}")
                    if is_selected:
                        selected_doc_ids.add(doc_id)
            else:
                st.info("No documents ingested yet.")
        
        selected_doc_ids = list(selected_doc_ids)
        st.caption(f"Selected: {len(selected_doc_ids)} documents")

    # Chat & Retrieval Settings
    with st.expander("⚙ Chat & Retrieval Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Model Parameters**")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
            top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.05)
            n_predict = st.number_input("Max Tokens", 1, 4096, 1024)
        with c2:
            st.write("**Retrieval Parameters**")
            top_k_search = st.slider("Search Results (Initial)", 5, 100, 20)
            top_k_rerank = st.slider("Final Context (Reranked)", 1, 20, 7)
        with c3:
            st.write("**System Prompt**")
            system_prompt = st.text_area(
                "Custom System Instructions", 
                value="", 
                placeholder="Leave blank for default RAGGEDY prompt...",
                height=150
            )
            if not system_prompt:
                system_prompt = None

    # Main Chat Area
    chat_col, side_col = st.columns([4, 1.2])
    
    with chat_col:
        # Display chat messages
        for msg_idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                content_data = get_message_content(message)
                content = content_data["content"]
                sources = content_data.get("sources", [])
                
                if st.session_state.get("editing_idx") == msg_idx:
                    # EDIT MODE
                    new_content = st.text_area("Edit message:", value=content, key=f"edit_area_{msg_idx}", height=100)
                    ec1, ec2, _ = st.columns([1, 1, 4])
                    if ec1.button("✅ Save", key=f"save_edit_{msg_idx}", type="primary"):
                        if "versions" not in message:
                            message["versions"] = [{"content": content, "sources": sources}]
                        message["versions"].append({"content": new_content})
                        message["active_version"] = len(message["versions"]) - 1
                        st.session_state.messages = st.session_state.messages[:msg_idx+1]
                        st.session_state.chat_manager.save_chat(st.session_state.active_chat_id, st.session_state.messages)
                        st.session_state.trigger_response = True
                        del st.session_state.editing_idx
                        st.rerun()
                    if ec2.button("❌ Cancel", key=f"cancel_edit_{msg_idx}"):
                        del st.session_state.editing_idx
                        st.rerun()
                else:
                    # View MODE
                    display_content = render_citations(content, sources)
                    st.markdown(display_content, unsafe_allow_html=True)
                    
                    # Message Actions Row
                    st.markdown('<div class="action-row">', unsafe_allow_html=True)
                    act_c1, act_c2, v_col, _ = st.columns([0.6, 0.6, 3, 6])
                    
                    if message["role"] == "user":
                        with act_c1:
                            st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                            if st.button("✎", key=f"edit_btn_{msg_idx}", help="Edit message"):
                                st.session_state.editing_idx = msg_idx
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with act_c2:
                        st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                        if st.button("❐", key=f"copy_btn_{msg_idx}", help="Toggle copyable text"):
                            if st.session_state.show_copy_idx == msg_idx:
                                st.session_state.show_copy_idx = None
                            else:
                                st.session_state.show_copy_idx = msg_idx
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Version navigation (Combined small buttons)
                    if "versions" in message and len(message["versions"]) > 1:
                        with v_col:
                            st.markdown('<div class="version-nav">', unsafe_allow_html=True)
                            vc1, vc2, vc3 = st.columns([1, 2, 1])
                            curr_v = message["active_version"]
                            total_v = len(message["versions"])
                            with vc1:
                                st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                                if st.button("◀", key=f"prev_{msg_idx}", disabled=(curr_v == 0)):
                                    message["active_version"] -= 1
                                    st.session_state.chat_manager.save_chat(st.session_state.active_chat_id, st.session_state.messages)
                                    st.rerun()
                                st.markdown('</div>', unsafe_allow_html=True)
                            with vc2:
                                st.caption(f"{curr_v + 1}/{total_v}")
                            with vc3:
                                st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                                if st.button("▶", key=f"next_{msg_idx}", disabled=(curr_v == total_v - 1)):
                                    message["active_version"] += 1
                                    st.session_state.chat_manager.save_chat(st.session_state.active_chat_id, st.session_state.messages)
                                    st.rerun()
                                st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Copy box (Conditional)
                    if st.session_state.show_copy_idx == msg_idx:
                        st.code(content, language="text")

                    # Sourced Evidence section
                    st.markdown("---")
                    with st.expander("🔍 Sourced Evidence (All Context Fed to AI)", expanded=False):
                        if sources:
                            for src in sources:
                                # Color coding for methods
                                method_label = src['method'].upper()
                                method_color = "#58a6ff" # Default blue
                                if "GRAPH" in method_label:
                                    method_color = "#d2a8ff" # Purple
                                elif "VECTOR" in method_label:
                                    method_color = "#3fb950" # Green
                                elif "LEXICAL" in method_label:
                                    method_color = "#d29922" # Yellow

                                st.markdown(f'<div id="source-{src["index"]}" class="source-anchor"></div>', unsafe_allow_html=True)
                                st.markdown(f"**[{src['index']}]** {src['filename']} — :{method_color}[{method_label}]")
                                st.info(src['text'])
                                st.caption(f"Chunk ID: `{src['chunk_id']}`")
                        else:
                            st.info("No source documents were retrieved for this query. The AI used its internal knowledge or general context.")

    with side_col:
        st.markdown('<div style="background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d;">', unsafe_allow_html=True)
        render_system_info()
        st.divider()
        st.markdown("### 👁️ Contextual Insights")
        
        chat_id = st.session_state.active_chat_id
        chat_data_full = st.session_state.chat_manager.load_chat(chat_id)
        
        # Display background worker status
        bg_status = st.session_state.bg_worker.statuses.get(chat_id)
        if bg_status:
            if "Complete" in bg_status:
                st.caption(f"Status: {bg_status}")
            else:
                st.markdown(f'<p style="color: #58a6ff; font-size: 0.8em;">{bg_status}...</p>', unsafe_allow_html=True)
        
        st.caption("Live concepts, terms, and pooled ideas from your conversation.")
        st.divider()
        
        # In a real implementation, we'd pull these from the chat manager or a background worker
        chat_id = st.session_state.active_chat_id
        insights = st.session_state.chat_manager.load_chat(chat_id).get("insights", [])
        
        if insights:
            for insight in insights:
                if isinstance(insight, dict) and 'term' in insight:
                    with st.expander(f"🔹 {insight['term']}", expanded=True):
                        st.write(insight.get('description', ''))
                        if insight.get('relevance'):
                            st.progress(insight['relevance'] / 100)
        else:
            st.info("Start chatting to see live concepts and pooled data represented here.")
            
        # Discovery Trace Section
        trace = chat_data_full.get("trace")
        if trace:
            st.divider()
            st.markdown("### 🕸️ Discovery Trace")
            with st.expander("🔍 Path to Findings", expanded=False):
                if trace.get("entities"):
                    st.write("**Entities Identified:**")
                    st.caption(", ".join(trace["entities"]))
                
                if trace.get("graph_expansions"):
                    st.write("**Graph Expansions:**")
                    for exp in trace["graph_expansions"][:3]:
                        st.caption(f"↳ Linked via graph to `{exp['filename']}`")
                
                if trace.get("idea_pools"):
                    st.write("**Idea Pooling:**")
                    for pool in trace["idea_pools"]:
                        st.caption(f"↳ Searched `{pool['term']}` → Found {pool['count']} chunks")

        # Answer Grading Section
        grade = chat_data_full.get("grade")
        if grade:
            st.divider()
            st.markdown("### 🎯 Answer Grade")
            with st.expander(f"Score: {grade.get('total_score', 'N/A')}/10", expanded=True):
                st.write(f"*{grade.get('critique', '')}*")
                cols = st.columns(3)
                cols[0].metric("Faithful", grade.get("faithfulness", 0))
                cols[1].metric("Ground", grade.get("grounding", 0))
                cols[2].metric("Complete", grade.get("completeness", 0))
        
        # Proactive AI Reflection section
        pooled_data = chat_data_full.get("pooled_data")
        proactive_thought = chat_data_full.get("proactive_thought")
        
        if (pooled_data and pooled_data.get("chunks")) or proactive_thought:
            # Check if there are new chunks not cited in the last assistant message
            last_msg = st.session_state.messages[-1] if st.session_state.messages else None
            if last_msg and last_msg["role"] == "assistant":
                content_data = get_message_content(last_msg)
                last_chunk_ids = {s['chunk_id'] for s in content_data.get("sources", [])}
                new_chunks_count = 0
                if pooled_data:
                    new_chunks_count = sum(1 for c in pooled_data["chunks"] if c['chunk_id'] not in last_chunk_ids)
                
                if new_chunks_count > 0 or proactive_thought:
                    st.divider()
                    st.markdown("#### ✨ AI Reflection")
                    
                    if proactive_thought:
                        st.info(f"*{proactive_thought}*")
                    
                    if new_chunks_count > 0:
                        st.caption(f"Found {new_chunks_count} additional relevant data points.")
                    
                    if st.button("🔍 Refine with Pooled Data", use_container_width=True, help="Re-generate answer using expanded context from background pooling"):
                        st.session_state.trigger_proactive = True
                        st.rerun()

        # Yield Section (Export)
        st.divider()
        st.markdown("#### 📂 Yield & Export")
        md_content = st.session_state.chat_manager.export_as_markdown(st.session_state.active_chat_id)
        st.download_button(
            label="📥 Download Research Brief",
            data=md_content,
            file_name=f"Research_Brief_{st.session_state.active_chat_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Helper to generate and stream response
    def generate_and_stream(prompt, history, doc_ids, params):
        with st.chat_message("assistant"):
            # Step 1: START User Request Immediately
            status_container = st.empty()
            status_container.status("📂 Gathering initial search results...", expanded=False)
            
            # Prepare result with sources (Stage A only now, as updated in flow.py)
            result = st.session_state.orchestrator.ask(
                prompt, 
                history=history,
                doc_ids=doc_ids,
                stream=True,
                **params
            )
            
            status_container.empty()
            
            # Create a placeholder for the "Stop" button and status
            stop_col, status_col = st.columns([1, 5])
            stop_pressed = stop_col.button("⏹", key=f"stop_{len(st.session_state.messages)}")
            
            stream_gen = result["answer"]
            sources = result["sources"]
            
            # Stream the answer
            full_response = ""
            resp_container = st.empty()
            bg_started = False
            
            for chunk in stream_gen:
                if not bg_started:
                    # Step 2: START Background analysis and pooling ONLY after stream begins
                    # Only start if LLM is online and there are docs or if we want insights
                    if st.session_state.orchestrator.llm.is_available():
                        st.session_state.bg_worker.start_task(
                            st.session_state.active_chat_id,
                            prompt,
                            history,
                            result.get("query_emb"),
                            result.get("all_candidate_chunks_map", {}),
                            result.get("chunk_methods", {}),
                            doc_ids
                        )
                    bg_started = True

                if stop_pressed or st.session_state.get(f"stop_{len(st.session_state.messages)}"):
                    full_response += "... [Interrupted]"
                    display_text = render_citations(full_response, sources)
                    resp_container.markdown(display_text, unsafe_allow_html=True)
                    break
                full_response += chunk
                display_text = render_citations(full_response, sources)
                resp_container.markdown(display_text + "▌", unsafe_allow_html=True)
            
            display_text = render_citations(full_response, sources)
            resp_container.markdown(display_text, unsafe_allow_html=True)
            
            # Sourced Evidence section (streaming)
            st.markdown("---")
            with st.expander("🔍 Sourced Evidence (All Context Fed to AI)", expanded=False):
                if sources:
                    for src in sources:
                        method_label = src['method'].upper()
                        method_color = "#58a6ff"
                        if "GRAPH" in method_label:
                            method_color = "#d2a8ff"
                        elif "VECTOR" in method_label:
                            method_color = "#3fb950"
                        elif "LEXICAL" in method_label:
                            method_color = "#d29922"

                        st.markdown(f'<div id="source-streaming-{src["index"]}" class="source-anchor"></div>', unsafe_allow_html=True)
                        st.markdown(f"**[{src['index']}]** {src['filename']} — :{method_color}[{method_label}]")
                        st.info(src['text'])
                        st.caption(f"Chunk ID: `{src['chunk_id']}`")
                else:
                    st.info("No source documents were retrieved for this query. The AI used its internal knowledge or general context.")
            
            # Save to messages
            new_msg = {
                "role": "assistant",
                "versions": [{"content": full_response, "sources": sources}],
                "active_version": 0
            }
            st.session_state.messages.append(new_msg)
            st.session_state.chat_manager.save_chat(st.session_state.active_chat_id, st.session_state.messages)
            st.rerun()

    # Show background status above chat input
    bg_status = st.session_state.bg_worker.statuses.get(st.session_state.active_chat_id)
    if bg_status and "Complete" not in bg_status and "Issue" not in bg_status:
        st.markdown(f'<p style="color: #58a6ff; font-size: 0.8em; margin-bottom: -10px; margin-left: 5px;">⚡ {bg_status}...</p>', unsafe_allow_html=True)

    # Chat input and Triggered response
    prompt = st.chat_input("Ask RAGGEDY...")
    
    # Handle triggered response (from edit)
    if st.session_state.get("trigger_response"):
        del st.session_state.trigger_response
        last_user_msg = st.session_state.messages[-1]
        last_content = get_message_content(last_user_msg)["content"]
        
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "n_predict": n_predict,
            "top_k_search": top_k_search,
            "top_k_rerank": top_k_rerank,
            "system_prompt": system_prompt
        }
        generate_and_stream(last_content, format_history_for_llm(st.session_state.messages[:-1]), selected_doc_ids, params)
        st.rerun()

    # Handle proactive trigger (AI Reflection)
    if st.session_state.get("trigger_proactive"):
        del st.session_state.trigger_proactive
        chat_data = st.session_state.chat_manager.load_chat(st.session_state.active_chat_id)
        pooled_data = chat_data.get("pooled_data", {})
        
        if pooled_data and pooled_data.get("chunks"):
            user_msg = next((m for m in reversed(st.session_state.messages) if m['role'] == 'user'), None)
            if user_msg:
                user_content = get_message_content(user_msg)["content"]
                params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "n_predict": n_predict,
                    "top_k_search": top_k_search,
                    "top_k_rerank": top_k_rerank,
                    "system_prompt": system_prompt,
                    "use_pooled_data": pooled_data
                }
                st.toast("Refining answer with expanded background context...")
                generate_and_stream(user_content, format_history_for_llm(st.session_state.messages[:-1]), selected_doc_ids, params)
                st.rerun()

    if prompt:
        st.chat_message("user").markdown(prompt)
        # Create user message in new format
        user_msg = {
            "role": "user",
            "versions": [{"content": prompt}],
            "active_version": 0
        }
        st.session_state.messages.append(user_msg)
        st.session_state.chat_manager.save_chat(st.session_state.active_chat_id, st.session_state.messages)

        params = {
            "temperature": temperature,
            "top_p": top_p,
            "n_predict": n_predict,
            "top_k_search": top_k_search,
            "top_k_rerank": top_k_rerank,
            "system_prompt": system_prompt
        }
        generate_and_stream(prompt, format_history_for_llm(st.session_state.messages[:-1]), selected_doc_ids, params)
        st.rerun()

# --- TAB: Files ---
elif st.session_state.active_tab == "Files":
    st.title("📁 Document Management")
    
    from utils.auditor import SystemAuditor
    auditor = SystemAuditor()
    
    pool_manager = st.session_state.orchestrator.pool_manager
    ingestor = st.session_state.orchestrator.ingestor
    
    # Audit Status Header
    with st.container(border=True):
        ac1, ac2, ac3 = st.columns([1, 2, 1])
        with ac1:
            st.metric("System Health", f"{st.session_state.get('audit_score', '??')}%")
        with ac2:
            st.write("**Integrity Audit**")
            st.caption("Verifying consistency across raw files, chunks, and embeddings.")
        with ac3:
            if st.button("🔍 Run Audit", use_container_width=True):
                report = auditor.perform_audit()
                st.session_state.audit_report = report
                st.session_state.audit_score = report["consistency_score"]
                st.rerun()
    
    if st.session_state.get("audit_report"):
        report = st.session_state.audit_report
        if report["consistency_score"] < 100:
            st.warning(f"Audit found issues in {len([d for d in report['documents'] if d['status'] != 'OK'])} documents.")
            with st.expander("View Audit Details"):
                for doc in report["documents"]:
                    if doc["status"] != "OK":
                        st.write(f"❌ **{doc.get('filename', doc['doc_id'])}**: {', '.join(doc['missing'])}")
                if report["orphaned_chunks"] > 0 or report["orphaned_embeddings"] > 0:
                    st.write(f"🧹 Orphans: {report['orphaned_chunks']} chunks, {report['orphaned_embeddings']} embeddings.")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        # --- Section: Ingestion ---
        with st.container(border=True):
            st.subheader("📥 Upload & Ingest")
            uploaded_files = st.file_uploader("Upload new documents", accept_multiple_files=True)
            if st.button("🚀 Process & Ingest"):
                if uploaded_files:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    new_docs = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Ingesting {uploaded_file.name}...")
                        save_uploaded_file(uploaded_file)
                        doc_data = ingestor.ingest_file(uploaded_file.name)
                        new_docs.append(doc_data)
                        progress_bar.progress((i + 0.5) / len(uploaded_files))
                    
                    status_text.text("Building indices and graph...")
                    st.session_state.orchestrator.process_new_documents(new_docs)
                    progress_bar.progress(1.0)
                    
                    st.success("Ingestion complete!")
                    st.rerun()
                else:
                    st.warning("Please upload files first.")

        # --- Section: Data Pools ---
        with st.container(border=True):
            st.subheader("🏊 Data Pools")
            
            # Create new pool
            with st.form("create_pool_form"):
                new_pool_name = st.text_input("New Pool Name")
                if st.form_submit_button("Create Pool"):
                    if new_pool_name:
                        pool_manager.create_pool(new_pool_name)
                        st.success(f"Pool '{new_pool_name}' created!")
                        st.rerun()
            
            # List and manage pools
            pools = pool_manager.list_pools()
            if pools:
                for pool_name in pools:
                    with st.expander(f"Pool: {pool_name}"):
                        p_docs = pool_manager.get_pool_docs(pool_name)
                        st.write(f"Documents: {len(p_docs)}")
                        if p_docs:
                            for d_id in p_docs:
                                # Find filename
                                d_data = ingestor.get_document_data(d_id)
                                fname = d_data.get('metadata', {}).get('filename', d_id)
                                c1, c2 = st.columns([4, 1])
                                c1.caption(f"- {fname}")
                                if c2.button("❌", key=f"rm_{pool_name}_{d_id}"):
                                    pool_manager.remove_from_pool(pool_name, d_id)
                                    st.rerun()
                        
                        if st.button(f"🗑️ Delete Pool: {pool_name}", key=f"del_pool_{pool_name}"):
                            pool_manager.delete_pool(pool_name)
                            st.rerun()
            else:
                st.info("No pools created yet.")

    with col2:
        # --- Section: Ingested Documents ---
        st.subheader("📚 Ingested Documents")
        ingested_docs = ingestor.get_ingested_documents()
        if ingested_docs:
            for doc in ingested_docs:
                doc_id = doc['doc_id']
                with st.expander(f"📄 {doc['filename']}"):
                    st.caption(f"ID: {doc_id} | Ingested: {doc['ingested_at']}")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Processor:** `{doc.get('processor', 'Unknown')}`")
                        st.write(f"**Type:** `{doc.get('type', 'Unknown')}`")
                    with c2:
                        if 'pages' in doc: st.write(f"**Pages:** {doc['pages']}")
                        if 'conversations' in doc: st.write(f"**Conversations:** {doc['conversations']}")
                        if 'paragraphs' in doc: st.write(f"**Paragraphs:** {doc['paragraphs']}")

                    # Pool Assignment
                    st.divider()
                    st.write("**Pools:**")
                    all_pools = pool_manager.list_pools()
                    doc_pools = [p for p in all_pools if doc_id in pool_manager.get_pool_docs(p)]
                    
                    # Multiselect for pools
                    selected_pools = st.multiselect(
                        "Assign to Pools", 
                        options=all_pools, 
                        default=doc_pools,
                        key=f"assign_{doc_id}"
                    )
                    
                    # Update pools if changed
                    if set(selected_pools) != set(doc_pools):
                        if st.button("Update Pools", key=f"up_pools_{doc_id}"):
                            # Remove from pools not selected
                            for p in doc_pools:
                                if p not in selected_pools:
                                    pool_manager.remove_from_pool(p, doc_id)
                            # Add to pools selected
                            for p in selected_pools:
                                if p not in doc_pools:
                                    pool_manager.add_to_pool(p, doc_id)
                            st.rerun()

                    # Indexing Status
                    st.divider()
                    status = ["Lexical"]
                    full_data = ingestor.get_document_data(doc_id)
                    logs = full_data.get("ingestion_logs", [])
                    
                    if any("vector embeddings" in l.lower() for l in logs):
                        status.append("Vector")
                    if any("graph" in l.lower() for l in logs):
                        status.append("Graph")
                    
                    st.info(f"Indexing Status: {' | '.join(status)}")
                    
                    # Action Buttons
                    act1, act2, act3 = st.columns(3)
                    if act1.button("🔄 Rerun", key=f"rerun_{doc_id}"):
                        with st.spinner("Re-processing..."):
                            # To rerun, we just call ingest_file again (it overwrites)
                            # then process_new_documents for that specific doc
                            doc_data = ingestor.ingest_file(doc['filename'])
                            st.session_state.orchestrator.process_new_documents([doc_data])
                            st.success("Re-processing complete!")
                            st.rerun()
                    
                    if act2.button("🗑️ Delete", key=f"del_{doc_id}", type="secondary"):
                        st.session_state.orchestrator.delete_document(doc_id)
                        st.rerun()
                    
                    if act3.button("📝 Logs", key=f"logs_{doc_id}"):
                        st.write("**Processing History:**")
                        for log in logs:
                            st.caption(f"✅ {log}")
        else:
            st.info("No documents ingested yet.")

# --- TAB: Models ---
elif st.session_state.active_tab == "Models":
    st.title("🛠️ Model Management")
    
    manager = st.session_state.orchestrator.llm.manager
    
    # 1. Local Models Management
    st.subheader("📁 Local Models")
    local_models = manager.list_local_models()
    if local_models:
        for model_file in local_models:
            nickname = manager.get_nickname(model_file)
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.write(f"**{nickname}**")
                    st.caption(f"File: `{model_file}`")
                with c2:
                    new_nick = st.text_input("New Nickname", value=nickname, key=f"nick_{model_file}", label_visibility="collapsed")
                    if new_nick != nickname:
                        if st.button("Save", key=f"save_{model_file}"):
                            manager.set_nickname(model_file, new_nick)
                            st.rerun()
                with c3:
                    if st.button("🗑️ Delete", key=f"del_{model_file}", type="secondary"):
                        manager.delete_model(model_file)
                        st.success(f"Deleted {model_file}")
                        st.rerun()
    else:
        st.info("No models found locally.")

    st.divider()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🚀 LLM Server Control")
        if llm_available:
            st.success("🟢 Server is Online")
            # Show which model is likely running (if we can detect it, for now we just show a stop button)
            if st.button("🛑 Stop Local Server", type="primary", use_container_width=True):
                st.session_state.orchestrator.llm.stop_local_server()
                st.rerun()
        else:
            st.warning("🔴 Server is Offline")
            if local_models:
                model_options = {manager.get_nickname(m): m for m in local_models}
                selected_nick = st.selectbox("Select model to start", options=list(model_options.keys()))
                selected_file = model_options[selected_nick]
                
                # Context Length Setting
                n_ctx = st.select_slider(
                    "Context Length (n_ctx)", 
                    options=[2048, 4096, 8192, 12288, 16384, 32768], 
                    value=4096,
                    help="Larger context allows more information but uses more VRAM/RAM."
                )
                
                if st.button("▶️ Start Server", type="primary", use_container_width=True):
                    with st.spinner(f"Starting server with {selected_nick} (ctx: {n_ctx})..."):
                        if st.session_state.orchestrator.llm.start_local_server(selected_file, n_ctx=n_ctx):
                            st.success("Server started!")
                            st.rerun()
                        else:
                            st.error("Failed to start server. See 'Logs' for details.")
                            st.info("💡 Check **userinstruction.md** for installation help.")
            else:
                st.error("No models found. Search and download one below or from recommended list.")

        st.divider()
        st.subheader("🎯 Task Model Assignments")
        st.caption("Assign specific models to background tasks.")
        
        tasks = {
            "reranking": "Reranker Model",
            "embedding": "Embedding Model",
            "chunking": "Chunking Model (LLM-based)",
            "live_notes": "Live Conversation Notes",
            "data_pooling": "Data Pooling Reasoning"
        }
        
        for task_id, task_name in tasks.items():
            current_assigned = manager.get_task_model(task_id)
            options = ["Default / None"] + local_models
            nick_options = ["Default / None"] + [manager.get_nickname(m) for m in local_models]
            
            # Map nickname back to file
            nick_to_file = {manager.get_nickname(m): m for m in local_models}
            nick_to_file["Default / None"] = None
            
            default_idx = 0
            if current_assigned in local_models:
                default_idx = local_models.index(current_assigned) + 1
            
            selected_nick = st.selectbox(f"{task_name}", options=nick_options, index=default_idx, key=f"task_sel_{task_id}")
            if nick_to_file[selected_nick] != current_assigned:
                manager.set_task_model(task_id, nick_to_file[selected_nick])
                st.toast(f"Updated {task_name} assignment.")

    with col2:
        st.subheader("🌟 Recommended Models")
        for mod in manager.recommended_models:
            with st.container(border=True):
                st.write(f"**{mod['name']}**")
                if st.button(f"Download", key=f"rec_{mod['file']}", use_container_width=True):
                    if mod['file'] in local_models:
                        st.warning("Model already exists.")
                    else:
                        progress_bar = st.progress(0)
                        def update_prog(current, total):
                            progress_bar.progress(current / total)
                        with st.spinner(f"Downloading {mod['file']}..."):
                            manager.download_model(mod['url'], mod['file'], progress_callback=update_prog)
                        st.success(f"Downloaded {mod['file']}")
                        st.rerun()

    st.divider()
    
    # 2. Hugging Face Search
    st.subheader("🤗 Search Hugging Face (GGUF)")
    search_query = st.text_input("Search for models (e.g. 'llama-3', 'phi-3', 'qwen')", key="hf_search_input")
    if st.button("🔍 Search", use_container_width=True):
        if search_query:
            with st.spinner(f"Searching for '{search_query}'..."):
                results = manager.search_huggingface(search_query)
                st.session_state.search_results = results
        else:
            st.warning("Please enter a search query.")

    if "search_results" in st.session_state and st.session_state.search_results:
        st.write(f"Found {len(st.session_state.search_results)} repositories:")
        for res in st.session_state.search_results:
            with st.expander(f"📦 {res['id']} ({res['downloads']} downloads)"):
                st.write(f"Author: {res['author']}")
                selected_file = st.selectbox("Select file to download", res['files'], key=f"files_{res['id']}")
                if st.button("📥 Download Selected File", key=f"dl_{res['id']}"):
                    if selected_file in local_models:
                        st.warning("Model already exists.")
                    else:
                        progress_bar = st.progress(0)
                        def update_prog(current, total):
                            progress_bar.progress(current / total)
                        with st.spinner(f"Downloading {selected_file}..."):
                            manager.download_from_hf(res['id'], selected_file, progress_callback=update_prog)
                        st.success(f"Downloaded {selected_file}")
                        st.rerun()
    elif "search_results" in st.session_state:
        st.info("No GGUF models found matching your query.")

# --- TAB: Logs ---
elif st.session_state.active_tab == "Logs":
    st.title("📋 System Logs")
    
    from utils.logger import raggedy_logger
    
    log_files = raggedy_logger.get_all_log_files()
    if log_files:
        selected_log = st.selectbox("Select log file", log_files)
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh Logs"):
                st.rerun()
        
        log_content = raggedy_logger.read_log_file(selected_log)
        st.code(log_content, language="text")
    else:
        st.info("No log files found yet.")

# --- TAB: Map ---
elif st.session_state.active_tab == "Map":
    st.title("🌐 Knowledge Map")
    st.caption("Visualizing conceptual relationships and entity connections.")
    
    graph_store = st.session_state.orchestrator.graph_store
    G = graph_store.graph
    
    if G.number_of_nodes() == 0:
        st.info("Knowledge Graph is empty. Ingest some documents to build it!")
    else:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.write("**Filter & Settings**")
            search_entity = st.text_input("Search Entity", placeholder="e.g. RAGGEDY")
            max_nodes = st.slider("Max Nodes to Show", 10, 200, 50)
            show_labels = st.checkbox("Show Edge Labels", value=True)
            
            st.divider()
            st.write(f"**Total Nodes:** {G.number_of_nodes()}")
            st.write(f"**Total Edges:** {G.number_of_edges()}")

        with col2:
            # Subgraph extraction
            nodes_to_show = []
            if search_entity:
                # Find nodes matching search
                matches = [n for n in G.nodes() if search_entity.lower() in str(n).lower()]
                if matches:
                    # Get neighbors of matches
                    for match in matches:
                        nodes_to_show.append(match)
                        nodes_to_show.extend(list(G.neighbors(match)))
                else:
                    st.warning(f"No entity found matching '{search_entity}'")
            
            if not nodes_to_show:
                # Just show some nodes based on degree
                nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
                nodes_to_show = [n for n, d in nodes_by_degree[:max_nodes]]
            
            subgraph = G.subgraph(nodes_to_show)
            
            # Generate Graphviz DOT code
            dot = "digraph {\n"
            dot += '  graph [bgcolor="#0e1117", rankdir=LR];\n'
            dot += '  node [shape=box, style=filled, fillcolor="#161b22", color="#30363d", fontcolor="#fafafa", fontname="Arial"];\n'
            dot += '  edge [color="#58a6ff", fontcolor="#8b949e", fontname="Arial", fontsize=10];\n'
            
            for node in subgraph.nodes():
                safe_node = str(node).replace('"', '\\"')
                dot += f'  "{safe_node}" [label="{safe_node}"];\n'
            
            for u, v, data in subgraph.edges(data=True):
                safe_u = str(u).replace('"', '\\"')
                safe_v = str(v).replace('"', '\\"')
                label = data.get('relation', '') if show_labels else ''
                safe_label = str(label).replace('"', '\\"')
                dot += f'  "{safe_u}" -> "{safe_v}" [label="{safe_label}"];\n'
            
            dot += "}"
            
            try:
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"Could not render graph: {e}")
                st.info("Try reducing 'Max Nodes' or search for a specific entity.")
                # Fallback: List relationships
                st.write("**Relationships List:**")
                for u, v, data in subgraph.edges(data=True):
                    st.caption(f"• **{u}** --({data.get('relation', '')})--> **{v}**")
