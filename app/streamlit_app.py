import sys
import os
import streamlit as st
import yaml
import json
import numpy as np

# --- Add project root to sys.path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.rag_pipeline import RAGPipeline
from app import ui_components as ui

# --- Helper: convert numpy / bytes to Python native for JSON ---
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    else:
        return obj

# --- Clean text for display ---
def clean_text(text):
    return str(text).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

# --- Load config ---
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TOP_K = cfg.get("pipeline", {}).get("top_k_display", 5)
LOG_FILE = cfg.get("paths", {}).get("query_log", "logs/query_logs.json")

# --- Lazy-load RAG pipeline ---
@st.cache_resource
def load_rag_pipeline(cfg):
    return RAGPipeline(cfg)

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

# --- Streamlit UI ---
st.title("📄 RAG Document QA System")

# --- File upload ---
uploaded_text, uploaded_filename = ui.file_uploader_widget()
if uploaded_text:
    st.sidebar.success(f"Uploaded: {uploaded_filename}")
    ui.display_file_preview(uploaded_text)

# --- Chat input ---
query = ui.chat_input_widget("Ask a question about the documents:")

if st.button("Ask") and query.strip():
    # --- Load model only when first needed ---
    if st.session_state.rag_pipeline is None:
        with st.spinner("Loading RAG pipeline..."):
            st.session_state.rag_pipeline = load_rag_pipeline(cfg)
    rag_pipeline = st.session_state.rag_pipeline

    # --- Run pipeline ---
    with st.spinner("Generating answer..."):
        answer = rag_pipeline.run(query)

        # --- Get top-k chunks ---
        top_chunks = rag_pipeline.retriever.get_top_k(query, top_k=TOP_K)

        # --- Convert to JSON-safe types and ensure 'text' exists ---
        top_chunks_serializable = make_json_serializable(top_chunks)
        for c in top_chunks_serializable:
            if "chunk" in c:
                c["text"] = clean_text(c["chunk"])
            elif "text" in c:
                c["text"] = clean_text(c["text"])
            else:
                c["text"] = "N/A"

    # --- Display answer and chunks ---
    ui.display_answer(clean_text(answer))
    ui.display_chunks(top_chunks_serializable, limit=TOP_K)

    # --- Save to chat history ---
    st.session_state.chat_history.append({
        "query": query,
        "answer": answer,
        "chunks": top_chunks_serializable
    })

    # --- Save log ---
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(st.session_state.chat_history[-1])
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# --- Display chat history ---
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**Q{i+1}:** {clean_text(entry['query'])}")
        st.markdown(f"**A{i+1}:** {clean_text(entry['answer'])}")
        st.markdown("---")

# --- Clear chat button ---
if st.button("Clear Chat"):
    ui.clear_chat()
    st.session_state.chat_history = []
    st.success("Chat cleared.")
