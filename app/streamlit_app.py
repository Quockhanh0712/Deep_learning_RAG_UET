import sys
import os

# --- Add project root to sys.path so Python can find pipeline and app ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import yaml
import json
from pipeline.rag_pipeline import RAGPipeline
from app import ui_components as ui

# --- Load config ---
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TOP_K = cfg.get("pipeline", {}).get("top_k_display", 5)
LOG_FILE = cfg.get("paths", {}).get("query_log", "logs/query_logs.json")

# --- Initialize RAG pipeline ---
rag_pipeline = RAGPipeline(cfg)

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    # --- Run RAG pipeline ---
    answer = rag_pipeline.run(query)
    
    # Lấy top-k chunks nếu muốn hiển thị
    top_chunks = rag_pipeline.retriever.get_top_k(query, top_k=TOP_K)
    
    # --- Display answer ---
    ui.display_answer(answer)
    
    # --- Display top-k chunks ---
    ui.display_chunks(top_chunks, limit=TOP_K)
    
    # --- Save to chat history ---
    st.session_state.chat_history.append({
        "query": query,
        "answer": answer,
        "chunks": top_chunks
    })
    
    # --- Save log ---
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(st.session_state.chat_history[-1])
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)




# --- Display chat history ---
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**Q{i+1}:** {entry['query']}")
        st.markdown(f"**A{i+1}:** {entry['answer']}")
        st.markdown("---")

# --- Clear chat button ---
if st.button("Clear Chat"):
    ui.clear_chat()
    st.success("Chat cleared.")
