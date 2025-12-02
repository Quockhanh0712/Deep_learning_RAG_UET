import os
from pathlib import Path
import logging

import streamlit as st
from dotenv import load_dotenv

# Cấu hình logging ra terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1) Load .env trước
load_dotenv()

# 2) Sau đó mới import các module dùng GOOGLE_API_KEY
from src.file_manager import load_file, delete_file
from src.vector_store import list_files
from src.rag_pipeline import rag_answer
from src.ui.components import show_context

# Thư mục lưu file upload
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG bge + Chroma + Gemini", layout="wide")

st.title("RAG Demo (bge-large + Chroma + Gemini)")

st.sidebar.header("Quản lý tài liệu")

# ----- Upload file -----
uploaded = st.sidebar.file_uploader("Tải file (PDF/TXT)", type=["pdf", "txt"])
if uploaded is not None:
    logger.info("[UI] file_uploader triggered with %s", uploaded.name)
    save_path = UPLOAD_DIR / uploaded.name
    with open(save_path, "wb") as f:
        f.write(uploaded.read())
    logger.info("[UI] saved to %s", save_path)

    # tránh nạp trùng: kiểm tra các file_id đã có trong Chroma
    existing_files = list_files()
    already_loaded = any(fid.startswith(uploaded.name + "-") for fid in existing_files)

    if already_loaded:
        logger.info("[UI] %s already loaded, skip embedding", uploaded.name)
        st.sidebar.info(f"File {uploaded.name} đã được nạp trước đó, bỏ qua.")
    else:
        file_id = load_file(str(save_path))
        logger.info("[UI] load_file returned file_id=%s", file_id)
        st.sidebar.success(f"Đã nạp file: {uploaded.name} (id: {file_id})")

# ----- Danh sách file trong Chroma -----
files = list_files()
st.sidebar.write("Các file đã nạp:")
for fid in files:
    col1, col2 = st.sidebar.columns([3, 1])
    col1.write(fid)
    if col2.button("Xóa", key=f"del-{fid}"):
        delete_file(fid)
        st.sidebar.warning(f"Đã xóa {fid}")
        st.rerun()

# ----- Hỏi đáp -----
st.header("Hỏi đáp trên tài liệu đã nạp")

question = st.text_area("Nhập câu hỏi", height=120)

if st.button("Hỏi"):
    logger.info("[UI] Ask button clicked with question=%r", question)
    if not question.strip():
        st.error("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang suy nghĩ..."):
            result = rag_answer(question)
        logger.info("[UI] rag_answer finished")

        st.subheader("Trả lời")
        st.write(result["answer"])

        with st.expander("Ngữ cảnh đã dùng"):
            show_context(result["context"], result["metadatas"])
