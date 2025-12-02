"""
Modern Streamlit Chat Interface for RAG Chatbot
Clean, fast, and professional design
"""

import os
from pathlib import Path
import logging
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from src.file_manager import load_file, delete_file
from src.vector_store import list_files, query_documents
from src.llm_client import generate_answer

# Constants
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = int(os.getenv("TOP_K", "5"))

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
    }
    
    .assistant-message {
        background: #f8f9fa;
        color: #1a1a2e;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 85%;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .file-card {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Input styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Context expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)


def build_prompt(question: str, docs: list) -> str:
    """Build the prompt for the LLM."""
    context = "\n\n".join(docs)
    prompt = (
        "Bạn là trợ lý AI thông minh và thân thiện. "
        "Trả lời chi tiết, chính xác và dễ hiểu dựa trên ngữ cảnh được cung cấp.\n"
        "Sử dụng markdown để format câu trả lời đẹp hơn khi cần thiết.\n"
        "Nếu không tìm thấy thông tin phù hợp, hãy thông báo rằng bạn không có đủ thông tin.\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời:"
    )
    return prompt


def process_query(question: str) -> dict:
    """Process a question and return the answer with context."""
    res = query_documents(question, k=TOP_K)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    
    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt)
    
    return {
        "answer": answer,
        "context": docs,
        "metadatas": metadatas,
    }


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False


def render_sidebar():
    """Render the sidebar with document management."""
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin:0;">📚 Quản lý tài liệu</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload
        st.markdown("### 📤 Tải lên tài liệu")
        uploaded = st.file_uploader(
            "Kéo thả hoặc chọn file",
            type=["pdf", "txt"],
            help="Hỗ trợ PDF và TXT",
            label_visibility="collapsed"
        )
        
        if uploaded is not None:
            save_path = UPLOAD_DIR / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.read())
            
            existing_files = list_files()
            already_loaded = any(fid.startswith(uploaded.name + "-") for fid in existing_files)
            
            if already_loaded:
                st.info(f"📄 `{uploaded.name}` đã được nạp trước đó")
            else:
                with st.spinner("⏳ Đang xử lý file..."):
                    try:
                        file_id = load_file(str(save_path))
                        st.success(f"✅ Đã nạp: `{uploaded.name}`")
                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
        
        st.markdown("---")
        
        # List of loaded files
        st.markdown("### 📋 Tài liệu đã nạp")
        files = list_files()
        
        if not files:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #888;">
                <p>📭 Chưa có tài liệu nào</p>
                <p style="font-size: 0.85rem;">Tải lên file để bắt đầu</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**Tổng số:** {len(files)} tài liệu")
            for fid in files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Truncate long filenames
                    display_name = fid[:30] + "..." if len(fid) > 30 else fid
                    st.markdown(f"📄 `{display_name}`")
                with col2:
                    if st.button("🗑️", key=f"del-{fid}", help=f"Xóa {fid}"):
                        delete_file(fid)
                        st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Cài đặt")
        if st.button("🧹 Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #888;">
            <p>🔧 BGE-M3 + ChromaDB + Gemini</p>
            <p>Made with ❤️</p>
        </div>
        """, unsafe_allow_html=True)


def render_chat():
    """Render the main chat interface."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Chatbot</h1>
        <p>Trợ lý AI thông minh - Hỏi đáp trên tài liệu của bạn</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if there are documents
    files = list_files()
    if not files:
        st.warning("⚠️ Chưa có tài liệu nào được nạp. Vui lòng tải lên file từ sidebar để bắt đầu.")
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message animate-in">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
                    
                    # Show context if available
                    if "context" in message and message["context"]:
                        with st.expander("📚 Xem nguồn tham khảo"):
                            for i, (doc, meta) in enumerate(zip(message["context"], message["metadatas"])):
                                source = meta.get("source", "Unknown")
                                chunk_idx = meta.get("chunk_index", i)
                                st.markdown(f"**📖 Nguồn {i+1}:** `{source}` (chunk {chunk_idx})")
                                st.markdown(f"> {doc[:500]}..." if len(doc) > 500 else f"> {doc}")
                                st.markdown("---")
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.chat_input(
            placeholder="💬 Nhập câu hỏi của bạn...",
            key="chat_input"
        )
    
    if user_input:
        if not files:
            st.error("❌ Vui lòng tải lên tài liệu trước khi đặt câu hỏi!")
            return
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Process and get response
        with st.spinner("🤔 Đang suy nghĩ..."):
            try:
                result = process_query(user_input)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "context": result["context"],
                    "metadatas": result["metadatas"]
                })
                
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Đã xảy ra lỗi: {str(e)}"
                })
        
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
