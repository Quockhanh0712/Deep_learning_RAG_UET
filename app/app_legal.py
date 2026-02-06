"""
Vietnamese Legal RAG Application

Ứng dụng hỏi đáp pháp luật Việt Nam với:
- Graph Database: Lưu trữ cấu trúc phân cấp (Luật → Điều → Khoản → Điểm)
- Vector Search: Tìm kiếm ngữ nghĩa với embedding 768D
- Hybrid Search: Kết hợp vector + graph context
"""

import os
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after dotenv
from src.hybrid_legal_search import HybridLegalSearch
from src.llm_client import generate_answer
from src.graph_db import LegalGraphDB

# Page config
st.set_page_config(
    page_title="⚖️ Trợ lý Pháp luật Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E3A5F;
        margin: 0.5rem 0;
    }
    .source-tag {
        background: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #1565c0;
    }
    .score-badge {
        background: #c8e6c9;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #2e7d32;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'hybrid_search' not in st.session_state:
    st.session_state.hybrid_search = None
if 'graph_db' not in st.session_state:
    st.session_state.graph_db = None


@st.cache_resource
def load_search_engine():
    """Load search engine (cached)"""
    return HybridLegalSearch()


@st.cache_resource
def load_graph_db():
    """Load graph database (cached)"""
    return LegalGraphDB()


def build_legal_prompt(question: str, context: str) -> str:
    """Build prompt for legal Q&A"""
    return f"""Bạn là trợ lý pháp luật Việt Nam chuyên nghiệp. 
Nhiệm vụ: Trả lời câu hỏi pháp luật dựa trên các văn bản pháp luật được cung cấp.

Quy tắc:
1. Trả lời chính xác, có trích dẫn nguồn (tên văn bản, điều, khoản)
2. Nếu có nhiều quy định liên quan, liệt kê tất cả
3. Giải thích rõ ràng, dễ hiểu cho người dân
4. Nếu không tìm thấy thông tin, nói rõ "Không tìm thấy quy định liên quan"
5. Cảnh báo nếu quy định có thể đã thay đổi

Ngữ cảnh pháp luật:
{context}

Câu hỏi: {question}

Trả lời (có trích dẫn nguồn):"""


def search_and_answer(question: str, strategy: str, top_k: int):
    """Search and generate answer"""
    hybrid = load_search_engine()
    
    # Search
    results = hybrid.search(query=question, strategy=strategy, k=top_k)
    
    if not results:
        return {
            "answer": "Không tìm thấy văn bản pháp luật liên quan. Vui lòng thử với từ khóa khác.",
            "results": [],
            "context": ""
        }
    
    # Build context
    context = hybrid.build_rag_context(results)
    
    # Generate answer
    prompt = build_legal_prompt(question, context)
    answer = generate_answer(prompt)
    
    return {
        "answer": answer,
        "results": results,
        "context": context
    }


def display_search_results(results):
    """Display search results with nice formatting"""
    if not results:
        st.info("Không có kết quả")
        return
    
    for i, r in enumerate(results):
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            
            with col1:
                st.markdown(f"**#{i+1}**")
                st.markdown(f"<span class='score-badge'>Score: {r.score:.2f}</span>", 
                           unsafe_allow_html=True)
            
            with col2:
                # Source info
                source_info = f"📜 {r.law_id}"
                if r.article_title:
                    source_info += f" | {r.article_title}"
                if r.clause_id:
                    source_info += f" | Khoản {r.clause_id}"
                
                st.markdown(f"<span class='source-tag'>{source_info}</span>", 
                           unsafe_allow_html=True)
                
                # Content
                st.markdown(f"<div class='result-card'>{r.content[:500]}...</div>" 
                           if len(r.content) > 500 else f"<div class='result-card'>{r.content}</div>",
                           unsafe_allow_html=True)
            
            with col3:
                if st.button("📋", key=f"copy_{i}", help="Copy nội dung"):
                    st.toast("Đã copy!")
        
        st.divider()


def main():
    # Header
    st.markdown("<h1 class='main-header'>⚖️ Trợ lý Pháp luật Việt Nam</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Hỏi đáp pháp luật thông minh với AI - Powered by Graph + Vector Search</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Search strategy
        strategy = st.selectbox(
            "Chiến lược tìm kiếm",
            options=['graph_enhanced', 'vector_only', 'hierarchical'],
            index=0,
            help="""
            - graph_enhanced: Tìm vector + làm giàu context từ graph
            - vector_only: Chỉ tìm kiếm ngữ nghĩa
            - hierarchical: Tìm trong graph + mở rộng liên quan
            """
        )
        
        top_k = st.slider("Số kết quả", min_value=3, max_value=15, value=5)
        
        st.divider()
        
        # Statistics
        st.header("📊 Thống kê")
        try:
            graph = load_graph_db()
            stats = graph.stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📚 Văn bản", f"{stats['node_types'].get('law', 0):,}")
                st.metric("📄 Điều", f"{stats['node_types'].get('article', 0):,}")
            with col2:
                st.metric("📝 Khoản", f"{stats['node_types'].get('clause', 0):,}")
                st.metric("🔹 Điểm", f"{stats['node_types'].get('point', 0):,}")
            
            st.caption(f"Tổng: {stats['total_nodes']:,} nodes")
        except Exception as e:
            st.error(f"Lỗi load graph: {e}")
        
        st.divider()
        
        # Quick examples
        st.header("💡 Câu hỏi mẫu")
        examples = [
            "Điều kiện kết hôn theo pháp luật?",
            "Thủ tục đăng ký kinh doanh?",
            "Quy định về hợp đồng lao động?",
            "Độ tuổi chịu trách nhiệm hình sự?",
            "Quyền thừa kế theo pháp luật?"
        ]
        
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:10]}"):
                st.session_state['question'] = ex
    
    # Main content
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        # Question input
        question = st.text_area(
            "🔍 Nhập câu hỏi pháp luật",
            value=st.session_state.get('question', ''),
            height=100,
            placeholder="Ví dụ: Điều kiện để được đăng ký kết hôn theo pháp luật Việt Nam là gì?"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        
        with col_btn1:
            search_btn = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("🗑️ Xóa", use_container_width=True):
                st.session_state['question'] = ''
                st.rerun()
        
        # Process search
        if search_btn and question.strip():
            with st.spinner("🔄 Đang tìm kiếm và phân tích..."):
                try:
                    result = search_and_answer(question, strategy, top_k)
                    
                    # Store in session
                    st.session_state.search_history.append({
                        'question': question,
                        'answer': result['answer']
                    })
                    
                    # Display answer
                    st.subheader("💬 Trả lời")
                    st.markdown(result['answer'])
                    
                    # Display sources
                    with st.expander("📚 Nguồn tham khảo", expanded=True):
                        display_search_results(result['results'])
                    
                    # Show raw context
                    with st.expander("📝 Context đầy đủ"):
                        st.text(result['context'])
                        
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    logger.exception("Search error")
        
        elif search_btn:
            st.warning("Vui lòng nhập câu hỏi")
    
    with col_side:
        # Search history
        if st.session_state.search_history:
            st.subheader("📜 Lịch sử")
            for i, h in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.container():
                    st.caption(h['question'][:50] + "...")
                    if st.button("🔄", key=f"hist_{i}"):
                        st.session_state['question'] = h['question']
                        st.rerun()


if __name__ == "__main__":
    main()
