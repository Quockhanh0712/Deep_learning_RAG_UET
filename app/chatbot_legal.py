"""
Vietnamese Legal RAG Chatbot - Chainlit Interface

Chatbot hỏi đáp pháp luật Việt Nam với:
- Graph Database: Lưu trữ cấu trúc phân cấp
- Hybrid Search: Vector + Graph context enrichment
- Vietnamese Legal Embedding (768D)
- Local LLM (Qwen3-4B hoặc Ollama)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import asyncio

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Configure
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = int(os.getenv("TOP_K", "15"))  # More context for better answers

# Global caching for models - PRELOAD to avoid reload
_hybrid_search = None
_graph_db = None
_llm_preloaded = False


def get_hybrid_search():
    """Lazy load HybridLegalSearch"""
    global _hybrid_search
    if _hybrid_search is None:
        from src.hybrid_legal_search import HybridLegalSearch
        _hybrid_search = HybridLegalSearch()
    return _hybrid_search


def get_graph_db():
    """Lazy load GraphDB"""
    global _graph_db
    if _graph_db is None:
        from src.graph_db import LegalGraphDB
        _graph_db = LegalGraphDB()
    return _graph_db


def preload_models():
    """Preload HuggingFace model ONLY if using transformers provider"""
    global _llm_preloaded
    if not _llm_preloaded:
        provider = os.getenv("LLM_PROVIDER", "ollama")
        if provider == "transformers":
            try:
                from src.llm_client import _load_hf_model
                print("[PRELOAD] Loading HuggingFace model at startup...")
                _load_hf_model()
                _llm_preloaded = True
                print("[PRELOAD] Model loaded successfully!")
            except Exception as e:
                print(f"[PRELOAD] Warning: Could not preload model: {e}")
        else:
            print(f"[PRELOAD] Using provider: {provider} - No preload needed")
            _llm_preloaded = True


def build_legal_prompt(question: str, context: str) -> str:
    """Build prompt for legal Q&A - Optimized for Ollama"""
    # Detailed legal prompt with Vietnamese focus
    return f"""Bạn là chuyên gia pháp luật Việt Nam với nhiều năm kinh nghiệm. LUÔN trả lời BẰNG TIẾNG VIỆT.

## VĂN BẢN PHÁP LUẬT THAM KHẢO:
{context}

## CÂU HỎI:
{question}

## YÊU CẦU TRẢ LỜI:
1. Trả lời CHI TIẾT, CỤ THỂ dựa trên văn bản pháp luật trên
2. Trích dẫn RÕ RÀNG: tên luật, số/năm, điều, khoản, điểm
3. Giải thích THỰC TẾ, dễ hiểu cho người dân
4. BẰNG TIẾNG VIỆT, không dùng tiếng Anh

## TRẢ LỜI:

VĂN BẢN PHÁP LUẬT:
{context}

CÂU HỎI: {question}

TRẢ LỜI (BẰNG TIẾNG VIỆT):

TRẢ LỜI:"""


def sync_legal_query(question: str, strategy: str = 'graph_enhanced', top_k: int = 5) -> Dict:
    """Process legal query with hybrid search"""
    hybrid = get_hybrid_search()
    
    # Search
    results = hybrid.search(query=question, strategy=strategy, k=top_k)
    
    if not results:
        return {
            "answer": "❌ Không tìm thấy văn bản pháp luật liên quan. Vui lòng thử với từ khóa khác hoặc diễn đạt câu hỏi rõ ràng hơn.",
            "sources": [],
            "context": ""
        }
    
    # Build context
    context = hybrid.build_rag_context(results)
    
    # Generate answer
    from src.llm_client import generate_answer
    prompt = build_legal_prompt(question, context)
    answer = generate_answer(prompt)
    
    # Build sources list - show more content
    sources = []
    for r in results:
        sources.append({
            "law_id": r.law_id,
            "article_title": r.article_title,
            "clause_id": r.clause_id,
            "point_id": r.point_id,
            "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
            "score": r.score
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "context": context
    }


def format_sources_markdown(sources: List[Dict]) -> str:
    """Format sources as markdown - SIMPLIFIED"""
    if not sources:
        return ""
    
    md = "\n\n---\n## 📚 Nguồn tham khảo\n\n"
    
    for i, s in enumerate(sources, 1):
        # Simple header
        header = f"**{s['law_id']}**"
        if s.get('article_title'):
            header += f" - {s['article_title']}"
        
        md += f"{i}. {header} (Score: {s['score']:.2f})\n"
        
        # Show content in collapsed format
        content_preview = s['content'][:150] + "..." if len(s['content']) > 150 else s['content']
        md += f"   > {content_preview}\n\n"
    
    return md


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Preload models ONCE at startup
    preload_models()
    
    # Get statistics
    try:
        graph = get_graph_db()
        stats = graph.stats()
        
        stats_str = f"""
| Loại | Số lượng |
|------|----------|
| 📚 Văn bản | {stats['node_types'].get('law', 0):,} |
| 📄 Điều | {stats['node_types'].get('article', 0):,} |
| 📝 Khoản | {stats['node_types'].get('clause', 0):,} |
| 🔹 Điểm | {stats['node_types'].get('point', 0):,} |
| **Tổng** | **{stats['total_nodes']:,}** |
"""
    except Exception as e:
        stats_str = f"_Không thể load thống kê: {e}_"
    
    # Welcome message
    welcome_msg = f"""# ⚖️ Trợ lý Pháp luật Việt Nam

Xin chào! Tôi là trợ lý AI chuyên về pháp luật Việt Nam.

## 📊 Cơ sở dữ liệu pháp luật
{stats_str}

## 🚀 Tính năng
- 🔍 **Hybrid Search**: Kết hợp tìm kiếm vector + graph
- 📚 **Graph Database**: Cấu trúc phân cấp Luật → Điều → Khoản → Điểm
- 🇻🇳 **Vietnamese NLP**: Tokenization tiếng Việt chuyên biệt
- 📖 **Trích nguồn**: Luôn kèm nguồn tham khảo

## 💡 Câu hỏi mẫu
- Điều kiện kết hôn theo pháp luật Việt Nam?
- Thủ tục đăng ký doanh nghiệp?
- Quy định về hợp đồng lao động?
- Độ tuổi chịu trách nhiệm hình sự?

## 📱 Commands
- `/stats` - Xem thống kê cơ sở dữ liệu
- `/search <từ khóa>` - Tìm kiếm nhanh
- `/law <mã luật>` - Xem chi tiết văn bản
- `/help` - Trợ giúp

---
**Hãy đặt câu hỏi pháp luật của bạn! ⚖️**
"""
    
    await cl.Message(content=welcome_msg).send()
    
    # Store settings in session
    cl.user_session.set("search_strategy", "graph_enhanced")
    cl.user_session.set("top_k", TOP_K)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    user_input = message.content.strip()
    
    # Handle commands
    if user_input.lower() == "/help":
        help_msg = """## ❓ Trợ giúp

### Commands
| Command | Mô tả |
|---------|-------|
| `/stats` | Xem thống kê cơ sở dữ liệu |
| `/search <từ khóa>` | Tìm kiếm từ khóa trong văn bản |
| `/law <mã luật>` | Xem chi tiết một văn bản pháp luật |
| `/mode` | Xem/đổi chế độ tìm kiếm |
| `/help` | Hiển thị trợ giúp |

### Cách hỏi hiệu quả
1. **Hỏi cụ thể**: "Điều kiện kết hôn" thay vì "cho hỏi về kết hôn"
2. **Dùng từ khóa pháp lý**: "quy định", "thủ tục", "điều kiện", "xử phạt"
3. **Nêu rõ lĩnh vực**: "hợp đồng lao động", "hôn nhân gia đình", "doanh nghiệp"

### Chiến lược tìm kiếm
- `graph_enhanced` (mặc định): Kết hợp vector + graph context
- `vector_only`: Chỉ tìm kiếm ngữ nghĩa
- `hierarchical`: Tìm trong graph + mở rộng

Đổi bằng lệnh `/mode <strategy>`
"""
        await cl.Message(content=help_msg).send()
        return
    
    if user_input.lower() == "/stats":
        try:
            graph = get_graph_db()
            stats = graph.stats()
            
            msg = f"""## 📊 Thống kê Cơ sở dữ liệu

| Loại | Số lượng |
|------|----------|
| 📚 Văn bản luật | {stats['node_types'].get('law', 0):,} |
| 📄 Điều | {stats['node_types'].get('article', 0):,} |
| 📝 Khoản | {stats['node_types'].get('clause', 0):,} |
| 🔹 Điểm | {stats['node_types'].get('point', 0):,} |

**Tổng nodes:** {stats['total_nodes']:,}
**Tổng edges:** {stats['total_edges']:,}

---
_Graph Database: NetworkX với pickle persistence_
"""
            await cl.Message(content=msg).send()
        except Exception as e:
            await cl.Message(content=f"❌ Lỗi: {e}").send()
        return
    
    if user_input.lower().startswith("/search "):
        keyword = user_input[8:].strip()
        if not keyword:
            await cl.Message(content="⚠️ Vui lòng nhập từ khóa: `/search <từ khóa>`").send()
            return
        
        try:
            graph = get_graph_db()
            results = graph.search_by_content(keyword, limit=10)
            
            if results:
                msg = f"## 🔍 Kết quả tìm kiếm: \"{keyword}\"\n\n"
                for i, (node_id, data) in enumerate(results[:5], 1):
                    content = data.get('content', '')[:150]
                    msg += f"### {i}. {node_id[:50]}\n"
                    msg += f"> {content}...\n\n"
                
                msg += f"\n_Tìm thấy {len(results)} kết quả_"
                await cl.Message(content=msg).send()
            else:
                await cl.Message(content=f"📭 Không tìm thấy kết quả cho \"{keyword}\"").send()
        except Exception as e:
            await cl.Message(content=f"❌ Lỗi tìm kiếm: {e}").send()
        return
    
    if user_input.lower().startswith("/law "):
        law_id = user_input[5:].strip()
        if not law_id:
            await cl.Message(content="⚠️ Vui lòng nhập mã văn bản: `/law <mã>`").send()
            return
        
        try:
            graph = get_graph_db()
            results = graph.search_by_law(law_id)
            
            if results:
                msg = f"## 📜 Văn bản: {law_id}\n\n"
                msg += f"**Số điều khoản:** {len(results)}\n\n"
                
                # Group by article
                articles = {}
                for node_id, data in results[:20]:
                    art_id = data.get('article_id', 0)
                    if art_id not in articles:
                        articles[art_id] = data.get('article_title', f'Điều {art_id}')
                
                for art_id, title in sorted(articles.items()):
                    msg += f"- **{title}**\n"
                
                if len(results) > 20:
                    msg += f"\n_...và {len(results) - 20} mục khác_"
                
                await cl.Message(content=msg).send()
            else:
                await cl.Message(content=f"📭 Không tìm thấy văn bản: {law_id}").send()
        except Exception as e:
            await cl.Message(content=f"❌ Lỗi: {e}").send()
        return
    
    if user_input.lower() == "/mode":
        strategy = cl.user_session.get("search_strategy", "graph_enhanced")
        msg = f"""## ⚙️ Chế độ tìm kiếm

**Hiện tại:** `{strategy}`

**Các chế độ:**
- `graph_enhanced` - Vector + Graph context (khuyến nghị)
- `vector_only` - Chỉ tìm kiếm vector
- `hierarchical` - Graph traversal + mở rộng

**Đổi chế độ:** `/mode <strategy>`
"""
        await cl.Message(content=msg).send()
        return
    
    if user_input.lower().startswith("/mode "):
        new_mode = user_input[6:].strip().lower()
        valid_modes = ['graph_enhanced', 'vector_only', 'hierarchical']
        if new_mode in valid_modes:
            cl.user_session.set("search_strategy", new_mode)
            await cl.Message(content=f"✅ Đã đổi chế độ tìm kiếm sang: `{new_mode}`").send()
        else:
            await cl.Message(content=f"⚠️ Chế độ không hợp lệ. Chọn: {', '.join(valid_modes)}").send()
        return
    
    # Skip empty messages
    if not user_input:
        return
    
    # Process legal question
    await process_legal_question(user_input)


async def process_legal_question(question: str):
    """Process a legal question and send response"""
    strategy = cl.user_session.get("search_strategy", "graph_enhanced")
    top_k = cl.user_session.get("top_k", TOP_K)
    
    # Show thinking message
    thinking_msg = await cl.Message(content="🔄 Đang tìm kiếm và phân tích...").send()
    
    try:
        # Run query in thread pool
        result = await asyncio.to_thread(
            sync_legal_query,
            question,
            strategy,
            top_k
        )
        
        # Build response
        response = result["answer"]
        
        # Add sources
        if result["sources"]:
            response += format_sources_markdown(result["sources"])
        
        # Update thinking message with result
        await thinking_msg.remove()
        
        # Send final response
        await cl.Message(content=response).send()
        
    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(content=f"❌ Lỗi xử lý: {str(e)}").send()
        raise


# Entry point
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
