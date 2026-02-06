"""
Vietnamese Legal RAG Chatbot - DEMO VERSION (Optimized for Speed)

Tối ưu cho demo/presentation:
- Tốc độ cao: top_k=5, không dùng reranker, chỉ dùng RRF
- Upload file → chunk + embed → RAG thật sự
- Có ngữ cảnh dài: Upload 1 lần, hỏi nhiều câu
- Streaming nhanh
"""

import os
import sys
from pathlib import Path
import asyncio
import time
from typing import Optional, List, Dict
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global pipeline
_pipeline = None
_pipeline_lock = None
_pipeline_initialized = False

# Session storage for uploaded file chunks
# Format: {session_id: {"chunks": [...], "metadata": {...}}}
_session_file_storage = {}


def _get_pipeline_lock():
    """Get or create pipeline lock"""
    global _pipeline_lock
    if _pipeline_lock is None:
        import threading
        _pipeline_lock = threading.Lock()
    return _pipeline_lock


def get_demo_pipeline():
    """Get singleton Legal RAG Pipeline (DEMO config)"""
    global _pipeline, _pipeline_initialized
    
    if _pipeline_initialized and _pipeline is not None:
        return _pipeline
    
    lock = _get_pipeline_lock()
    with lock:
        if _pipeline_initialized and _pipeline is not None:
            return _pipeline
        
        from src.legal_rag_pipeline import get_legal_rag_pipeline
        _pipeline = get_legal_rag_pipeline()
        _pipeline.initialize()
        _pipeline_initialized = True
        
    return _pipeline


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            return "❌ Cần cài đặt pdfplumber hoặc PyPDF2. Chạy: pip install pdfplumber"
    except Exception as e:
        return f"❌ Lỗi đọc PDF: {str(e)}"


def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """Extract text from uploaded file"""
    if mime_type == "application/pdf":
        return extract_text_from_pdf(file_path)
    elif mime_type in ["text/plain", "text/markdown"]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        return f"❌ Không hỗ trợ định dạng: {mime_type}"


def chunk_text_simple(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Chunk text into overlapping segments (simple sliding window)
    
    Args:
        text: Full text
        chunk_size: Characters per chunk
        overlap: Overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move forward with overlap
        start += (chunk_size - overlap)
    
    return chunks


def semantic_search_in_chunks(
    query: str, 
    chunks: List[str], 
    embedding_model,
    top_k: int = 5
) -> List[Dict]:
    """
    Search relevant chunks using semantic similarity (dense only for uploaded files)
    
    Args:
        query: User query
        chunks: List of text chunks
        embedding_model: Embedding model
        top_k: Number of results
        
    Returns:
        List of {content, score, chunk_id}
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Encode query
    query_embedding = embedding_model.encode_query(query)
    
    # Encode all chunks
    chunk_embeddings = []
    for chunk in chunks:
        emb = embedding_model.encode_document(chunk)
        chunk_embeddings.append(emb)
    
    # Calculate cosine similarity
    query_emb_2d = np.array(query_embedding).reshape(1, -1)
    chunk_embs_2d = np.array(chunk_embeddings)
    
    similarities = cosine_similarity(query_emb_2d, chunk_embs_2d)[0]
    
    # Get top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "content": chunks[idx],
            "score": float(similarities[idx]),
            "chunk_id": f"chunk_{idx}"
        })
    
    return results


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session"""
    msg = cl.Message(content="🔄 Đang khởi tạo hệ thống...")
    await msg.send()
    
    # Initialize session storage
    session_id = cl.user_session.get("id")
    if session_id not in _session_file_storage:
        _session_file_storage[session_id] = {
            "chunks": [],
            "metadata": {}
        }
    
    try:
        # Initialize pipeline
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_demo_pipeline)
        
        # Get stats
        pipeline = get_demo_pipeline()
        stats = pipeline.get_stats()
        qdrant_stats = stats.get("qdrant", {})
        doc_count = qdrant_stats.get("points_count", 0)
        
        msg.content = f"""✅ **Hệ Thống Tư Vấn Pháp Luật Việt Nam** (DEMO Mode)

📊 **Thông tin:**
- Văn bản: **{doc_count:,}** chunks
- Tốc độ: ⚡ **OPTIMIZED** (top_k=5, RRF only)
- Embedding: `{stats.get('embedding_model', 'N/A')}`
- LLM: `{stats.get('llm_model', 'N/A')}`

💡 **Ví dụ câu hỏi:**
- "Tội giết người bị phạt bao nhiêu năm tù?"
- "Điều kiện thành lập doanh nghiệp?"
- "Điều 128 BLHS quy định gì?"

📎 **Upload file PDF/TXT:** 
- Upload 1 lần → Hỏi nhiều câu (có ngữ cảnh dài)
- Hệ thống sẽ chunk + embed + RAG thật sự!

⚡ Tối ưu cho demo: Trả lời nhanh, đầy đủ!"""
        
        await msg.update()
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[INIT ERROR] {error_trace}")
        msg.content = f"⚠️ Đang khởi tạo, vui lòng thử gửi câu hỏi.\n\n_Nếu lỗi tiếp tục, reload trang._"
        await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message with RAG from uploaded file or legal database"""
    query = message.content.strip()
    session_id = cl.user_session.get("id")
    
    # Check for new file upload
    if message.elements:
        for element in message.elements:
            if hasattr(element, 'path') and element.path:
                # New file uploaded - process and store in session
                mime = getattr(element, 'mime', 'text/plain')
                file_name = element.name
                
                # Show processing message
                process_msg = cl.Message(content=f"📄 **Đang xử lý file: {file_name}**\n\n🔄 Đang đọc và chunk nội dung...")
                await process_msg.send()
                
                # Extract text
                file_text = extract_text_from_file(element.path, mime)
                
                if file_text.startswith("❌"):
                    await process_msg.stream_token(f"\n\n{file_text}")
                    return
                
                # Chunk text
                await process_msg.stream_token(f"\n✅ Đọc thành công: {len(file_text):,} ký tự\n🔄 Đang chunk và embed...")
                
                loop = asyncio.get_event_loop()
                
                def chunk_file():
                    return chunk_text_simple(file_text, chunk_size=800, overlap=100)
                
                chunks = await loop.run_in_executor(None, chunk_file)
                
                await process_msg.stream_token(f"\n✅ Tạo {len(chunks)} chunks\n🔄 Đang embed (có thể mất vài giây)...")
                
                # Pre-embed all chunks (để search nhanh sau này)
                pipeline = get_demo_pipeline()
                
                def embed_chunks():
                    embedded = []
                    for i, chunk in enumerate(chunks):
                        # Use encode() method for single chunk
                        emb = pipeline.embedding_model.encode([chunk])[0]
                        embedded.append({
                            "content": chunk,
                            "embedding": emb,
                            "chunk_id": i
                        })
                    return embedded
                
                embedded_chunks = await loop.run_in_executor(None, embed_chunks)
                
                # Store in session
                _session_file_storage[session_id] = {
                    "chunks": embedded_chunks,
                    "metadata": {
                        "file_name": file_name,
                        "total_chars": len(file_text),
                        "num_chunks": len(chunks)
                    }
                }
                
                await process_msg.stream_token(f"\n✅ Hoàn thành!\n\n🎯 **File đã được index:** {file_name}\n- {len(chunks)} chunks\n- {len(file_text):,} ký tự\n\n💬 Bây giờ bạn có thể hỏi bất kỳ câu hỏi nào về file này!")
                
                return
    
    if not query:
        return
    
    # Create response message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        start_time = time.time()
        pipeline = get_demo_pipeline()
        loop = asyncio.get_event_loop()
        
        # Check if session has uploaded file
        session_data = _session_file_storage.get(session_id, {})
        has_file = len(session_data.get("chunks", [])) > 0
        
        if has_file:
            # MODE: RAG from uploaded file
            file_name = session_data["metadata"]["file_name"]
            chunks_data = session_data["chunks"]
            
            await msg.stream_token(f"📄 **Đang tìm kiếm trong: {file_name}**\n\n")
            
            # Semantic search in file chunks
            def search_in_file():
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Encode query
                query_emb = pipeline.embedding_model.encode_query(query)
                query_emb_2d = np.array(query_emb).reshape(1, -1)
                
                # Get all chunk embeddings
                chunk_embeddings = [c["embedding"] for c in chunks_data]
                chunk_embs_2d = np.array(chunk_embeddings)
                
                # Calculate similarity
                similarities = cosine_similarity(query_emb_2d, chunk_embs_2d)[0]
                
                # Get top 5
                top_indices = np.argsort(similarities)[::-1][:5]
                
                results = []
                for idx in top_indices:
                    results.append({
                        "content": chunks_data[idx]["content"],
                        "score": float(similarities[idx]),
                        "chunk_id": idx
                    })
                
                return results
            
            search_results = await loop.run_in_executor(None, search_in_file)
            search_time = time.time() - start_time
            
            await msg.stream_token(f"🔍 Tìm thấy **{len(search_results)} chunks liên quan** (⏱️ {search_time:.1f}s)\n\n---\n\n")
            
            # Build context from top chunks
            context = "\n\n".join([r["content"] for r in search_results])
            
            # Generate answer
            def generate_answer():
                return pipeline.llm.generate(query, context)
            
            answer = await loop.run_in_executor(None, generate_answer)
            gen_time = time.time() - start_time - search_time
            
            # Stream answer
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                await msg.stream_token(chunk)
                await asyncio.sleep(0.005)
            
            # Show sources
            await msg.stream_token(f"\n\n---\n\n📎 **Nguồn:** {file_name} (top 5/{session_data['metadata']['num_chunks']} chunks)")
            await msg.stream_token(f"\n⚡ _Hoàn thành trong {time.time() - start_time:.1f}s (Search: {search_time:.1f}s | Gen: {gen_time:.1f}s)_")
            
        else:
            # MODE: Normal legal database search
            await msg.stream_token("🔍 Đang tìm kiếm trong cơ sở dữ liệu pháp luật...\n\n")
            
            # DEMO CONFIG: top_k=5, no reranker, RRF only
            def run_query():
                return pipeline.query(
                    question=query,
                    top_k=5,  # DEMO: 5 documents
                    use_reranker=False  # DEMO: No reranker, RRF only
                )
            
            result = await loop.run_in_executor(None, run_query)
            
            # Clear progress
            msg.content = ""
            
            # Show stats
            await msg.stream_token(f"📚 **{len(result.citations)} văn bản** (⏱️ {result.retrieval_time:.1f}s)\n\n---\n\n")
            
            # Stream answer
            answer_text = result.answer
            chunk_size = 20
            
            for i in range(0, len(answer_text), chunk_size):
                chunk = answer_text[i:i+chunk_size]
                await msg.stream_token(chunk)
                await asyncio.sleep(0.005)
            
            # Citations
            if result.citations:
                await msg.stream_token("\n\n---\n\n📖 **Nguồn:**\n")
                
                for i, cite in enumerate(result.citations[:5], 1):
                    ten_vb = cite.get("ten_van_ban", "N/A")
                    dieu = cite.get("dieu_so", "")
                    
                    if dieu:
                        citation_text = f"- Điều {dieu}"
                    else:
                        citation_text = f"- Văn bản"
                    
                    if len(ten_vb) > 50:
                        citation_text += f" - _{ten_vb[:50]}..._\n"
                    else:
                        citation_text += f" - _{ten_vb}_\n"
                    
                    await msg.stream_token(citation_text)
            
            # Timing
            total_time = time.time() - start_time
            await msg.stream_token(f"\n⚡ _Hoàn thành trong {total_time:.1f}s (Search: {result.retrieval_time:.1f}s | Gen: {result.generation_time:.1f}s)_")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] {error_trace}")
        
        await msg.stream_token(f"\n\n❌ **Lỗi:** {str(e)}\n\n_Vui lòng thử lại._")


if __name__ == "__main__":
    print("🚀 DEMO Mode - Optimized for Speed!")
    print("Run with: chainlit run chatbot_demo.py -w")
