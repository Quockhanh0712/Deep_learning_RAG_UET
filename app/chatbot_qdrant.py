"""
Vietnamese Legal RAG Chatbot - Qdrant Version with Streaming

Chatbot hỏi đáp pháp luật Việt Nam với:
- Qdrant Vector DB: Hybrid search (dense + sparse)
- Streaming response (hiển thị từng phần)
- PDF/TXT file upload
- Intent-aware retrieval
- Legal citation formatting
- Qwen2.5:3b via Ollama
"""

import os
import sys
from pathlib import Path
import asyncio
import time
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global pipeline with thread safety
_pipeline = None
_pipeline_lock = None
_pipeline_initialized = False


def _get_pipeline_lock():
    """Get or create pipeline lock"""
    global _pipeline_lock
    if _pipeline_lock is None:
        import threading
        _pipeline_lock = threading.Lock()
    return _pipeline_lock


def get_pipeline():
    """Get singleton Legal RAG Pipeline with thread safety"""
    global _pipeline, _pipeline_initialized
    
    # Fast path: already initialized
    if _pipeline_initialized and _pipeline is not None:
        return _pipeline
    
    lock = _get_pipeline_lock()
    with lock:
        # Double-check after acquiring lock
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
        # Fallback to PyPDF2
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
            return "Lỗi: Cần cài đặt pdfplumber hoặc PyPDF2 để đọc PDF. Chạy: pip install pdfplumber"
    except Exception as e:
        return f"Lỗi đọc PDF: {str(e)}"


def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """Extract text from uploaded file"""
    if mime_type == "application/pdf":
        return extract_text_from_pdf(file_path)
    elif mime_type in ["text/plain", "text/markdown"]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        return f"Không hỗ trợ định dạng file: {mime_type}"


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session"""
    # Show loading message
    msg = cl.Message(content="🔄 Đang khởi tạo hệ thống tư vấn pháp luật...")
    await msg.send()
    
    try:
        # Initialize pipeline in background
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_pipeline)
        
        # Get stats
        pipeline = get_pipeline()
        stats = pipeline.get_stats()
        
        # Update message
        qdrant_stats = stats.get("qdrant", {})
        doc_count = qdrant_stats.get("points_count", 0)
        
        msg.content = f"""✅ **Hệ thống Tư vấn Pháp luật Việt Nam**

📊 **Thống kê:**
- Văn bản được index: **{doc_count:,}** chunks
- Embedding: `{stats.get('embedding_model', 'N/A')}`
- LLM: `{stats.get('llm_model', 'N/A')}`

💡 **Hướng dẫn:**
- Hỏi quy định: "Điều 128 BLHS quy định gì?"
- Hỏi thủ tục: "Điều kiện đăng ký kinh doanh?"
- **Upload PDF/TXT** để hỏi về nội dung file

📚 Câu trả lời sẽ trích dẫn điều luật cụ thể."""
        
        await msg.update()
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[INIT ERROR] {error_trace}")
        
        msg.content = f"""⚠️ **Đang khởi tạo hệ thống...**

Hệ thống đang tải model, vui lòng thử gửi câu hỏi.

_Nếu lỗi tiếp tục, hãy reload trang._"""
        await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message with streaming response"""
    query = message.content.strip()
    
    # Handle file uploads
    uploaded_text = ""
    if message.elements:
        for element in message.elements:
            if hasattr(element, 'path') and element.path:
                mime = getattr(element, 'mime', 'text/plain')
                file_text = extract_text_from_file(element.path, mime)
                if file_text and not file_text.startswith("Lỗi"):
                    uploaded_text += f"\n\n📄 **Nội dung file {element.name}:**\n{file_text[:3000]}..."
                    query = f"Dựa trên tài liệu được upload, hãy trả lời: {query}\n\nNội dung tài liệu:\n{file_text[:5000]}"
                else:
                    await cl.Message(content=file_text).send()
                    return
    
    if not query:
        return
    
    # Create streaming message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        pipeline = get_pipeline()
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        # Step 1: Show progress
        await msg.stream_token("🔍 Đang tìm kiếm văn bản pháp luật...\n\n")
        
        # Run full query in executor (get result object)
        result = await loop.run_in_executor(None, pipeline.query, query)
        
        # Clear message
        msg.content = ""
        
        # Show retrieval stats
        await msg.stream_token(f"📚 Tìm thấy **{len(result.citations)}** văn bản liên quan (⏱️ {result.retrieval_time:.2f}s)\n\n---\n\n")
        
        # Stream the answer (already generated, just display character by character for UX)
        answer_text = result.answer
        
        # Stream answer in chunks for better UX
        chunk_size = 10  # characters per chunk
        for i in range(0, len(answer_text), chunk_size):
            chunk = answer_text[i:i+chunk_size]
            await msg.stream_token(chunk)
            await asyncio.sleep(0.01)  # Small delay for smooth streaming effect
        
        # Add citations
        await msg.stream_token("\n\n---\n\n📖 **Nguồn tham khảo:**\n")
        
        seen_sources = set()
        for i, cite in enumerate(result.citations[:5], 1):
            source_key = (cite.get("ten_van_ban", ""), cite.get("dieu_so", ""))
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                ten_vb = cite.get("ten_van_ban", "N/A")
                dieu = cite.get("dieu_so", "")
                khoan = cite.get("khoan_so", "")
                
                citation = f"- **Điều {dieu}**" if dieu else f"- Văn bản"
                if khoan and khoan != "0":
                    citation += f", Khoản {khoan}"
                citation += f" - _{ten_vb[:60]}..._\n" if len(ten_vb) > 60 else f" - _{ten_vb}_\n"
                
                await msg.stream_token(citation)
        
        # Timing info with reranker details
        total_time = time.time() - start_time
        timing_parts = [f"Tổng: {total_time:.1f}s"]
        
        if result.used_reranker:
            search_only = result.retrieval_time - result.rerank_time
            timing_parts.append(f"Search: {search_only:.2f}s")
            timing_parts.append(f"Rerank: {result.rerank_time:.2f}s")
        else:
            timing_parts.append(f"Tìm kiếm: {result.retrieval_time:.2f}s")
        
        timing_parts.append(f"Sinh: {result.generation_time:.1f}s")
        timing_parts.append(f"Intent: {result.intent}")
        
        if result.used_reranker:
            timing_parts.append("✓Reranker")
        
        await msg.stream_token(f"\n⏱️ _{ ' | '.join(timing_parts)}_")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] {error_trace}")
        
        await msg.stream_token(f"\n\n❌ **Lỗi:** {str(e)}\n\n_Vui lòng thử lại._")


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings update"""
    pass


if __name__ == "__main__":
    print("Run with: chainlit run chatbot_qdrant.py -w")
