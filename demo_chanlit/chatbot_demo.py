"""
Vietnamese Legal RAG Chatbot - DEMO VERSION v3 (Optimized)

Features:
- 3 search modes: Legal DB | Uploaded Files | Both (merged)
- Upload NHIỀU file → Mỗi file = 1 Qdrant collection
- Legal-aware chunking (Điều → Khoản → Điểm)
- Hybrid search (RRF only) + GPU embedding
- Auto cleanup khi session end
- top_k=10, no reranker (tốc độ tối ưu)
"""

import os
import sys
from pathlib import Path
import asyncio
import time
from typing import Optional, List, Dict
import hashlib
import uuid

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

# Session storage for uploaded file collections
# Format: {session_id: [{"collection_name": str, "file_name": str, "chunks": int}, ...]}
_session_collections = {}


def _get_pipeline_lock():
    """Get or create pipeline lock"""
    global _pipeline_lock
    if _pipeline_lock is None:
        import threading
        _pipeline_lock = threading.Lock()
    return _pipeline_lock


def get_demo_pipeline():
    """Get singleton Legal RAG Pipeline"""
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


def chunk_uploaded_file(text: str, file_name: str) -> List[Dict]:
    """
    Chunk uploaded file using legal-aware chunking
    Returns list of chunks with metadata
    """
    from src.legal_chunker_v2 import LegalChunkerV2
    
    chunker = LegalChunkerV2(
        min_chars=1200,
        max_chars=2800,
        overlap_chars=100
    )
    
    # Create pseudo legal document row
    pseudo_row = {
        'van_ban_id': f'UPLOAD_{hashlib.md5(file_name.encode()).hexdigest()[:8]}',
        'ten_van_ban': f'Tài liệu: {file_name}',
        'loai_van_ban': 'Tài liệu upload',
        'co_quan': 'User Upload',
        'chuong': '',
        'ten_chuong': '',
        'dieu_so': '1',
        'tieu_de_dieu': file_name,
        'clean_text': text,
        'noi_dung': text
    }
    
    # Chunk using legal chunker
    chunks = chunker.chunk_article(pseudo_row, 0)
    
    return [chunk.to_dict() for chunk in chunks]


async def upload_to_qdrant_collection(
    chunks: List[Dict],
    collection_name: str,
    pipeline
) -> bool:
    """
    Upload chunks to Qdrant temporary collection
    Returns True if successful
    """
    try:
        from qdrant_client.http import models
        
        loop = asyncio.get_event_loop()
        qdrant = pipeline.qdrant_store.client
        
        # Create collection
        def create_collection():
            try:
                qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=768,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams()
                        )
                    }
                )
                return True
            except Exception as e:
                print(f"[UPLOAD] Error creating collection: {e}")
                return False
        
        success = await loop.run_in_executor(None, create_collection)
        if not success:
            return False
        
        # Prepare points for upload
        def prepare_and_upload():
            points = []
            
            # Batch embed all chunks
            contents = [c["content"] for c in chunks]
            embeddings = pipeline.embedding_model.encode(contents)
            
            # Prepare sparse vectors (BM25)
            bm25_encoder = pipeline.qdrant_store.bm25_encoder
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Dense vector
                dense_vector = embedding
                
                # Sparse vector
                indices, values = bm25_encoder.encode(chunk["content"])
                sparse_vector = models.SparseVector(
                    indices=indices,
                    values=values
                )
                
                # Create point
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector
                    },
                    payload=chunk
                )
                points.append(point)
            
            # Upload in batches
            batch_size = 100
            for j in range(0, len(points), batch_size):
                batch = points[j:j+batch_size]
                qdrant.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            return True
        
        await loop.run_in_executor(None, prepare_and_upload)
        return True
        
    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def search_in_collection(
    query: str,
    collection_name: str,
    pipeline,
    top_k: int = 10
) -> List[Dict]:
    """
    Search in specific Qdrant collection
    Returns list of search results
    """
    try:
        loop = asyncio.get_event_loop()
        
        def do_search():
            # Encode query
            query_emb = pipeline.embedding_model.encode_query(query)
            
            # Search using hybrid
            results = pipeline.qdrant_store.hybrid_search(
                query=query,
                query_embedding=query_emb,
                top_k=top_k,
                dense_weight=0.6,
                sparse_weight=0.4,
                collection_name=collection_name  # Custom collection
            )
            
            return [
                {
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        
        return await loop.run_in_executor(None, do_search)
        
    except Exception as e:
        print(f"[SEARCH] Error: {e}")
        return []


def merge_results_rrf(
    results1: List[Dict],
    results2: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    Merge two result lists using RRF (Reciprocal Rank Fusion)
    """
    doc_scores = {}
    
    # Process results1
    for rank, result in enumerate(results1, 1):
        doc_id = result["content"][:50]  # Use first 50 chars as ID
        score = 1 / (k + rank)
        doc_scores[doc_id] = {
            "rrf_score": doc_scores.get(doc_id, {}).get("rrf_score", 0) + score,
            "result": result
        }
    
    # Process results2
    for rank, result in enumerate(results2, 1):
        doc_id = result["content"][:50]
        score = 1 / (k + rank)
        if doc_id in doc_scores:
            doc_scores[doc_id]["rrf_score"] += score
        else:
            doc_scores[doc_id] = {
                "rrf_score": score,
                "result": result
            }
    
    # Sort by RRF score
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    return [item["result"] for item in sorted_docs]


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session"""
    msg = cl.Message(content="🔄 Đang khởi tạo hệ thống...")
    await msg.send()
    
    # Initialize session
    session_id = cl.user_session.get("id")
    if session_id not in _session_collections:
        _session_collections[session_id] = []  # List of uploaded files
    
    # Set default search mode
    cl.user_session.set("search_mode", "legal_only")  # Default: legal DB only
    
    try:
        # Initialize pipeline
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_demo_pipeline)
        
        # Get stats
        pipeline = get_demo_pipeline()
        stats = pipeline.get_stats()
        qdrant_stats = stats.get("qdrant", {})
        doc_count = qdrant_stats.get("points_count", 0)
        
        msg.content = f"""# 🏛️ Hệ Thống Tư Vấn Pháp Luật Việt Nam


---

## 💡 Hướng Dẫn Sử Dụng

### 🔍 Tìm kiếm pháp luật
Hỏi trực tiếp về bất kỳ vấn đề pháp lý:
- _"Quy định về tội giết người trong BLHS 2015?"_
- _"Điều kiện và thủ tục thành lập doanh nghiệp?"_
- _"Mức phạt vi phạm giao thông không đội mũ bảo hiểm?"_

**Mode mặc định:** 📚 Legal Database

### 📄 Upload tài liệu
- Hỗ trợ **nhiều file** PDF/TXT cùng lúc
- Mỗi file được phân tích theo cấu trúc pháp luật (Điều → Khoản → Điểm)
- Tự động chuyển sang **Mode: Uploaded Files**

### ⚙️ Chuyển đổi chế độ
- `/legal` → Chỉ tìm trong cơ sở dữ liệu pháp luật
- `/file` → Chỉ tìm trong các file đã upload
- `/both` → Tìm kiếm kết hợp cả hai nguồn (RRF merge)

---
"""
        
        await msg.update()
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[INIT ERROR] {error_trace}")
        msg.content = f"⚠️ Đang khởi tạo hệ thống, vui lòng đợi trong giây lát...\n\n_Nếu lỗi tiếp tục, vui lòng reload trang._"
        await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message with multi-source RAG"""
    query = message.content.strip()
    session_id = cl.user_session.get("id")
    
    # Handle mode switch commands
    if query.startswith("/"):
        if query == "/legal":
            cl.user_session.set("search_mode", "legal_only")
            await cl.Message(content="✅ **Mode:** 📚 Legal Database Only").send()
            return
        elif query == "/file":
            cl.user_session.set("search_mode", "file_only")
            await cl.Message(content="✅ **Mode:** 📄 Uploaded File Only").send()
            return
        elif query == "/both":
            cl.user_session.set("search_mode", "both")
            await cl.Message(content="✅ **Mode:** 🔀 Both (Legal + File)").send()
            return
    
    # Handle file upload
    if message.elements:
        for element in message.elements:
            if hasattr(element, 'path') and element.path:
                mime = getattr(element, 'mime', 'text/plain')
                file_name = element.name
                
                process_msg = cl.Message(content=f"📄 **Đang xử lý: {file_name}**")
                await process_msg.send()
                
                # Extract text
                await process_msg.stream_token("\n🔄 Đang đọc file...")
                file_text = extract_text_from_file(element.path, mime)
                
                if file_text.startswith("❌"):
                    await process_msg.stream_token(f"\n\n{file_text}")
                    return
                
                await process_msg.stream_token(f"\n✅ Đọc: {len(file_text):,} ký tự")
                
                # Chunk using legal chunker
                await process_msg.stream_token("\n🔄 Legal chunking (Điều→Khoản→Điểm)...")
                loop = asyncio.get_event_loop()
                
                chunks = await loop.run_in_executor(
                    None, chunk_uploaded_file, file_text, file_name
                )
                
                await process_msg.stream_token(f"\n✅ Tạo: {len(chunks)} chunks")
                
                # Create collection name
                collection_name = f"upload_{session_id[:8]}_{int(time.time())}"
                
                # Upload to Qdrant
                await process_msg.stream_token(f"\n🔄 Đang embed & upload vào Qdrant...")
                pipeline = get_demo_pipeline()
                
                success = await upload_to_qdrant_collection(
                    chunks, collection_name, pipeline
                )
                
                if success:
                    # Append to session files list
                    _session_collections[session_id].append({
                        "collection_name": collection_name,
                        "file_name": file_name,
                        "chunks": len(chunks)
                    })
                    
                    # Auto switch to file mode
                    cl.user_session.set("search_mode", "file_only")
                    
                    total_files = len(_session_collections[session_id])
                    await process_msg.stream_token(f"\n\n🎯 **File đã được index!**\n- Collection: `{collection_name}`\n- {len(chunks)} chunks\n- {len(file_text):,} ký tự\n\n✅ **Mode tự động:** 📄 Uploaded Files\n📚 **Tổng:** {total_files} file(s) đã upload\n💬 Giờ bạn có thể hỏi về các file này!\n\n_Gõ `/legal` để search legal DB, `/both` để search cả 2_")
                else:
                    await process_msg.stream_token("\n\n❌ Lỗi upload file, vui lòng thử lại")
                
                return
    
    if not query:
        return
    
    # Get search mode
    search_mode = cl.user_session.get("search_mode", "legal_only")
    session_data = _session_collections.get(session_id, [])  # List of files
    has_uploaded_files = len(session_data) > 0
    
    # Validate mode
    if search_mode in ["file_only", "both"] and not has_uploaded_files:
        await cl.Message(content="⚠️ Chưa có file upload. Đang search legal DB...").send()
        search_mode = "legal_only"
        cl.user_session.set("search_mode", "legal_only")
    
    # Create response message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        start_time = time.time()
        pipeline = get_demo_pipeline()
        loop = asyncio.get_event_loop()
        
        # Show mode indicator
        mode_emoji = {
            "legal_only": "📚",
            "file_only": "📄",
            "both": "🔀"
        }.get(search_mode, "🔍")
        
        await msg.stream_token(f"{mode_emoji} Đang tìm kiếm...\n\n")
        
        # Search based on mode
        if search_mode == "legal_only":
            # Search only legal DB
            def search_legal():
                return pipeline.query(
                    question=query,
                    top_k=10,
                    use_reranker=False
                )
            
            result = await loop.run_in_executor(None, search_legal)
            
            # Clear and display
            msg.content = ""
            await msg.stream_token(f"📚 **{len(result.citations)} văn bản pháp luật** (⏱️ {result.retrieval_time:.1f}s)\n\n---\n\n")
            
            # Stream answer
            answer = result.answer
            for i in range(0, len(answer), 20):
                await msg.stream_token(answer[i:i+20])
                await asyncio.sleep(0.005)
            
            # Citations
            if result.citations:
                await msg.stream_token("\n\n---\n\n📖 **Nguồn:**\n")
                for cite in result.citations[:5]:
                    ten_vb = cite.get("ten_van_ban", "N/A")
                    dieu = cite.get("dieu_so", "")
                    citation = f"- Điều {dieu}" if dieu else "- Văn bản"
                    citation += f" - _{ten_vb[:50]}..._\n" if len(ten_vb) > 50 else f" - _{ten_vb}_\n"
                    await msg.stream_token(citation)
            
            total_time = time.time() - start_time
            await msg.stream_token(f"\n⚡ _{total_time:.1f}s (Search: {result.retrieval_time:.1f}s | Gen: {result.generation_time:.1f}s)_")
        
        elif search_mode == "file_only":
            # Search in ALL uploaded files and merge
            all_results = []
            
            for file_info in session_data:
                collection_name = file_info["collection_name"]
                file_results = await search_in_collection(
                    query, collection_name, pipeline, top_k=10
                )
                # Tag results with file name
                for r in file_results:
                    r["source_file"] = file_info["file_name"]
                all_results.extend(file_results)
            
            # Sort by score and take top 10
            all_results.sort(key=lambda x: x["score"], reverse=True)
            search_results = all_results[:10]
            
            search_time = time.time() - start_time
            
            total_files = len(session_data)
            total_chunks = sum(f["chunks"] for f in session_data)
            
            msg.content = ""
            await msg.stream_token(f"📄 **{total_files} file(s)** - {len(search_results)} chunks từ {total_chunks} (⏱️ {search_time:.1f}s)\n\n---\n\n")
            
            # Build context
            context = "\n\n".join([r["content"] for r in search_results])
            
            # Generate answer
            def gen():
                return pipeline.llm.generate(query, context)
            
            answer = await loop.run_in_executor(None, gen)
            gen_time = time.time() - start_time - search_time
            
            # Stream answer
            for i in range(0, len(answer), 20):
                await msg.stream_token(answer[i:i+20])
                await asyncio.sleep(0.005)
            
            # Show sources with file names
            await msg.stream_token(f"\n\n---\n\n📎 **Nguồn từ {len(set(r['source_file'] for r in search_results))} file(s):**\n")
            seen_files = set()
            for r in search_results[:5]:
                if r["source_file"] not in seen_files:
                    await msg.stream_token(f"- {r['source_file']}\n")
                    seen_files.add(r["source_file"])
            
            await msg.stream_token(f"\n⚡ _{time.time()-start_time:.1f}s (Search: {search_time:.1f}s | Gen: {gen_time:.1f}s)_")
        
        elif search_mode == "both":
            # Search legal DB + ALL uploaded files, merge with RRF
            
            # Search legal DB
            legal_results_task = loop.run_in_executor(
                None, 
                lambda: pipeline.query(query, top_k=20, use_reranker=False)
            )
            
            # Search all uploaded files in parallel
            file_tasks = []
            for file_info in session_data:
                collection_name = file_info["collection_name"]
                task = search_in_collection(query, collection_name, pipeline, top_k=20)
                file_tasks.append((file_info["file_name"], task))
            
            # Wait for legal results
            result_legal = await legal_results_task
            
            # Wait for all file results
            all_file_results = []
            for file_name, task in file_tasks:
                results = await task
                for r in results:
                    r["source_file"] = file_name
                all_file_results.extend(results)
            
            search_time = time.time() - start_time
            
            # Merge all results with RRF
            legal_dicts = [
                {"content": c.get("content", ""), "score": 0.9, "metadata": c, "source_file": "Legal DB"}
                for c in result_legal.citations
            ]
            
            # Merge legal + all files
            merged = merge_results_rrf(legal_dicts, all_file_results, k=60)[:20]
            
            total_files = len(session_data)
            msg.content = ""
            await msg.stream_token(f"🔀 **Kết hợp:** 📚 Legal + 📄 {total_files} file(s) ({len(merged)} results, ⏱️ {search_time:.1f}s)\n\n---\n\n")
            
            # Build context from merged
            context = "\n\n".join([r["content"] for r in merged])
            
            # Generate
            def gen():
                return pipeline.llm.generate(query, context)
            
            answer = await loop.run_in_executor(None, gen)
            gen_time = time.time() - start_time - search_time
            
            # Stream
            for i in range(0, len(answer), 20):
                await msg.stream_token(answer[i:i+20])
                await asyncio.sleep(0.005)
            
            # Show source breakdown
            await msg.stream_token(f"\n\n---\n\n🔀 **Nguồn (RRF merge):**\n")
            source_counts = {}
            for r in merged:
                src = r.get("source_file", "Unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            for src, count in source_counts.items():
                await msg.stream_token(f"- {src}: {count} chunk(s)\n")
            
            await msg.stream_token(f"\n⚡ _{time.time()-start_time:.1f}s (Search: {search_time:.1f}s | Gen: {gen_time:.1f}s)_")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] {error_trace}")
        await msg.stream_token(f"\n\n❌ **Lỗi:** {str(e)}")


@cl.on_chat_end
async def on_chat_end():
    """Cleanup when session ends - delete ALL uploaded collections"""
    session_id = cl.user_session.get("id")
    
    if session_id in _session_collections:
        session_files = _session_collections[session_id]  # List of files
        
        if session_files:
            pipeline = get_demo_pipeline()
            for file_info in session_files:
                collection_name = file_info.get("collection_name")
                if collection_name:
                    try:
                        pipeline.qdrant_store.client.delete_collection(collection_name)
                        print(f"[CLEANUP] Deleted collection: {collection_name}")
                    except Exception as e:
                        print(f"[CLEANUP] Error deleting {collection_name}: {e}")
        
        # Remove from dict
        del _session_collections[session_id]


if __name__ == "__main__":
    print("🚀 DEMO v3 - Multi-File RAG with Reranker!")
    print("Run: chainlit run chatbot_demo.py -w")
