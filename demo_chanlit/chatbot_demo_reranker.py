"""
Vietnamese Legal RAG Chatbot - RERANKER VERSION (Premium Quality)

Phiên bản cao cấp với Cross-Encoder Reranking:
- 3 search modes: Legal DB | Uploaded Files | Both (merged)
- BGE-Reranker-v2-m3 tích hợp trong pipeline
- Hybrid Search (Dense + BM25 + RRF) + Cross-Encoder Reranking
- So sánh Before/After reranking (toggle option)
- Upload NHIỀU file → Mỗi file = 1 Qdrant collection
- GPU accelerated (Embedding + Reranker + LLM)

Usage:
    chainlit run chatbot_demo_reranker.py -w

Commands:
    /legal  - Search Legal DB only
    /file   - Search uploaded files only  
    /both   - Search both (RRF merge)
    /rerank - Toggle reranker ON/OFF
    /compare - Compare with/without reranker
"""

import os
import sys
from pathlib import Path
import asyncio
import time
from typing import Optional, List, Dict, Tuple
import hashlib
import uuid

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global pipeline and reranker
_pipeline = None
_pipeline_lock = None
_pipeline_initialized = False
_reranker = None
_reranker_loaded = False

# Session storage for uploaded file collections
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


def get_reranker():
    """Get or initialize Cross-Encoder Reranker"""
    global _reranker, _reranker_loaded
    
    if _reranker_loaded and _reranker is not None:
        return _reranker
    
    from src.reranker import LegalReranker
    
    _reranker = LegalReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_gpu=True,
        max_length=512,
        batch_size=16
    )
    _reranker.load()
    _reranker_loaded = True
    
    return _reranker


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
            return "❌ Cần cài đặt pdfplumber hoặc PyPDF2"
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
    """Chunk uploaded file using legal-aware chunking"""
    from src.legal_chunker_v2 import LegalChunkerV2
    
    chunker = LegalChunkerV2(
        min_chars=1200,
        max_chars=2800,
        overlap_chars=100
    )
    
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
    
    chunks = chunker.chunk_article(pseudo_row, 0)
    return [chunk.to_dict() for chunk in chunks]


async def upload_to_qdrant_collection(
    chunks: List[Dict],
    collection_name: str,
    pipeline
) -> bool:
    """Upload chunks to Qdrant temporary collection"""
    try:
        from qdrant_client.http import models
        
        loop = asyncio.get_event_loop()
        qdrant = pipeline.qdrant_store.client
        
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
        
        def prepare_and_upload():
            points = []
            contents = [c["content"] for c in chunks]
            embeddings = pipeline.embedding_model.encode(contents)
            bm25_encoder = pipeline.qdrant_store.bm25_encoder
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                indices, values = bm25_encoder.encode(chunk["content"])
                sparse_vector = models.SparseVector(indices=indices, values=values)
                
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": embedding, "sparse": sparse_vector},
                    payload=chunk
                )
                points.append(point)
            
            batch_size = 100
            for j in range(0, len(points), batch_size):
                batch = points[j:j+batch_size]
                qdrant.upsert(collection_name=collection_name, points=batch)
            
            return True
        
        await loop.run_in_executor(None, prepare_and_upload)
        return True
        
    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        return False


async def search_in_collection(
    query: str,
    collection_name: str,
    pipeline,
    top_k: int = 10
) -> List[Dict]:
    """Search in specific Qdrant collection"""
    try:
        loop = asyncio.get_event_loop()
        
        def do_search():
            query_emb = pipeline.embedding_model.encode_query(query)
            results = pipeline.qdrant_store.hybrid_search(
                query=query,
                query_embedding=query_emb,
                top_k=top_k,
                dense_weight=0.6,
                sparse_weight=0.4,
                collection_name=collection_name
            )
            return [
                {"content": r.content, "score": r.score, "metadata": r.metadata}
                for r in results
            ]
        
        return await loop.run_in_executor(None, do_search)
        
    except Exception as e:
        print(f"[SEARCH] Error: {e}")
        return []


def rerank_results(
    query: str,
    results: List[Dict],
    reranker,
    top_k: int = 5,
    alpha: float = 0.6
) -> Tuple[List[Dict], float]:
    """
    Rerank results using Cross-Encoder
    
    Returns:
        reranked_results: List sorted by rerank score
        rerank_time: Time taken for reranking
    """
    if not results:
        return [], 0.0
    
    start = time.time()
    
    # Extract contents
    contents = [r.get("content", "") for r in results]
    original_scores = [r.get("score", 0.0) for r in results]
    
    # Compute reranker scores
    rerank_scores = reranker.compute_scores(query, contents)
    
    # Combine scores
    reranked = []
    for i, (result, orig_score, rerank_score) in enumerate(zip(results, original_scores, rerank_scores)):
        final_score = alpha * rerank_score + (1 - alpha) * orig_score
        reranked.append({
            **result,
            "original_score": orig_score,
            "rerank_score": rerank_score,
            "final_score": final_score,
            "rank_before": i + 1
        })
    
    # Sort by final score
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Add rank after
    for i, r in enumerate(reranked):
        r["rank_after"] = i + 1
    
    rerank_time = time.time() - start
    return reranked[:top_k], rerank_time


def merge_results_rrf(
    results1: List[Dict],
    results2: List[Dict],
    k: int = 60
) -> List[Dict]:
    """Merge two result lists using RRF (Reciprocal Rank Fusion)"""
    doc_scores = {}
    
    for rank, result in enumerate(results1, 1):
        doc_id = result["content"][:50]
        score = 1 / (k + rank)
        doc_scores[doc_id] = {
            "rrf_score": doc_scores.get(doc_id, {}).get("rrf_score", 0) + score,
            "result": result
        }
    
    for rank, result in enumerate(results2, 1):
        doc_id = result["content"][:50]
        score = 1 / (k + rank)
        if doc_id in doc_scores:
            doc_scores[doc_id]["rrf_score"] += score
        else:
            doc_scores[doc_id] = {"rrf_score": score, "result": result}
    
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return [item["result"] for item in sorted_docs]


def format_top1_reranked(result: Dict) -> str:
    """
    Format TOP 1 reranked document with FULL content
    Chỉ hiển thị 1 điều luật được rerank lên vị trí cao nhất
    """
    if not result:
        return ""
    
    lines = []
    
    rank_before = result.get("rank_before", "?")
    rerank_score = result.get("rerank_score", 0.0)
    
    # Calculate rank change
    if isinstance(rank_before, int) and rank_before > 1:
        delta = f"⬆️ +{rank_before - 1}"
    else:
        delta = "🏆"
    
    # Get metadata
    metadata = result.get("metadata", {})
    ten_van_ban = metadata.get("ten_van_ban", "N/A")
    dieu_so = metadata.get("dieu_so", "")
    tieu_de = metadata.get("tieu_de_dieu", "")
    
    # Header
    lines.append(f"### 🏆 ĐIỀU LUẬT PHÙ HỢP NHẤT (Reranked #{rank_before} → #1 {delta})")
    lines.append(f"")
    
    if dieu_so:
        lines.append(f"**📜 Điều {dieu_so}** - {tieu_de}" if tieu_de else f"**📜 Điều {dieu_so}**")
    
    if ten_van_ban:
        lines.append(f"_📚 {ten_van_ban}_")
    
    lines.append(f"")
    lines.append(f"> **Rerank Score:** {rerank_score:.3f}")
    lines.append(f"")
    
    # Get content - try multiple sources
    content = result.get("content", "")
    if not content:
        content = metadata.get("content", "")
    if not content:
        content = metadata.get("noi_dung", "")
    if not content:
        content = metadata.get("clean_text", "")
    
    content = content.strip() if content else "(Không có nội dung)"
    
    # Clean content - remove metadata noise
    import re
    # Remove metadata prefix like "[TEN_VAN_BAN - Dieu X.X - Tieu de]"
    content = re.sub(r'^\[.*?\]\s*', '', content)
    # Remove repeated "none" noise
    content = re.sub(r'\bnone\s+none\b', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\bnone\b', '', content, flags=re.IGNORECASE)
    # Clean up extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Format nicely with line breaks for readability
    # Split by sentence patterns in Vietnamese legal text
    content = re.sub(r'(\d+[-\.]\s)', r'\n\1', content)  # "1. " or "1- "
    content = re.sub(r'([a-zđ]\)\s)', r'\n\1', content)  # "a) " "b) "
    
    # Format as code block for clean display
    lines.append("**📄 Nội dung điều luật:**")
    lines.append("")
    lines.append("```")
    lines.append(content.strip())
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with reranker"""
    msg = cl.Message(content="🔄 Đang khởi tạo hệ thống Premium (với Reranker)...")
    await msg.send()
    
    session_id = cl.user_session.get("id")
    if session_id not in _session_collections:
        _session_collections[session_id] = []
    
    # Default settings
    cl.user_session.set("search_mode", "legal_only")
    cl.user_session.set("use_reranker", True)  # ON by default
    
    try:
        loop = asyncio.get_event_loop()
        
        # Initialize pipeline
        await msg.stream_token("\n📚 Loading RAG Pipeline...")
        await loop.run_in_executor(None, get_demo_pipeline)
        
        # Initialize reranker
        await msg.stream_token("\n🔄 Loading Cross-Encoder Reranker (BGE-v2-m3)...")
        await loop.run_in_executor(None, get_reranker)
        
        # Get stats
        pipeline = get_demo_pipeline()
        stats = pipeline.get_stats()
        qdrant_stats = stats.get("qdrant", {})
        doc_count = qdrant_stats.get("points_count", 0)
        
        msg.content = f"""# 🏛️ Hệ Thống Tư Vấn Pháp Luật 


---

## 💡 Hướng Dẫn

### 🔍 Tìm kiếm pháp luật
Hỏi bất kỳ câu hỏi pháp lý - **điều luật phù hợp nhất** sẽ được hiển thị đầy đủ!

### ⚙️ Commands
| Lệnh | Mô tả |
|------|-------|
| `/legal` | Chỉ tìm Legal DB |
| `/file` | Chỉ tìm files đã upload |
| `/both` | Kết hợp cả hai (RRF) |
| `/rerank` | Bật/Tắt reranker |

### 📄 Upload tài liệu
Click 📎 để upload PDF/TXT 

---

**🔄 Reranker: ON** | 

"""
        
        await msg.update()
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[INIT ERROR] {error_trace}")
        msg.content = f"⚠️ Lỗi khởi tạo: {str(e)}\n\nVui lòng reload trang."
        await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message with reranking"""
    query = message.content.strip()
    session_id = cl.user_session.get("id")
    
    # Handle commands
    if query.startswith("/"):
        if query == "/legal":
            cl.user_session.set("search_mode", "legal_only")
            await cl.Message(content="✅ **Mode:** 📚 Legal Database Only").send()
            return
        elif query == "/file":
            cl.user_session.set("search_mode", "file_only")
            await cl.Message(content="✅ **Mode:** 📄 Uploaded Files Only").send()
            return
        elif query == "/both":
            cl.user_session.set("search_mode", "both")
            await cl.Message(content="✅ **Mode:** 🔀 Both (Legal + Files)").send()
            return
        elif query == "/rerank":
            current = cl.user_session.get("use_reranker", True)
            cl.user_session.set("use_reranker", not current)
            status = "ON ✅" if not current else "OFF ❌"
            await cl.Message(content=f"🔄 **Reranker:** {status}").send()
            return
    
    # Handle file upload
    if message.elements:
        for element in message.elements:
            if hasattr(element, 'path') and element.path:
                mime = getattr(element, 'mime', 'text/plain')
                file_name = element.name
                
                process_msg = cl.Message(content=f"📄 **Đang xử lý: {file_name}**")
                await process_msg.send()
                
                await process_msg.stream_token("\n🔄 Đang đọc file...")
                file_text = extract_text_from_file(element.path, mime)
                
                if file_text.startswith("❌"):
                    await process_msg.stream_token(f"\n\n{file_text}")
                    return
                
                await process_msg.stream_token(f"\n✅ Đọc: {len(file_text):,} ký tự")
                
                await process_msg.stream_token("\n🔄 Legal chunking...")
                loop = asyncio.get_event_loop()
                chunks = await loop.run_in_executor(None, chunk_uploaded_file, file_text, file_name)
                await process_msg.stream_token(f"\n✅ Tạo: {len(chunks)} chunks")
                
                collection_name = f"upload_{session_id[:8]}_{int(time.time())}"
                
                await process_msg.stream_token(f"\n🔄 Đang embed & upload...")
                pipeline = get_demo_pipeline()
                success = await upload_to_qdrant_collection(chunks, collection_name, pipeline)
                
                if success:
                    _session_collections[session_id].append({
                        "collection_name": collection_name,
                        "file_name": file_name,
                        "chunks": len(chunks)
                    })
                    
                    cl.user_session.set("search_mode", "file_only")
                    total_files = len(_session_collections[session_id])
                    
                    await process_msg.stream_token(f"""

🎯 **File indexed thành công!**
- Collection: `{collection_name}`
- {len(chunks)} chunks
- {len(file_text):,} ký tự

✅ **Mode:** 📄 Uploaded Files
📚 **Tổng:** {total_files} file(s)
🔄 **Reranker:** Sẵn sàng rerank kết quả!

_Gõ `/legal` để search legal DB, `/both` để search cả 2_""")
                else:
                    await process_msg.stream_token("\n\n❌ Lỗi upload file")
                
                return
    
    if not query:
        return
    
    # Get settings
    search_mode = cl.user_session.get("search_mode", "legal_only")
    use_reranker = cl.user_session.get("use_reranker", True)
    session_data = _session_collections.get(session_id, [])
    has_uploaded_files = len(session_data) > 0
    
    # Validate mode
    if search_mode in ["file_only", "both"] and not has_uploaded_files:
        await cl.Message(content="⚠️ Chưa có file upload. Đang search legal DB...").send()
        search_mode = "legal_only"
        cl.user_session.set("search_mode", "legal_only")
    
    # Create response
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        start_time = time.time()
        pipeline = get_demo_pipeline()
        reranker = get_reranker() if use_reranker else None
        loop = asyncio.get_event_loop()
        
        mode_emoji = {"legal_only": "📚", "file_only": "📄", "both": "🔀"}.get(search_mode, "🔍")
        rerank_status = "🔄 ON" if use_reranker else "❌ OFF"
        
        await msg.stream_token(f"{mode_emoji} Đang tìm kiếm... | Reranker: {rerank_status}\n\n")
        
        # ================== LEGAL ONLY MODE ==================
        if search_mode == "legal_only":
            # Initial retrieval (more candidates for reranking)
            retrieval_k = 20 if use_reranker else 10
            
            def search_legal():
                return pipeline.query(
                    question=query,
                    top_k=retrieval_k,
                    use_reranker=False  # We'll rerank ourselves
                )
            
            result = await loop.run_in_executor(None, search_legal)
            search_time = result.retrieval_time
            
            # Prepare results for reranking
            results = [
                {"content": c.get("content", ""), "score": 0.9, "metadata": c}
                for c in result.citations
            ]
            
            # Rerank if enabled
            rerank_time = 0.0
            if use_reranker and reranker and results:
                results, rerank_time = rerank_results(query, results, reranker, top_k=10, alpha=0.6)
            
            # Build context from (reranked) results
            context = "\n\n".join([r["content"] for r in results[:10]])
            
            # Generate answer
            gen_start = time.time()
            def gen():
                return pipeline.llm.generate(query, context)
            answer = await loop.run_in_executor(None, gen)
            gen_time = time.time() - gen_start
            
            # Display
            msg.content = ""
            header = f"📚 **{len(results)} văn bản pháp luật**"
            if use_reranker:
                header += f" | 🔄 Reranked"
            await msg.stream_token(f"{header} (⏱️ {search_time:.1f}s + {rerank_time:.1f}s)\n\n---\n\n")
            
            # Show TOP 1 reranked document với FULL content
            if use_reranker and results:
                await msg.stream_token(format_top1_reranked(results[0]))
                await msg.stream_token("\n---\n\n")
            
            # Stream answer
            for i in range(0, len(answer), 20):
                await msg.stream_token(answer[i:i+20])
                await asyncio.sleep(0.005)
            
            # Citations
            if results:
                await msg.stream_token("\n\n---\n\n📖 **Nguồn:**\n")
                for r in results[:5]:
                    meta = r.get("metadata", {})
                    ten_vb = meta.get("ten_van_ban", "N/A")
                    dieu = meta.get("dieu_so", "")
                    cite = f"- Điều {dieu}" if dieu else "- Văn bản"
                    cite += f" - _{ten_vb[:50]}..._\n" if len(ten_vb) > 50 else f" - _{ten_vb}_\n"
                    
                    if use_reranker and "rerank_score" in r:
                        cite = cite.rstrip("\n") + f" [Score: {r['rerank_score']:.2f}]\n"
                    
                    await msg.stream_token(cite)
            
            total_time = time.time() - start_time
            await msg.stream_token(f"\n⚡ _{total_time:.1f}s (Search: {search_time:.1f}s | Rerank: {rerank_time:.1f}s | Gen: {gen_time:.1f}s)_")
        
        # ================== FILE ONLY MODE ==================
        elif search_mode == "file_only":
            retrieval_k = 20 if use_reranker else 10
            all_results = []
            
            for file_info in session_data:
                collection_name = file_info["collection_name"]
                file_results = await search_in_collection(query, collection_name, pipeline, top_k=retrieval_k)
                for r in file_results:
                    r["source_file"] = file_info["file_name"]
                all_results.extend(file_results)
            
            all_results.sort(key=lambda x: x["score"], reverse=True)
            search_time = time.time() - start_time
            
            # Rerank
            rerank_time = 0.0
            if use_reranker and reranker and all_results:
                all_results, rerank_time = rerank_results(query, all_results, reranker, top_k=10, alpha=0.6)
            else:
                all_results = all_results[:10]
            
            total_files = len(session_data)
            total_chunks = sum(f["chunks"] for f in session_data)
            
            msg.content = ""
            header = f"📄 **{total_files} file(s)** - {len(all_results)} chunks từ {total_chunks}"
            if use_reranker:
                header += f" | 🔄 Reranked"
            await msg.stream_token(f"{header} (⏱️ {search_time:.1f}s + {rerank_time:.1f}s)\n\n---\n\n")
            
            # Show TOP 1 reranked document
            if use_reranker and all_results:
                await msg.stream_token(format_top1_reranked(all_results[0]))
                await msg.stream_token("\n---\n\n")
            
            # Generate
            context = "\n\n".join([r["content"] for r in all_results])
            gen_start = time.time()
            def gen():
                return pipeline.llm.generate(query, context)
            answer = await loop.run_in_executor(None, gen)
            gen_time = time.time() - gen_start
            
            for i in range(0, len(answer), 20):
                await msg.stream_token(answer[i:i+20])
                await asyncio.sleep(0.005)
            
            # Sources
            await msg.stream_token(f"\n\n---\n\n📎 **Nguồn từ files:**\n")
            seen_files = set()
            for r in all_results[:5]:
                src = r.get("source_file", "Unknown")
                if src not in seen_files:
                    score_str = f" [Score: {r['rerank_score']:.2f}]" if use_reranker and "rerank_score" in r else ""
                    await msg.stream_token(f"- {src}{score_str}\n")
                    seen_files.add(src)
            
            await msg.stream_token(f"\n⚡ _{time.time()-start_time:.1f}s (Search: {search_time:.1f}s | Rerank: {rerank_time:.1f}s | Gen: {gen_time:.1f}s)_")
        
        # ================== BOTH MODE ==================
        elif search_mode == "both":
            retrieval_k = 25 if use_reranker else 15
            
            # Search legal DB
            legal_task = loop.run_in_executor(
                None, 
                lambda: pipeline.query(query, top_k=retrieval_k, use_reranker=False)
            )
            
            # Search all files
            file_tasks = []
            for file_info in session_data:
                task = search_in_collection(query, file_info["collection_name"], pipeline, top_k=retrieval_k)
                file_tasks.append((file_info["file_name"], task))
            
            result_legal = await legal_task
            
            all_file_results = []
            for file_name, task in file_tasks:
                results = await task
                for r in results:
                    r["source_file"] = file_name
                all_file_results.extend(results)
            
            search_time = time.time() - start_time
            
            # RRF merge
            legal_dicts = [
                {"content": c.get("content", ""), "score": 0.9, "metadata": c, "source_file": "Legal DB"}
                for c in result_legal.citations
            ]
            merged = merge_results_rrf(legal_dicts, all_file_results, k=60)
            
            # Rerank merged results
            rerank_time = 0.0
            if use_reranker and reranker and merged:
                merged, rerank_time = rerank_results(query, merged, reranker, top_k=15, alpha=0.6)
            else:
                merged = merged[:15]
            
            total_files = len(session_data)
            msg.content = ""
            header = f"🔀 **Kết hợp:** 📚 Legal + 📄 {total_files} file(s) ({len(merged)} results)"
            if use_reranker:
                header += f" | 🔄 Reranked"
            await msg.stream_token(f"{header} (⏱️ {search_time:.1f}s + {rerank_time:.1f}s)\n\n---\n\n")
            
            # Show TOP 1 reranked document
            if use_reranker and merged:
                await msg.stream_token(format_top1_reranked(merged[0]))
                await msg.stream_token("\n---\n\n")
            
            # Generate
            context = "\n\n".join([r["content"] for r in merged])
            gen_start = time.time()
            def gen():
                return pipeline.llm.generate(query, context)
            answer = await loop.run_in_executor(None, gen)
            gen_time = time.time() - gen_start
            
            for i in range(0, len(answer), 20):
                await msg.stream_token(answer[i:i+20])
                await asyncio.sleep(0.005)
            
            # Source breakdown
            await msg.stream_token(f"\n\n---\n\n🔀 **Nguồn (RRF + Rerank):**\n")
            source_counts = {}
            for r in merged:
                src = r.get("source_file", "Unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            for src, count in source_counts.items():
                await msg.stream_token(f"- {src}: {count} chunk(s)\n")
            
            await msg.stream_token(f"\n⚡ _{time.time()-start_time:.1f}s (Search: {search_time:.1f}s | Rerank: {rerank_time:.1f}s | Gen: {gen_time:.1f}s)_")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] {error_trace}")
        await msg.stream_token(f"\n\n❌ **Lỗi:** {str(e)}")


@cl.on_chat_end
async def on_chat_end():
    """Cleanup when session ends"""
    session_id = cl.user_session.get("id")
    
    if session_id in _session_collections:
        session_files = _session_collections[session_id]
        
        if session_files:
            pipeline = get_demo_pipeline()
            for file_info in session_files:
                collection_name = file_info.get("collection_name")
                if collection_name:
                    try:
                        pipeline.qdrant_store.client.delete_collection(collection_name)
                        print(f"[CLEANUP] Deleted: {collection_name}")
                    except Exception as e:
                        print(f"[CLEANUP] Error: {e}")
        
        del _session_collections[session_id]


if __name__ == "__main__":
    print("🚀 PREMIUM VERSION - Legal RAG with Cross-Encoder Reranking!")
    print("Run: chainlit run chatbot_demo_reranker.py -w")
