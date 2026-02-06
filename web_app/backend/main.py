"""
Legal RAG Web API - FastAPI Backend

Kiến trúc Dual-Store:
1. legal_rag_hybrid - Kho Luật chính (100k+ điều luật)
2. user_docs_private - Kho Cá nhân (per user/session)

Endpoints:
- POST /api/chat - Chat với AI (hỗ trợ reranker toggle)
- POST /api/search - Tìm kiếm hybrid
- POST /api/upload - Upload file người dùng
- GET /api/documents - Lấy danh sách tài liệu user
- DELETE /api/documents/{doc_id} - Xóa tài liệu
- GET /api/history - Lịch sử chat
- POST /api/score - Chấm điểm câu trả lời
- GET/PUT /api/settings/reranker - Toggle reranker

Author: Legal RAG System
Date: 2024-12
Updated: 2024-12 - Added Vietnamese Reranker & Answer Scoring
"""

import os
import sys
import uuid
import time
import json
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Add parent to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
from user_store import get_user_store, UserDocumentStore
from smart_chunker import SmartRecursiveChunker
from vietnamese_reranker import get_reranker, get_answer_scorer
from quality_monitor import get_quality_monitor
from database import (
    get_db,
    save_chat_session,
    save_message,
    save_answer_metrics,
    save_message_sources,
    get_session_messages,
    get_metrics_stats
)

# Environment setup
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("OLLAMA_NUM_GPU", "999")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Collections
    LEGAL_COLLECTION = "legal_rag_hybrid"  # Core legal docs
    USER_COLLECTION = "user_docs_private"   # User uploads
    
    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "huyydangg/DEk21_hcmute_embedding")
    
    # LLM
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    
    # Paths
    UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
    BM25_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "bm25_hybrid.json"
    
    # Search
    TOP_K = int(os.getenv("TOP_K", "10"))
    
    # Reranker Settings
    RERANKER_ENABLED = True  # Default enabled
    RERANKER_ALPHA = 1     # Weight: 0.5 = balanced (50% reranker + 50% original)
    
    # Quality Evaluation Settings
    TOP_CONTEXTS_FOR_LLM = 5  # Number of contexts sent to LLM and used for quality eval


# Global settings state
_app_settings = {
    "reranker_enabled": Config.RERANKER_ENABLED,
    "reranker_alpha": Config.RERANKER_ALPHA,
    "top_contexts_for_llm": Config.TOP_CONTEXTS_FOR_LLM
}


# Ensure upload directory exists
Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    search_mode: str = "hybrid"  # legal, user, hybrid
    top_k: int = 10
    use_reranker: Optional[bool] = None  # None = use global setting

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    search_time: float
    generation_time: float
    total_time: float
    rerank_time: float = 0.0  # Time spent on reranking
    reranker_used: bool = False  # Whether reranker was applied
    
    # Quality metrics (added for consistency)
    quality_grade: Optional[str] = None
    quality_score: Optional[float] = None
    bert_score: Optional[float] = None
    hallucination_score: Optional[float] = None
    factuality_score: Optional[float] = None
    context_relevance: Optional[float] = None
    quality_feedback: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: Optional[str] = None
    search_mode: str = "hybrid"  # legal, user, hybrid
    top_k: int = 10
    use_reranker: Optional[bool] = None  # None = use global setting

class SearchResponse(BaseModel):
    results: List[Dict]
    search_time: float
    mode: str
    rerank_time: float = 0.0
    reranker_used: bool = False


class ScoreRequest(BaseModel):
    """Request để chấm điểm câu trả lời"""
    query: str
    answer: str
    contexts: List[Dict]  # List of context documents
    top_k: int = 5  # Số contexts để đánh giá


class ScoreResponse(BaseModel):
    """Response chấm điểm câu trả lời"""
    overall_score: float  # 0-100
    grade: str  # A, B, C, D, F
    feedback: str
    details: Dict  # Detailed scoring breakdown


class RerankerSettingsResponse(BaseModel):
    """Reranker settings"""
    enabled: bool
    alpha: float
    model_name: str
    model_loaded: bool

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    upload_time: str
    status: str

class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    chunk_count: int
    message: str

class ChatHistoryItem(BaseModel):
    id: str
    title: str
    timestamp: str
    preview: str


# ============================================================================
# Global Resources (Singleton Pattern)
# ============================================================================

_qdrant_client = None
_embedding_model = None
_bm25_encoder = None
_ollama_client = None
_chat_history: Dict[str, List[Dict]] = {}  # session_id -> messages


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT
        )
        logger.info(f"[QDRANT] Connected to {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
    return _qdrant_client


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[EMBED] Loading {Config.EMBEDDING_MODEL} on {device}...")
        
        _embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
        
        if device == "cuda":
            _embedding_model.half()
            import torch
            logger.info(f"[EMBED] GPU: {torch.cuda.get_device_name(0)}, FP16 enabled")
    
    return _embedding_model


def get_bm25_encoder():
    global _bm25_encoder
    if _bm25_encoder is None:
        if Config.BM25_CACHE_PATH.exists():
            with open(Config.BM25_CACHE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            class SimpleBM25:
                def __init__(self, data):
                    self.vocab = data["vocab"]
                    self.idf = data["idf"]
                    self.avg_doc_len = data["avg_doc_len"]
                    self.k1 = data.get("k1", 1.5)
                    self.b = data.get("b", 0.75)
                    self._tokenizer = None
                
                def _load_tokenizer(self):
                    if self._tokenizer is None:
                        try:
                            from pyvi import ViTokenizer
                            self._tokenizer = ViTokenizer.tokenize
                        except:
                            self._tokenizer = lambda x: x
                
                def tokenize(self, text):
                    self._load_tokenizer()
                    text = self._tokenizer(text.lower())
                    tokens = []
                    for word in text.split():
                        clean = ''.join(c for c in word if c.isalnum() or c == '_')
                        if clean:
                            tokens.append(clean)
                    return tokens
                
                def encode(self, text):
                    tokens = self.tokenize(text)
                    doc_len = len(tokens)
                    
                    if doc_len == 0:
                        return [], []
                    
                    tf = {}
                    for token in tokens:
                        tf[token] = tf.get(token, 0) + 1
                    
                    indices = []
                    values = []
                    
                    for token, freq in tf.items():
                        if token in self.vocab:
                            idx = self.vocab[token]
                            idf = self.idf.get(token, 0)
                            numerator = freq * (self.k1 + 1)
                            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                            score = idf * (numerator / denominator)
                            if score > 0:
                                indices.append(idx)
                                values.append(float(score))
                    
                    return indices, values
            
            _bm25_encoder = SimpleBM25(data)
            logger.info(f"[BM25] Loaded with {len(data['vocab'])} vocab")
        else:
            logger.warning("[BM25] Cache not found, sparse search disabled")
    
    return _bm25_encoder


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama
        
        class OllamaClient:
            def __init__(self, model):
                self.model = model
            
            def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
                try:
                    response = ollama.generate(
                        model=self.model,
                        prompt=prompt,
                        options={
                            'num_predict': max_tokens,
                            'temperature': temperature,
                            'num_ctx': 2048
                        }
                    )
                    return response['response']
                except Exception as e:
                    return f"❌ Lỗi LLM: {str(e)}"
        
        _ollama_client = OllamaClient(Config.OLLAMA_MODEL)
        logger.info(f"[LLM] Initialized Ollama with {Config.OLLAMA_MODEL}")
    
    return _ollama_client


# ============================================================================
# Search Functions
# ============================================================================

def search_legal_collection(
    query: str,
    top_k: int = 10
) -> List[Dict]:
    """Search trong Kho Luật chính (read-only)"""
    from qdrant_client.http import models
    
    client = get_qdrant_client()
    embed_model = get_embedding_model()
    bm25 = get_bm25_encoder()
    
    # Generate query vectors
    query_vector = embed_model.encode(query).tolist()
    
    if bm25:
        sparse_indices, sparse_values = bm25.encode(query)
        
        results = client.query_points(
            collection_name=Config.LEGAL_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=top_k * 2
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    ),
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )
    else:
        results = client.search(
            collection_name=Config.LEGAL_COLLECTION,
            query_vector=("dense", query_vector),
            limit=top_k,
            with_payload=True
        )
    
    documents = []
    for point in (results.points if hasattr(results, 'points') else results):
        doc = {
            "content": point.payload.get("content", ""),
            "score": point.score,
            "so_hieu": point.payload.get("so_hieu", ""),
            "dieu": point.payload.get("dieu", ""),
            "source_type": "legal",  # Đánh dấu nguồn từ Kho Luật
            "source_label": "Văn bản pháp luật"
        }
        documents.append(doc)
    
    return documents


def search_user_collection(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """Search trong Kho Cá nhân (với filter user_id)"""
    user_store = get_user_store()
    
    results = user_store.search(
        query=query,
        user_id=user_id,
        session_id=session_id,
        top_k=top_k,
        use_hybrid=True
    )
    
    # Add source labels
    for doc in results:
        doc["source_type"] = "user"  # Đánh dấu nguồn từ User Upload
        doc["source_label"] = f"📄 {doc.get('filename', 'Tài liệu người dùng')}"
    
    return results


def hybrid_search(
    query: str,
    user_id: str,
    session_id: Optional[str] = None,
    top_k: int = 10,
    search_mode: str = "hybrid",
    use_reranker: Optional[bool] = None  # None = use global setting
) -> tuple:
    """
    Tìm kiếm kết hợp từ cả 2 kho với optional reranking
    
    search_mode:
    - "legal": Chỉ tìm trong Kho Luật
    - "user": Chỉ tìm trong Kho Cá nhân
    - "hybrid": Tìm cả 2 kho và merge kết quả
    
    Returns:
    - (results, rerank_time, reranker_used)
    """
    legal_results = []
    user_results = []
    rerank_time = 0.0
    reranker_used = False
    
    if search_mode in ["legal", "hybrid"]:
        legal_results = search_legal_collection(query, top_k=top_k * 2)  # Fetch more for reranking
    
    if search_mode in ["user", "hybrid"]:
        user_results = search_user_collection(
            query=query,
            user_id=user_id,
            session_id=session_id,
            top_k=min(5, top_k)  # Limit user results
        )
    
    # Merge all results
    all_results = legal_results + user_results
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Determine if reranker should be used
    should_rerank = use_reranker if use_reranker is not None else _app_settings.get("reranker_enabled", False)
    
    if should_rerank and len(all_results) > 0:
        try:
            reranker = get_reranker()
            if reranker.load():
                rerank_start = time.time()
                
                # Prepare documents for reranking
                docs_to_rerank = [r.get("content", "") for r in all_results]
                
                # Rerank with alpha blending
                alpha = _app_settings.get("reranker_alpha", 0.5)
                reranked, _ = reranker.rerank(
                    query=query,
                    results=all_results,
                    top_k=top_k,
                    alpha=alpha
                )
                
                # Rebuild results from RerankedResult objects
                reranked_results = []
                for i, rr in enumerate(reranked):
                    # Find original result by content matching
                    original = next((r for r in all_results if r.get("content", "") == rr.content), {})
                    
                    # Create new result dict with rerank info
                    new_result = original.copy()
                    new_result["original_score"] = rr.original_score
                    new_result["score"] = rr.final_score
                    new_result["rerank_score"] = rr.rerank_score
                    new_result["rerank_rank"] = i + 1
                    
                    reranked_results.append(new_result)
                
                all_results = reranked_results
                reranker_used = True
                rerank_time = time.time() - rerank_start
                
                logger.info(f"[RERANKER] Reranked {len(all_results)} docs in {rerank_time:.3f}s")
            else:
                logger.warning("[RERANKER] Failed to load, using original ranking")
        except Exception as e:
            logger.error(f"[RERANKER] Error: {e}, using original ranking")
    
    return all_results[:top_k], rerank_time, reranker_used


# ============================================================================
# Prompt Building
# ============================================================================

def build_legal_prompt(query: str, contexts: List[Dict], top_k: int = None) -> str:
    """
    Build prompt with source-aware context
    
    Args:
        query: User question
        contexts: Retrieved contexts
        top_k: Number of contexts to use (default: TOP_CONTEXTS_FOR_LLM)
    """
    if top_k is None:
        top_k = Config.TOP_CONTEXTS_FOR_LLM
    
    context_text = ""
    for i, ctx in enumerate(contexts[:top_k], 1):  # Limit to top_k
        source_type = ctx.get("source_type", "legal")
        
        if source_type == "legal":
            # Nguồn từ Kho Luật
            so_hieu = ctx.get("so_hieu", "")
            dieu = ctx.get("dieu", "")
            header = ""
            if so_hieu and dieu:
                header = f"[📘 LUẬT] {dieu} - {so_hieu}"
            elif dieu:
                header = f"[📘 LUẬT] {dieu}"
            elif so_hieu:
                header = f"[📘 LUẬT] {so_hieu}"
            else:
                header = "[📘 LUẬT]"
        else:
            # Nguồn từ User Upload
            filename = ctx.get("filename", "Tài liệu")
            page = ctx.get("page", "")
            if page:
                header = f"[📄 TÀI LIỆU] {filename} | Trang {page}"
            else:
                header = f"[📄 TÀI LIỆU] {filename}"
        
        content = ctx.get("content", "")
        context_text += f"\n---\n**Nguồn {i}** {header}:\n{content}\n"
    
    prompt = f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên các tài liệu được cung cấp.


**NGUYÊN TẮC:**
1. Ưu tiên trích dẫn từ văn bản pháp luật chính thức (nguồn [📘 LUẬT])
2. Tham khảo thêm từ tài liệu người dùng (nguồn [📄 TÀI LIỆU]) nếu liên quan
3. Trích dẫn rõ ràng số hiệu văn bản, điều luật hoặc tên tài liệu
4. Nếu không tìm thấy thông tin, nói rõ "Không tìm thấy trong tài liệu"
5. Sử dụng ngôn ngữ pháp lý chính xác, dễ hiểu

**TÀI LIỆU THAM KHẢO:**
{context_text}

**CÂU HỎI:** {query}

**TRẢ LỜI:**"""
    
    return prompt


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan - initialize resources on startup"""
    logger.info("🚀 Starting Legal RAG API...")
    
    # Pre-load resources
    get_qdrant_client()
    get_embedding_model()
    get_bm25_encoder()
    get_ollama_client()
    get_user_store()
    
    logger.info("✅ All resources loaded")
    yield
    
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Legal RAG API",
    description="Vietnamese Legal RAG Chatbot API with Dual-Store Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Legal RAG API",
        "version": "1.0.0"
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    try:
        client = get_qdrant_client()
        
        # Get collection stats
        legal_info = client.get_collection(Config.LEGAL_COLLECTION)
        
        try:
            user_info = client.get_collection(Config.USER_COLLECTION)
            user_count = user_info.points_count
        except:
            user_count = 0
        
        return {
            "status": "ok",
            "legal_collection": Config.LEGAL_COLLECTION,
            "legal_documents": legal_info.points_count,
            "user_collection": Config.USER_COLLECTION,
            "user_documents": user_count,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.OLLAMA_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the legal AI assistant"""
    start_time = time.time()
    
    try:
        # Step 1: Search with optional reranking
        search_start = time.time()
        contexts, rerank_time, reranker_used = hybrid_search(
            query=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            search_mode=request.search_mode,
            use_reranker=request.use_reranker
        )
        search_time = time.time() - search_start - rerank_time  # Exclude rerank time
        
        if not contexts:
            # Build helpful error message based on search mode
            try:
                user_store = get_user_store()
                user_docs = user_store.get_user_documents(request.user_id, request.session_id)
                doc_count = len(user_docs)
            except:
                doc_count = 0
            
            if request.search_mode == "user":
                if doc_count == 0:
                    error_msg = "❌ Bạn chưa tải lên tài liệu nào. Vui lòng tải lên tài liệu trước khi tìm kiếm trong 'Tài liệu của tôi'."
                else:
                    error_msg = f"❌ Không tìm thấy nội dung phù hợp trong {doc_count} tài liệu của bạn.\n\n💡 Gợi ý:\n• Chuyển sang chế độ 'Kết hợp' (tìm cả tài liệu của bạn và văn bản luật)\n• Chuyển sang 'Văn bản pháp luật' (tìm trong kho luật)\n• Đặt câu hỏi khác liên quan đến nội dung tài liệu đã tải lên"
            elif request.search_mode == "legal":
                error_msg = "❌ Không tìm thấy văn bản pháp luật phù hợp.\n\n💡 Gợi ý:\n• Thử câu hỏi khác với từ khóa rõ ràng hơn\n• Sử dụng thuật ngữ pháp lý chính xác\n• Tải lên tài liệu riêng và chuyển sang chế độ 'Tài liệu của tôi'"
            else:  # hybrid
                error_msg = f"❌ Không tìm thấy tài liệu phù hợp (đã tìm trong {doc_count} tài liệu riêng và kho luật).\n\n💡 Gợi ý:\n• Đặt câu hỏi rõ ràng hơn với từ khóa cụ thể\n• Kiểm tra chính tả và ngữ pháp\n• Tải lên thêm tài liệu liên quan nếu cần"
            
            logger.warning(f"[CHAT] No results found. Mode={request.search_mode}, User docs={doc_count}, Legal results={len(legal_results)}, User results={len(user_results)}")
            
            return ChatResponse(
                answer=error_msg,
                sources=[],
                search_time=search_time,
                generation_time=0,
                total_time=time.time() - start_time,
                rerank_time=rerank_time,
                reranker_used=reranker_used
            )
        
        # Step 2: Generate answer (use TOP_CONTEXTS_FOR_LLM)
        gen_start = time.time()
        llm = get_ollama_client()
        contexts_for_llm = contexts[:Config.TOP_CONTEXTS_FOR_LLM]
        prompt = build_legal_prompt(request.message, contexts_for_llm, top_k=Config.TOP_CONTEXTS_FOR_LLM)
        answer = llm.generate(prompt, max_tokens=1024, temperature=0.1)
        gen_time = time.time() - gen_start
        
        # Step 3: Format sources for frontend (same as contexts used)
        sources = []
        seen = set()
        
        for ctx in contexts_for_llm:  # Same contexts used in LLM
            source_type = ctx.get("source_type", "legal")
            
            if source_type == "legal":
                so_hieu = ctx.get("so_hieu", "")
                dieu = ctx.get("dieu", "")
                key = (so_hieu, dieu)
                
                if key not in seen and (so_hieu or dieu):
                    seen.add(key)
                    sources.append({
                        "type": "legal",
                        "label": f"{dieu}" if dieu else so_hieu,
                        "detail": so_hieu if dieu else "",
                        "score": ctx.get("score", 0),
                        "content_preview": ctx.get("content", ""),  # Full content for preview
                        "content": ctx.get("content", "")  # Full content for modal
                    })
            else:
                filename = ctx.get("filename", "")
                page = ctx.get("page", "")
                key = (filename, page)
                
                if key not in seen and filename:
                    seen.add(key)
                    sources.append({
                        "type": "user",
                        "label": filename,
                        "detail": f"Trang {page}" if page else "",
                        "score": ctx.get("score", 0),
                        "content_preview": ctx.get("content", ""),  # Full content for preview
                        "content": ctx.get("content", "")  # Full content for modal
                    })
        
        # Step 4: Quality Monitoring (automatic evaluation with SAME contexts)
        quality_metrics = None
        try:
            monitor = get_quality_monitor()
            # Use EXACT SAME contexts that were sent to LLM
            context_texts = [ctx.get("content", "") for ctx in contexts_for_llm]
            quality_metrics = monitor.evaluate_answer(
                query=request.message,
                answer=answer,
                contexts=context_texts,
                prompt=prompt,
                top_k=Config.TOP_CONTEXTS_FOR_LLM  # Match contexts_for_llm
            )
            logger.info(f"✅ Quality metrics: Grade={quality_metrics.grade}, Score={quality_metrics.overall_score:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Quality monitoring failed: {e}")
        
        # Step 5: Save to database
        session_id = request.session_id
        user_id = request.user_id or "default"
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        
        try:
            # Ensure session exists
            save_chat_session(
                session_id=session_id,
                user_id=user_id,
                title=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            # Save user message
            save_message(
                message_id=user_message_id,
                session_id=session_id,
                role="user",
                content=request.message
            )
            
            # Save assistant message
            save_message(
                message_id=assistant_message_id,
                session_id=session_id,
                role="assistant",
                content=answer
            )
            
            # Save sources
            if sources:
                save_message_sources(assistant_message_id, sources)
            
            # Save quality metrics if available
            if quality_metrics:
                save_answer_metrics(
                    message_id=assistant_message_id,
                    overall_score=quality_metrics.overall_score,
                    grade=quality_metrics.grade,
                    feedback=quality_metrics.feedback,
                    bert_score=quality_metrics.bert_score,
                    hallucination_score=quality_metrics.hallucination_score,
                    factuality_score=quality_metrics.factuality_score,
                    context_relevance=quality_metrics.context_relevance,
                    query=request.message,
                    answer=answer,
                    top_k_contexts=Config.TOP_CONTEXTS_FOR_LLM,  # Actual contexts used
                    contexts_used=len(contexts_for_llm),  # Actual contexts count
                    reranker_used=reranker_used
                )
                logger.info(f"💾 Metrics saved to database")
            
        except Exception as e:
            logger.error(f"❌ Database save failed: {e}")
        
        # Save to in-memory history (for backward compatibility)
        if session_id not in _chat_history:
            _chat_history[session_id] = []
        
        _chat_history[session_id].append({
            "id": user_message_id,
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        _chat_history[session_id].append({
            "id": assistant_message_id,
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "quality_grade": quality_metrics.grade if quality_metrics else None,
            "quality_score": quality_metrics.overall_score if quality_metrics else None
        })
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            search_time=search_time,
            generation_time=gen_time,
            total_time=time.time() - start_time,
            rerank_time=rerank_time,
            reranker_used=reranker_used,
            # Quality metrics
            quality_grade=quality_metrics.grade if quality_metrics else None,
            quality_score=quality_metrics.overall_score if quality_metrics else None,
            bert_score=quality_metrics.bert_score if quality_metrics else None,
            hallucination_score=quality_metrics.hallucination_score if quality_metrics else None,
            factuality_score=quality_metrics.factuality_score if quality_metrics else None,
            context_relevance=quality_metrics.context_relevance if quality_metrics else None,
            quality_feedback=quality_metrics.feedback if quality_metrics else None
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search documents"""
    start_time = time.time()
    
    try:
        results, rerank_time, reranker_used = hybrid_search(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            search_mode=request.search_mode,
            use_reranker=request.use_reranker
        )
        
        return SearchResponse(
            results=results,
            search_time=time.time() - start_time - rerank_time,
            mode=request.search_mode,
            rerank_time=rerank_time,
            reranker_used=reranker_used
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    session_id: str = Form(default=None)
):
    """Upload user document"""
    
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Validate file type
    allowed_types = ['.pdf', '.docx', '.doc', '.txt']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Save file temporarily
    temp_path = Config.UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    
    try:
        logger.info(f"[UPLOAD] Receiving file: {file.filename} ({file_ext})")
        
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"[UPLOAD] Saved to: {temp_path}, size: {temp_path.stat().st_size} bytes")
        
        # Process file
        user_store = get_user_store()
        success, doc_id, chunk_count = await user_store.upload_file(
            file_path=str(temp_path),
            user_id=user_id,
            session_id=session_id,
            filename=file.filename
        )
        
        logger.info(f"[UPLOAD] Processing result: success={success}, chunks={chunk_count}")
        
        if success:
            return UploadResponse(
                success=True,
                doc_id=doc_id,
                filename=file.filename,
                chunk_count=chunk_count,
                message=f"Đã tải lên và xử lý {chunk_count} đoạn văn bản"
            )
        else:
            logger.error(f"[UPLOAD] Processing failed for {file.filename}")
            
            # Check if it's a .doc file that couldn't be processed
            if file_ext == '.doc':
                raise HTTPException(
                    status_code=400, 
                    detail="Không thể xử lý file .doc (Word 97-2003 cũ). Vui lòng chuyển đổi sang .docx: Mở file trong Word → Save As → chọn .docx"
                )
            
            raise HTTPException(status_code=500, detail=f"Không thể xử lý file {file.filename}")
            
    except Exception as e:
        logger.error(f"[UPLOAD] Upload error for {file.filename}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/documents")
async def get_documents(
    user_id: str = "default_user",
    session_id: Optional[str] = None
):
    """Get user's uploaded documents"""
    try:
        user_store = get_user_store()
        documents = user_store.get_user_documents(user_id, session_id)
        
        return {
            "documents": [doc.to_dict() for doc in documents]
        }
    except Exception as e:
        logger.error(f"Get documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user_id: str = "default_user"
):
    """Delete user document"""
    try:
        user_store = get_user_store()
        success = user_store.delete_document(doc_id, user_id)
        
        if success:
            return {"message": f"Document {doc_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(session_id: str):
    """Get chat history for session"""
    if session_id in _chat_history:
        return {"messages": _chat_history[session_id]}
    return {"messages": []}


@app.get("/api/sessions")
async def get_sessions(user_id: str = "default_user"):
    """Get list of chat sessions"""
    sessions = []
    
    for session_id, messages in _chat_history.items():
        if messages:
            # Get first user message as title
            first_msg = next(
                (m for m in messages if m.get("role") == "user"),
                None
            )
            if first_msg:
                sessions.append({
                    "id": session_id,
                    "title": first_msg["content"][:50] + "..." if len(first_msg["content"]) > 50 else first_msg["content"],
                    "timestamp": first_msg.get("timestamp", ""),
                    "message_count": len(messages)
                })
    
    # Sort by timestamp descending
    sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {"sessions": sessions}


# ============================================================================
# Reranker & Scoring Endpoints
# ============================================================================

@app.get("/api/settings/reranker", response_model=RerankerSettingsResponse)
async def get_reranker_settings():
    """Get current reranker settings"""
    reranker = get_reranker()
    
    return RerankerSettingsResponse(
        enabled=_app_settings.get("reranker_enabled", False),
        alpha=_app_settings.get("reranker_alpha", 0.5),
        model_name=reranker.MODEL_NAME,
        model_loaded=reranker.model is not None
    )


@app.put("/api/settings/reranker")
async def update_reranker_settings(
    enabled: bool = None,
    alpha: float = None
):
    """Update reranker settings"""
    global _app_settings
    
    if enabled is not None:
        _app_settings["reranker_enabled"] = enabled
        logger.info(f"[SETTINGS] Reranker enabled: {enabled}")
    
    if alpha is not None:
        if 0.0 <= alpha <= 1.0:
            _app_settings["reranker_alpha"] = alpha
            logger.info(f"[SETTINGS] Reranker alpha: {alpha}")
        else:
            raise HTTPException(status_code=400, detail="Alpha must be between 0.0 and 1.0")
    
    # Preload model if enabling
    if enabled:
        try:
            reranker = get_reranker()
            reranker.load()
        except Exception as e:
            logger.warning(f"[RERANKER] Preload failed: {e}")
    
    return {
        "enabled": _app_settings["reranker_enabled"],
        "alpha": _app_settings["reranker_alpha"],
        "message": "Settings updated successfully"
    }


@app.post("/api/score", response_model=ScoreResponse)
async def score_answer(request: ScoreRequest):
    """
    Score answer quality using QualityMonitor (BERTScore-based)
    
    [PRIMARY SCORER] Uses QualityMonitor for consistent evaluation across all features.
    This ensures chat, dashboard, and API scoring use the same methodology.
    
    Returns:
    - overall_score: Weighted combination of all metrics
    - grade: A/B/C/D/F letter grade
    - bert_score: Semantic similarity (BERTScore F1)
    - hallucination_score: Claim verification score
    - factuality_score: Fact accuracy score
    - context_relevance: Query-context relevance
    """
    try:
        # Use QualityMonitor as primary scorer
        monitor = get_quality_monitor()
        
        # Extract content strings from contexts
        context_texts = []
        for ctx in request.contexts[:request.top_k]:
            if isinstance(ctx, dict):
                content = ctx.get("content", "")
            else:
                content = str(ctx)
            if content:
                context_texts.append(content)
        
        if not context_texts:
            return ScoreResponse(
                overall_score=0.0,
                grade="F",
                feedback="Không có tài liệu nguồn để đánh giá.",
                details={}
            )
        
        # Build dummy prompt for evaluation
        prompt = f"Question: {request.query}\\n\\nAnswer: {request.answer}"
        
        # Evaluate with QualityMonitor
        metrics = monitor.evaluate_answer(
            query=request.query,
            answer=request.answer,
            contexts=context_texts,
            prompt=prompt,
            top_k=request.top_k
        )
        
        return ScoreResponse(
            overall_score=metrics.overall_score,
            grade=metrics.grade,
            feedback=metrics.feedback,
            details={
                "bert_score": metrics.bert_score,
                "hallucination_score": metrics.hallucination_score,
                "factuality_score": metrics.factuality_score,
                "context_relevance": metrics.context_relevance,
                **metrics.details
            }
        )
        
    except Exception as e:
        logger.error(f"Score error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preload-reranker")
async def preload_reranker():
    """Preload reranker model to GPU"""
    try:
        reranker = get_reranker()
        success = reranker.load()
        
        if success:
            return {
                "status": "ok",
                "message": "Reranker model loaded successfully",
                "model": reranker.MODEL_NAME,
                "device": str(reranker.device)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load reranker model")
            
    except Exception as e:
        logger.error(f"Preload reranker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/stats")
async def get_metrics_statistics(
    user_id: Optional[str] = None,
    days: int = 7
):
    """
    Get aggregated quality metrics statistics
    
    Args:
        user_id: Filter by user (optional)
        days: Number of days to look back (default: 7)
    
    Returns:
        Statistics: average scores, grade distribution, trends
    """
    try:
        stats = get_metrics_stats(user_id=user_id, days=days)
        
        return {
            "status": "ok",
            "user_id": user_id,
            "days": days,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Get metrics stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/history")
async def get_metrics_history(
    session_id: str
):
    """
    Get metrics history for a session
    
    Args:
        session_id: Session ID
    
    Returns:
        List of messages with quality metrics
    """
    try:
        messages = get_session_messages(session_id=session_id)
        
        # Format for frontend
        history = []
        for msg in messages:
            item = {
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("created_at", ""),
                "sources": msg.get("sources", [])
            }
            
            # Add metrics if available
            if "metrics" in msg and msg["metrics"]:
                item["quality"] = {
                    "overall_score": msg["metrics"].get("overall_score"),
                    "grade": msg["metrics"].get("grade"),
                    "feedback": msg["metrics"].get("feedback"),
                    "bert_score": msg["metrics"].get("bert_score"),
                    "hallucination_score": msg["metrics"].get("hallucination_score"),
                    "factuality_score": msg["metrics"].get("factuality_score"),
                    "context_relevance": msg["metrics"].get("context_relevance")
                }
            
            history.append(item)
        
        return {
            "status": "ok",
            "session_id": session_id,
            "count": len(history),
            "messages": history
        }
        
    except Exception as e:
        logger.error(f"Get metrics history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/all")
async def get_all_metrics(
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    grade_filter: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """
    Get all quality metrics for dashboard
    
    Args:
        user_id: Filter by user (optional)
        limit: Max records to return (default: 100)
        offset: Pagination offset (default: 0)
        grade_filter: Filter by grade A/B/C/D/F (optional)
        sort_by: Field to sort by (default: created_at)
        sort_order: asc or desc (default: desc)
    
    Returns:
        List of all metrics with full details
    """
    try:
        from database import AnswerMetrics, Message, ChatSession
        
        with get_db() as db:
            # Build query
            query = db.query(AnswerMetrics, Message, ChatSession)\
                .join(Message, AnswerMetrics.message_id == Message.id)\
                .join(ChatSession, Message.session_id == ChatSession.id)
            
            # Apply filters
            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            
            if grade_filter and grade_filter in ['A', 'B', 'C', 'D', 'F']:
                query = query.filter(AnswerMetrics.grade == grade_filter)
            
            # Apply sorting
            if sort_by == "created_at":
                order_col = AnswerMetrics.created_at
            elif sort_by == "overall_score":
                order_col = AnswerMetrics.overall_score
            elif sort_by == "grade":
                order_col = AnswerMetrics.grade
            else:
                order_col = AnswerMetrics.created_at
            
            if sort_order == "desc":
                order_col = order_col.desc()
            else:
                order_col = order_col.asc()
            
            query = query.order_by(order_col)
            
            # Get total count before pagination
            total_count = query.count()
            
            # Apply pagination
            results = query.offset(offset).limit(limit).all()
            
            # Format results
            from database import MessageSource
            metrics_list = []
            for metric, message, session in results:
                # Get sources for this message
                sources = db.query(MessageSource)\
                    .filter(MessageSource.message_id == message.id)\
                    .order_by(MessageSource.rank)\
                    .all()
                
                item = {
                    "id": metric.id,
                    "message_id": metric.message_id,
                    "session_id": session.id,
                    "user_id": session.user_id,
                    "timestamp": metric.created_at.isoformat() if metric.created_at else None,
                    
                    # Quality metrics
                    "overall_score": metric.overall_score,
                    "grade": metric.grade,
                    "feedback": metric.feedback,
                    "bert_score": metric.bert_score,
                    "hallucination_score": metric.hallucination_score,
                    "factuality_score": metric.factuality_score,
                    "context_relevance": metric.context_relevance,
                    
                    # Content
                    "query": metric.query,
                    "answer": metric.answer,
                    
                    # Sources
                    "sources": [
                        {
                            "rank": s.rank,
                            "score": s.score,
                            "label": s.label,
                            "detail": s.detail,
                            "content": s.content_preview
                        }
                        for s in sources
                    ],
                    
                    # Metadata
                    "top_k_contexts": metric.top_k_contexts,
                    "contexts_used": metric.contexts_used,
                    "reranker_used": metric.reranker_used,
                    
                    # Timing from message
                    "search_time": message.search_time,
                    "generation_time": message.generation_time,
                    "rerank_time": message.rerank_time,
                    "total_time": message.total_time
                }
                
                metrics_list.append(item)
            
            return {
                "status": "ok",
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "count": len(metrics_list),
                "metrics": metrics_list
            }
        
    except Exception as e:
        logger.error(f"Get all metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/export")
async def export_all_metrics(
    format: str = "json",
    user_id: Optional[str] = None
):
    """
    Export all metrics to file
    
    Args:
        format: Export format (json, csv, txt)
        user_id: Filter by user (optional)
    
    Returns:
        File download
    """
    try:
        from database import AnswerMetrics, Message, ChatSession, MessageSource
        import io
        import csv
        
        with get_db() as db:
            # Get all metrics with related data
            query = db.query(AnswerMetrics, Message, ChatSession)\
                .join(Message, AnswerMetrics.message_id == Message.id)\
                .join(ChatSession, Message.session_id == ChatSession.id)
            
            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            
            results = query.order_by(AnswerMetrics.created_at.desc()).all()
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "json":
                # JSON export
                data = {
                    "export_time": datetime.now().isoformat(),
                    "total_records": len(results),
                    "user_id": user_id,
                    "metrics": []
                }
                
                for metric, message, session in results:
                    # Get sources
                    sources = db.query(MessageSource)\
                        .filter(MessageSource.message_id == message.id)\
                        .all()
                    
                    item = {
                        "id": metric.id,
                        "timestamp": metric.created_at.isoformat() if metric.created_at else None,
                        "session_id": session.id,
                        "user_id": session.user_id,
                        "quality_metrics": {
                            "overall_score": metric.overall_score,
                            "grade": metric.grade,
                            "feedback": metric.feedback,
                            "bert_score": metric.bert_score,
                            "hallucination_score": metric.hallucination_score,
                            "factuality_score": metric.factuality_score,
                            "context_relevance": metric.context_relevance
                        },
                        "message_content": {
                            "query": metric.query,
                            "answer": metric.answer
                        },
                        "context_info": {
                            "top_k": metric.top_k_contexts,
                            "used": metric.contexts_used,
                            "reranker_used": metric.reranker_used
                        },
                        "timing": {
                            "search_time": message.search_time,
                            "generation_time": message.generation_time,
                            "rerank_time": message.rerank_time,
                            "total_time": message.total_time
                        },
                        "sources": [
                            {
                                "rank": s.rank,
                                "score": s.score,
                                "label": s.label,
                                "details": s.details
                            }
                            for s in sources
                        ]
                    }
                    
                    data["metrics"].append(item)
                
                content = json.dumps(data, ensure_ascii=False, indent=2)
                media_type = "application/json"
                filename = f"quality_metrics_export_{timestamp}.json"
                
            elif format == "csv":
                # CSV export
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Headers
                writer.writerow([
                    "ID", "Timestamp", "Session ID", "User ID",
                    "Overall Score", "Grade", "Feedback",
                    "BERTScore", "Hallucination Score", "Factuality Score", "Context Relevance",
                    "Query", "Answer",
                    "Top K", "Contexts Used", "Reranker Used",
                    "Search Time", "Generation Time", "Rerank Time", "Total Time"
                ])
                
                # Data rows
                for metric, message, session in results:
                    writer.writerow([
                        metric.id,
                        metric.created_at.isoformat() if metric.created_at else "",
                        session.id,
                        session.user_id,
                        metric.overall_score,
                        metric.grade,
                        metric.feedback,
                        metric.bert_score,
                        metric.hallucination_score,
                        metric.factuality_score,
                        metric.context_relevance,
                        (metric.query or "")[:100],  # Truncate for CSV
                        (metric.answer or "")[:200],  # Truncate for CSV
                        metric.top_k_contexts,
                        metric.contexts_used,
                        metric.reranker_used,
                        message.search_time,
                        message.generation_time,
                        message.rerank_time,
                        message.total_time
                    ])
                
                content = output.getvalue()
                media_type = "text/csv"
                filename = f"quality_metrics_export_{timestamp}.csv"
                
            elif format == "txt":
                # TXT export
                lines = []
                lines.append("=" * 80)
                lines.append("QUALITY METRICS EXPORT REPORT")
                lines.append("=" * 80)
                lines.append(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append(f"Total Records: {len(results)}")
                if user_id:
                    lines.append(f"User ID: {user_id}")
                lines.append("=" * 80)
                lines.append("")
                
                for idx, (metric, message, session) in enumerate(results, 1):
                    lines.append(f"\n{'=' * 80}")
                    lines.append(f"RECORD #{idx}")
                    lines.append(f"{'=' * 80}")
                    lines.append(f"ID: {metric.id}")
                    lines.append(f"Timestamp: {metric.created_at.strftime('%Y-%m-%d %H:%M:%S') if metric.created_at else 'N/A'}")
                    lines.append(f"Session: {session.id}")
                    lines.append(f"User: {session.user_id}")
                    lines.append("")
                    lines.append("QUALITY METRICS:")
                    lines.append(f"  Overall Score: {metric.overall_score:.2%} (Grade: {metric.grade})")
                    lines.append(f"  BERTScore: {metric.bert_score:.2%}" if metric.bert_score else "  BERTScore: N/A")
                    lines.append(f"  Hallucination Score: {metric.hallucination_score:.2%}" if metric.hallucination_score else "  Hallucination Score: N/A")
                    lines.append(f"  Factuality Score: {metric.factuality_score:.2%}" if metric.factuality_score else "  Factuality Score: N/A")
                    lines.append(f"  Context Relevance: {metric.context_relevance:.2%}" if metric.context_relevance else "  Context Relevance: N/A")
                    lines.append(f"  Feedback: {metric.feedback}")
                    lines.append("")
                    lines.append("QUERY:")
                    lines.append(f"  {metric.query}")
                    lines.append("")
                    lines.append("ANSWER:")
                    lines.append(f"  {metric.answer}")
                    lines.append("")
                    lines.append("METADATA:")
                    lines.append(f"  Top K: {metric.top_k_contexts}")
                    lines.append(f"  Contexts Used: {metric.contexts_used}")
                    lines.append(f"  Reranker Used: {'Yes' if metric.reranker_used else 'No'}")
                    lines.append("")
                    lines.append("TIMING:")
                    lines.append(f"  Search: {message.search_time:.3f}s" if message.search_time else "  Search: N/A")
                    lines.append(f"  Generation: {message.generation_time:.3f}s" if message.generation_time else "  Generation: N/A")
                    lines.append(f"  Rerank: {message.rerank_time:.3f}s" if message.rerank_time else "  Rerank: N/A")
                    lines.append(f"  Total: {message.total_time:.3f}s" if message.total_time else "  Total: N/A")
                
                lines.append(f"\n{'=' * 80}")
                lines.append("END OF REPORT")
                lines.append("=" * 80)
                
                content = "\n".join(lines)
                media_type = "text/plain"
                filename = f"quality_metrics_export_{timestamp}.txt"
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
            # Return file download
            return StreamingResponse(
                io.BytesIO(content.encode('utf-8')),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        
    except Exception as e:
        logger.error(f"Export metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
