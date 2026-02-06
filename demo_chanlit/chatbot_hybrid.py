"""
Vietnamese Legal RAG Chatbot - Hybrid Version

Sử dụng collection: legal_rag_hybrid
- Dense: huyydangg/DEk21_hcmute_embedding (768D)
- Sparse: BM25 (tự build)
- Fusion: RRF (Reciprocal Rank Fusion)

Features:
- Hybrid Search (Dense + Sparse + RRF)
- GPU acceleration cho embedding + LLM
- Ollama LLM (qwen2.5:3b) - 100% GPU
- Vietnamese legal citation format
- Optimized loading (singleton pattern)

Usage:
    chainlit run chatbot_hybrid.py -w
"""

import os
import sys
from pathlib import Path
import asyncio
import time
import json
import ollama
from typing import Optional, List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "legal_rag_hybrid"
TOP_K = int(os.getenv("TOP_K", "10"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "huyydangg/DEk21_hcmute_embedding")
BM25_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "bm25_hybrid.json"

# Global instances
_embedding_model = None
_qdrant_client = None
_bm25_encoder = None
_llm_client = None


class SimpleBM25:
    """Simple BM25 encoder for Vietnamese legal text"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_doc_len: float = 0
        self.doc_count: int = 0
        self._tokenizer = None
    
    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                from pyvi import ViTokenizer
                self._tokenizer = ViTokenizer.tokenize
            except ImportError:
                self._tokenizer = lambda x: x
    
    def tokenize(self, text: str) -> List[str]:
        self._load_tokenizer()
        text = self._tokenizer(text.lower())
        tokens = []
        for word in text.split():
            clean = ''.join(c for c in word if c.isalnum() or c == '_')
            if clean:
                tokens.append(clean)
        return tokens
    
    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        
        if doc_len == 0:
            return [], []
        
        tf: Dict[str, int] = {}
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
    
    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_count = data["doc_count"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        
        return True


def get_embedding_model():
    """Get singleton embedding model"""
    global _embedding_model
    
    if _embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EMBED] Loading {EMBEDDING_MODEL} on {device}...")
        
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        
        if device == "cuda":
            _embedding_model.half()
            print(f"[EMBED] GPU: {torch.cuda.get_device_name(0)}, FP16 enabled")
    
    return _embedding_model


def get_qdrant_client():
    """Get singleton Qdrant client"""
    global _qdrant_client
    
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"[QDRANT] Connected to {QDRANT_HOST}:{QDRANT_PORT}")
    
    return _qdrant_client


def get_bm25_encoder():
    """Get singleton BM25 encoder"""
    global _bm25_encoder
    
    if _bm25_encoder is None:
        _bm25_encoder = SimpleBM25()
        if _bm25_encoder.load(str(BM25_CACHE_PATH)):
            print(f"[BM25] Loaded encoder with {len(_bm25_encoder.vocab)} vocab")
        else:
            print(f"[BM25] ❌ Cannot load BM25 cache from {BM25_CACHE_PATH}")
    
    return _bm25_encoder


def get_llm_client():
    """Get singleton LLM client"""
    global _llm_client
    
    if _llm_client is None:
        # Simple Ollama client wrapper
        import ollama
        
        class OllamaClient:
            def __init__(self, model="qwen2.5:3b"):
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
        
        _llm_client = OllamaClient(model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    
    return _llm_client


async def hybrid_search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Hybrid search using Dense + Sparse + RRF fusion
    """
    loop = asyncio.get_event_loop()
    
    def _search():
        from qdrant_client.http import models
        
        client = get_qdrant_client()
        bm25 = get_bm25_encoder()
        embed_model = get_embedding_model()
        
        # Generate dense embedding
        query_embedding = embed_model.encode(query).tolist()
        
        # Generate sparse vector
        sparse_indices, sparse_values = bm25.encode(query)
        
        # Hybrid search with RRF fusion
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
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
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=top_k,
            with_payload=True
        )
        
        return [
            {
                "content": point.payload.get("content", ""),
                "score": point.score,
                "so_hieu": point.payload.get("so_hieu", ""),
                "dieu": point.payload.get("dieu", ""),
                "is_split": point.payload.get("is_split", False),
                "payload": point.payload
            }
            for point in results.points
        ]
    
    return await loop.run_in_executor(None, _search)


def build_legal_prompt(query: str, contexts: List[Dict]) -> str:
    """Build prompt for Vietnamese legal Q&A"""
    
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        so_hieu = ctx.get("so_hieu", "")
        dieu = ctx.get("dieu", "")
        content = ctx.get("content", "")
        
        header = ""
        if so_hieu and dieu:
            header = f"[{dieu} - {so_hieu}]"
        elif dieu:
            header = f"[{dieu}]"
        elif so_hieu:
            header = f"[{so_hieu}]"
        
        context_text += f"\n---\n**Tài liệu {i}** {header}:\n{content}\n"
    
    prompt = f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên các tài liệu pháp lý được cung cấp.

**NGUYÊN TẮC:**
1. Chỉ trả lời dựa trên nội dung tài liệu được cung cấp
2. Trích dẫn rõ ràng số hiệu văn bản và điều luật
3. Nếu không tìm thấy thông tin, nói rõ "Không tìm thấy trong tài liệu"
4. Sử dụng ngôn ngữ pháp lý chính xác
5. Trả lời ngắn gọn, đúng trọng tâm

**TÀI LIỆU THAM KHẢO:**
{context_text}

**CÂU HỎI:** {query}

**TRẢ LỜI:**"""
    
    return prompt


async def generate_answer(query: str, contexts: List[Dict]) -> str:
    """Generate answer using LLM"""
    loop = asyncio.get_event_loop()
    
    def _generate():
        llm = get_llm_client()
        prompt = build_legal_prompt(query, contexts)
        
        response = llm.generate(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1
        )
        
        return response
    
    return await loop.run_in_executor(None, _generate)


def format_citations(contexts: List[Dict]) -> str:
    """Format legal citations"""
    if not contexts:
        return ""
    
    citation_text = "\n\n---\n\n📚 **CĂN CỨ PHÁP LÝ:**\n\n"
    seen = set()
    
    for i, ctx in enumerate(contexts[:5], 1):  # Top 5 citations
        so_hieu = ctx.get("so_hieu", "")
        dieu = ctx.get("dieu", "")
        
        key = (so_hieu, dieu)
        if key in seen:
            continue
        seen.add(key)
        
        if so_hieu and dieu:
            citation_text += f"**{len(seen)}.** {dieu} - {so_hieu}\n"
        elif dieu:
            citation_text += f"**{len(seen)}.** {dieu}\n"
        elif so_hieu:
            citation_text += f"**{len(seen)}.** {so_hieu}\n"
    
    return citation_text


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session"""
    msg = cl.Message(content="🔄 Đang khởi tạo hệ thống...")
    await msg.send()
    
    try:
        # Initialize components in background
        loop = asyncio.get_event_loop()
        
        def _init():
            get_qdrant_client()
            get_bm25_encoder()
            get_embedding_model()
            get_llm_client()
            
            # Get collection info
            client = get_qdrant_client()
            info = client.get_collection(COLLECTION_NAME)
            return info.points_count
        
        doc_count = await loop.run_in_executor(None, _init)
        
        msg.content = f"""# 🏛️ Hệ Thống Tư Vấn Pháp Luật Việt Nam

**Phiên bản:** Hybrid RAG v4
**Dữ liệu:** {doc_count:,} điều luật

---

## 📊 Thông tin hệ thống:
- **Vector DB:** Qdrant (Collection: `{COLLECTION_NAME}`)
- **Embedding:** {EMBEDDING_MODEL}
- **Search:** Hybrid (Dense + BM25 + RRF Fusion)
- **LLM:** Ollama (qwen2.5:3b)

---

## 💡 Cách sử dụng:
Đặt câu hỏi về pháp luật Việt Nam, ví dụ:
- "Tội giết người bị phạt bao nhiêu năm?"
- "Điều kiện thành lập công ty TNHH?"
- "Mức phạt đi xe máy không đội mũ bảo hiểm?"

---

✅ **Sẵn sàng tư vấn!**
"""
        await msg.update()
        
    except Exception as e:
        msg.content = f"❌ **Lỗi khởi tạo:** {str(e)}\n\nVui lòng kiểm tra:\n- Qdrant đang chạy (`docker start qdrant`)\n- Ollama đã pull model (`ollama pull qwen2.5:3b`)"
        await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message"""
    query = message.content.strip()
    
    if not query:
        return
    
    # Show thinking indicator
    thinking_msg = cl.Message(content="🔍 Đang tìm kiếm và phân tích...")
    await thinking_msg.send()
    
    start_time = time.time()
    
    try:
        # Step 1: Hybrid search
        search_start = time.time()
        contexts = await hybrid_search(query, top_k=TOP_K)
        search_time = time.time() - search_start
        
        if not contexts:
            thinking_msg.content = "❌ Không tìm thấy tài liệu phù hợp."
            await thinking_msg.update()
            return
        
        # Step 2: Generate answer
        gen_start = time.time()
        answer = await generate_answer(query, contexts)
        gen_time = time.time() - gen_start
        
        # Step 3: Format citations
        citations = format_citations(contexts)
        
        # Step 4: Build final response
        total_time = time.time() - start_time
        
        # Performance metrics
        perf_text = f"\n\n---\n⏱️ _Retrieval: {search_time:.2f}s | Generation: {gen_time:.2f}s | Total: {total_time:.2f}s_"
        
        thinking_msg.content = answer + citations + perf_text
        await thinking_msg.update()
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] {error_detail}")
        
        thinking_msg.content = f"❌ **Lỗi xử lý:** {str(e)}"
        await thinking_msg.update()


@cl.on_chat_end
async def on_chat_end():
    """Cleanup on session end"""
    print("[SESSION] Chat ended")


if __name__ == "__main__":
    print("Run with: chainlit run chatbot_hybrid.py -w")
