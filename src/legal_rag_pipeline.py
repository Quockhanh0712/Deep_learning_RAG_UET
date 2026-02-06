"""
Legal RAG Pipeline with Qdrant

Features:
- Intent-aware hybrid search
- Legal citation formatting
- Multi-source answer generation
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Lazy imports
QdrantLegalStore = None
QueryIntentClassifier = None
IntentResult = None


def _lazy_imports():
    """Lazy import heavy modules"""
    global QdrantLegalStore, QueryIntentClassifier, IntentResult
    
    if QdrantLegalStore is None:
        from src.qdrant_store import QdrantLegalStore, SearchResult
        from src.query_intent import QueryIntentClassifier, IntentResult, QueryIntent
    
    return QdrantLegalStore, QueryIntentClassifier, IntentResult


@dataclass
class LegalAnswer:
    """Structured legal answer with citations"""
    answer: str
    citations: List[Dict]  # List of cited documents
    intent: str
    retrieval_time: float
    generation_time: float
    rerank_time: float = 0.0  # Time spent on cross-encoder reranking
    used_reranker: bool = False  # Whether reranker was used
    
    def format_with_citations(self) -> str:
        """Format answer with detailed legal citation list"""
        if not self.citations:
            return self.answer
        
        citation_text = "\n\n---\n\n📚 **CĂN CỨ PHÁP LÝ:**\n\n"
        seen = set()
        
        for i, cite in enumerate(self.citations, 1):
            # Tạo key duy nhất cho mỗi điều/khoản/điểm
            key = (
                cite.get("van_ban_id", ""), 
                cite.get("dieu_so", ""),
                cite.get("khoan_so", ""),
                cite.get("diem_so", "")
            )
            if key in seen:
                continue
            seen.add(key)
            
            # Thông tin văn bản
            ten_vb = cite.get("ten_van_ban", "Văn bản pháp luật")
            loai_vb = cite.get("loai_van_ban", "")
            co_quan = cite.get("co_quan", "")
            
            # Thông tin cấu trúc pháp lý
            chuong = cite.get("chuong", "")
            ten_chuong = cite.get("ten_chuong", "")
            dieu = cite.get("dieu_so", "")
            tieu_de = cite.get("tieu_de_dieu", "")
            khoan = cite.get("khoan_so", "")
            diem = cite.get("diem_so", "")
            
            # Build citation string với format rõ ràng
            citation_parts = []
            
            # Điều luật
            if dieu:
                citation_parts.append(f"**{dieu}**")
                if tieu_de:
                    citation_parts.append(f"_{tieu_de}_")
            
            # Khoản
            if khoan and khoan != "0":
                citation_parts.append(f"Khoản {khoan}")
            
            # Điểm
            if diem:
                citation_parts.append(f"Điểm {diem})")
            
            # Chương
            chuong_info = ""
            if chuong:
                chuong_info = f"Chương {chuong}"
                if ten_chuong:
                    chuong_info += f" ({ten_chuong})"
            
            # Văn bản nguồn
            source_info = f"📖 {ten_vb}"
            if loai_vb:
                source_info = f"📖 [{loai_vb}] {ten_vb}"
            if co_quan:
                source_info += f" - _{co_quan}_"
            
            # Format final citation
            if citation_parts:
                main_cite = " - ".join(citation_parts)
                citation_text += f"**{i}.** {main_cite}\n"
                if chuong_info:
                    citation_text += f"   📂 {chuong_info}\n"
                citation_text += f"   {source_info}\n\n"
            else:
                citation_text += f"**{i}.** {source_info}\n\n"
        
        return self.answer + citation_text


class LegalEmbedding:
    """Embedding model wrapper for legal documents"""
    
    def __init__(self, model_name: str = "huyydangg/DEk21_hcmute_embedding"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load embedding model"""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        logger.info(f"[EMBED] Loading vietnam_legal_embeddings...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Use GPU for embedding (optimized)
        import os
        embedding_device = os.getenv("EMBEDDING_DEVICE", "cuda")
        self.device = embedding_device if embedding_device in ["cpu", "cuda"] else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self._loaded = True
        logger.info(f"[EMBED] Loaded on {self.device}")
    
    def encode(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Encode texts to embeddings"""
        import torch
        
        self.load()
        
        all_embeddings = []
        
        # Pre-truncate long texts to avoid CUDA errors
        max_chars = 2000  # ~500 tokens for Vietnamese
        texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,  # Reduced for safety
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.extend(embeddings.cpu().tolist())
                
            except RuntimeError as e:
                # Fallback: process one by one on CPU
                logger.warning(f"[EMBED] Batch failed, processing individually: {e}")
                for text in batch:
                    try:
                        inputs = self.tokenizer(
                            [text[:1000]],  # Further truncate
                            padding=True,
                            truncation=True,
                            max_length=256,
                            return_tensors="pt"
                        ).to("cpu")
                        
                        self.model.to("cpu")
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            emb = outputs.last_hidden_state.mean(dim=1)
                            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                        all_embeddings.append(emb[0].tolist())
                        self.model.to(self.device)
                    except Exception as e2:
                        # Return zero vector as fallback
                        logger.error(f"[EMBED] Failed to encode: {e2}")
                        all_embeddings.append([0.0] * 768)
        
        return all_embeddings
    
    def encode_query(self, query: str) -> List[float]:
        """Encode single query"""
        return self.encode([query])[0]


class LegalLLM:
    """LLM wrapper for legal answer generation"""
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        
        # Legal prompt template - improved for clear citations
        self.system_prompt = """Bạn là trợ lý pháp lý chuyên nghiệp, chuyên về pháp luật Việt Nam.

NGUYÊN TẮC TRẢ LỜI:
1. CHỈ trả lời dựa trên thông tin trong CONTEXT được cung cấp
2. LUÔN trích dẫn cụ thể: "Theo Điều X, Khoản Y, Điểm Z của [Tên văn bản]..."
3. Nếu context có nhiều điều luật liên quan, tổng hợp và so sánh chúng
4. Nếu không tìm thấy thông tin → nói rõ "Không tìm thấy quy định liên quan trong các văn bản được cung cấp"
5. Giải thích rõ ràng, dễ hiểu cho người không chuyên pháp luật
6. KHÔNG tự suy diễn hay bịa thông tin ngoài context

ĐỊNH DẠNG TRẢ LỜI:
- Bắt đầu bằng câu trả lời trực tiếp, ngắn gọn
- Trích dẫn điều luật cụ thể với format: "Điều X, Khoản Y quy định: ..."
- Nếu có nhiều mức/điều kiện (ví dụ mức phạt), liệt kê rõ ràng từng mức
- Giải thích thêm ý nghĩa thực tế nếu cần thiết
- Kết thúc với lưu ý quan trọng (nếu có)"""

    def generate(self, query: str, context: str) -> str:
        """Generate answer from query and context"""
        
        prompt = f"""CONTEXT (Các điều luật liên quan):
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

        if self.provider == "ollama":
            return self._generate_ollama(prompt)
        else:
            return self._generate_fallback(prompt)
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        import requests
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                
                # Clean thinking tags if present
                if "<think>" in answer:
                    import re
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                
                return answer.strip()
            else:
                logger.error(f"[LLM] Ollama error: {response.status_code}")
                return "Lỗi khi gọi LLM. Vui lòng thử lại."
                
        except Exception as e:
            logger.error(f"[LLM] Error: {e}")
            return f"Lỗi: {str(e)}"
    
    def _generate_fallback(self, prompt: str) -> str:
        """Fallback when Ollama not available"""
        return "LLM không khả dụng. Vui lòng kiểm tra cấu hình."


class LegalRAGPipeline:
    """
    Complete Legal RAG Pipeline
    
    Flow:
    1. Classify query intent
    2. Generate query embedding
    3. Hybrid search with intent-aware weights
    4. Format context with citations
    5. Generate answer with LLM
    """
    
    def __init__(self):
        _lazy_imports()
        
        self.qdrant_store = None
        self.intent_classifier = None
        self.embedding_model = None
        self.llm = None
        
        self._initialized = False
    
    def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        from src.qdrant_store import QdrantLegalStore, get_qdrant_store
        from src.query_intent import QueryIntentClassifier, get_intent_classifier
        
        logger.info("[PIPELINE] Initializing Legal RAG Pipeline...")
        
        # Initialize components
        self.qdrant_store = get_qdrant_store()
        self.intent_classifier = get_intent_classifier()
        self.embedding_model = LegalEmbedding()
        self.llm = LegalLLM()
        
        # Pre-load embedding model
        self.embedding_model.load()
        
        self._initialized = True
        logger.info("[PIPELINE] Initialization complete")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_conditions: Optional[Dict] = None,
        use_reranker: bool = True
    ) -> LegalAnswer:
        """
        Process legal query and return structured answer
        
        Pipeline:
        1. Classify intent -> adjust search weights
        2. Generate query embedding  
        3. Hybrid search (Dense + BM25 + RRF)
        4. Cross-Encoder Reranking (optional, improves quality)
        5. Format context with citations
        6. Generate answer with LLM
        
        Args:
            question: User's legal question
            top_k: Override number of results (optional)
            filter_conditions: Metadata filters (optional)
            use_reranker: Whether to use cross-encoder reranking
        """
        self.initialize()
        
        # 1. Classify intent
        intent_result = self.intent_classifier.classify(question)
        
        # Use intent-based top_k if not specified
        if top_k is None:
            top_k = intent_result.top_k
        
        # 2. Generate query embedding
        start_retrieval = time.time()
        query_embedding = self.embedding_model.encode_query(question)
        
        # 3. Hybrid search with RRF + optional reranking
        # Use reranker for ALL intents to improve quality
        rerank_time = 0.0
        used_rerank = False
        
        if use_reranker:
            # Advanced search with reranking for all queries
            start_rerank = time.time()
            search_results = self.qdrant_store.hybrid_search_with_rerank(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k,
                use_reranker=True,
                rerank_weight=0.6,  # Balance between RRF and reranker
                filter_conditions=filter_conditions
            )
            rerank_time = time.time() - start_rerank
            used_rerank = True
        else:
            # Standard hybrid search (RRF only)
            search_results = self.qdrant_store.hybrid_search(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k,
                dense_weight=intent_result.dense_weight,
                sparse_weight=intent_result.sparse_weight,
                filter_conditions=filter_conditions
            )
        
        retrieval_time = time.time() - start_retrieval
        logger.info(f"[PIPELINE] Retrieved {len(search_results)} documents in {retrieval_time:.2f}s")
        
        # 4. Format context
        context, citations = self._format_context(search_results)
        
        # 5. Generate answer
        start_generation = time.time()
        
        if not context:
            answer = "Không tìm thấy thông tin pháp lý liên quan đến câu hỏi của bạn."
        else:
            answer = self.llm.generate(question, context)
        
        generation_time = time.time() - start_generation
        logger.info(f"[PIPELINE] Generated answer in {generation_time:.2f}s")
        
        return LegalAnswer(
            answer=answer,
            citations=citations,
            intent=intent_result.intent.value,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            rerank_time=rerank_time,
            used_reranker=used_rerank
        )
    
    def _format_context(
        self,
        results: List
    ) -> Tuple[str, List[Dict]]:
        """Format search results into context string and citations with full legal metadata"""
        if not results:
            return "", []
        
        context_parts = []
        citations = []
        
        for i, result in enumerate(results):
            # Format each result với đầy đủ thông tin pháp lý
            meta = result.metadata
            
            # Build header với cấu trúc rõ ràng
            header_parts = [f"[{i+1}]"]
            
            # Văn bản
            if meta.get("ten_van_ban"):
                loai_vb = meta.get("loai_van_ban", "")
                if loai_vb:
                    header_parts.append(f"[{loai_vb}] {meta['ten_van_ban']}")
                else:
                    header_parts.append(meta['ten_van_ban'])
            
            # Chương
            if meta.get("chuong"):
                chuong_str = f"Chương {meta['chuong']}"
                if meta.get("ten_chuong"):
                    chuong_str += f" - {meta['ten_chuong']}"
                header_parts.append(chuong_str)
            
            # Điều
            if meta.get("dieu_so"):
                dieu_str = meta['dieu_so']
                if meta.get("tieu_de_dieu"):
                    dieu_str += f": {meta['tieu_de_dieu']}"
                header_parts.append(dieu_str)
            
            # Khoản
            khoan = meta.get("khoan_so", "")
            if khoan and khoan != "0" and khoan != "":
                header_parts.append(f"Khoản {khoan}")
            
            # Điểm  
            if meta.get("diem_so"):
                header_parts.append(f"Điểm {meta['diem_so']})")
            
            header = " | ".join(header_parts)
            context_parts.append(f"{header}\n{result.content}")
            
            # Add to citations với đầy đủ metadata + content
            citations.append({
                "content": result.content,  # FULL CONTENT for reranking display
                "van_ban_id": meta.get("van_ban_id", ""),
                "ten_van_ban": meta.get("ten_van_ban", ""),
                "loai_van_ban": meta.get("loai_van_ban", ""),
                "co_quan": meta.get("co_quan", ""),
                "chuong": meta.get("chuong", ""),
                "ten_chuong": meta.get("ten_chuong", ""),
                "dieu_so": meta.get("dieu_so", ""),
                "tieu_de_dieu": meta.get("tieu_de_dieu", ""),
                "khoan_so": meta.get("khoan_so", ""),
                "diem_so": meta.get("diem_so", ""),
                "score": result.score
            })
        
        context = "\n\n---\n\n".join(context_parts)
        return context, citations
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        self.initialize()
        
        # Use friendly name for embedding model
        embedding_display = "vietnam_legal_embeddings"
        if self.embedding_model and hasattr(self.embedding_model, 'model_name'):
            if "DEk21" in self.embedding_model.model_name or "huyydangg" in self.embedding_model.model_name:
                embedding_display = "vietnam_legal_embeddings"
            else:
                embedding_display = self.embedding_model.model_name
        
        return {
            "qdrant": self.qdrant_store.get_stats(),
            "embedding_model": embedding_display,
            "llm_model": f"{self.llm.provider}/{self.llm.model}"
        }


# Singleton with thread safety
_pipeline: Optional[LegalRAGPipeline] = None
_pipeline_lock = None

def _get_pipeline_lock():
    """Get or create pipeline lock"""
    global _pipeline_lock
    if _pipeline_lock is None:
        import threading
        _pipeline_lock = threading.Lock()
    return _pipeline_lock

def get_legal_rag_pipeline() -> LegalRAGPipeline:
    """Get singleton pipeline instance with thread safety"""
    global _pipeline
    
    # Fast path
    if _pipeline is not None:
        return _pipeline
    
    lock = _get_pipeline_lock()
    with lock:
        if _pipeline is None:
            _pipeline = LegalRAGPipeline()
    return _pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test pipeline
    pipeline = LegalRAGPipeline()
    pipeline.initialize()
    
    print("\nStats:", pipeline.get_stats())
    
    # Test query (will fail if no data indexed)
    try:
        result = pipeline.query("Điều 128 Bộ luật Hình sự quy định gì?")
        print("\nAnswer:", result.answer)
        print("Citations:", result.citations)
    except Exception as e:
        print(f"Error (expected if no data): {e}")
