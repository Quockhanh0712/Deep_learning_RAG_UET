"""
Legal RAG Pipeline.
Orchestrates the full retrieval-augmented generation flow.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import SearchError, LLMError

from backend.core.embeddings import get_embedding_model
from backend.core.qdrant_store import get_qdrant_connector
from backend.core.reranker import get_reranker
from backend.core.llm_client import (
    get_llm_client, 
    build_rag_prompt, 
    LEGAL_RAG_SYSTEM_PROMPT
)

logger = get_logger("rag_pipeline")


class LegalRAGPipeline:
    """
    Legal RAG Pipeline.
    
    Flow:
    1. Embed Query
    2. Hybrid Search (Qdrant)
    3. Rerank (Optional)
    4. Build Context
    5. Generate Response (LLM)
    6. Format with Citations
    """
    
    _instance: Optional["LegalRAGPipeline"] = None
    
    def __new__(cls) -> "LegalRAGPipeline":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._embedding = None
        self._qdrant = None
        self._reranker = None
        self._llm = None
        self._initialized = True
        
        logger.info("LegalRAGPipeline initialized")
    
    # Lazy loading of components
    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = get_embedding_model()
        return self._embedding
    
    @property
    def qdrant(self):
        if self._qdrant is None:
            self._qdrant = get_qdrant_connector()
        return self._qdrant
    
    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker
    
    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm
    
    async def query(
        self,
        question: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        search_mode: str = "hybrid",
        top_k: int = 10,
        reranker_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline.
        
        Args:
            question: User's question
            user_id: Optional user ID for user document search
            session_id: Optional session ID
            search_mode: 'legal', 'user', or 'hybrid'
            top_k: Number of documents to retrieve
            reranker_enabled: Whether to use reranker
            
        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "search_time_ms": float,
                "generate_time_ms": float,
                "reranker_used": bool
            }
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Embed query
            logger.debug(f"Embedding query: {question[:50]}...")
            query_vector = self.embedding.embed_query(question)
            
            # Step 2: Hybrid search
            search_start = datetime.utcnow()
            
            if search_mode == "hybrid":
                # Search both collections
                legal_results = self.qdrant.hybrid_search(
                    query_vector=query_vector.tolist(),
                    top_k=top_k,
                    collection="legal"
                )
                
                user_results = []
                if user_id:
                    user_results = self.qdrant.hybrid_search(
                        query_vector=query_vector.tolist(),
                        top_k=top_k // 2,
                        collection="user",
                        user_id=user_id
                    )
                
                # Merge results (legal first, then user)
                search_results = legal_results + user_results
                
            elif search_mode == "user" and user_id:
                search_results = self.qdrant.hybrid_search(
                    query_vector=query_vector.tolist(),
                    top_k=top_k,
                    collection="user",
                    user_id=user_id
                )
            else:
                # Default to legal
                search_results = self.qdrant.hybrid_search(
                    query_vector=query_vector.tolist(),
                    top_k=top_k,
                    collection="legal"
                )
            
            search_time = (datetime.utcnow() - search_start).total_seconds() * 1000
            
            if not search_results:
                return {
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu pháp luật.",
                    "sources": [],
                    "search_time_ms": search_time,
                    "generate_time_ms": 0,
                    "reranker_used": False
                }
            
            # Step 3: Rerank (optional)
            reranker_used = False
            if reranker_enabled and settings.USE_RERANKER:
                try:
                    logger.debug("Reranking results...")
                    search_results = self.reranker.rerank(
                        query=question,
                        documents=search_results,
                        top_k=min(top_k, len(search_results))
                    )
                    reranker_used = True
                except Exception as e:
                    logger.warning(f"Reranking failed, using original order: {e}")
            
            # Step 4: Build context
            context = self._build_context(search_results[:top_k])
            
            # [DEBUG] Log context for verification
            logger.info(f"[DEBUG] RAG Context ({len(context)} chars):\n{context[:500]}...")
            if not context.strip():
                logger.warning("[DEBUG] Empty context built from results!")
            
            # Step 5: Generate response
            generate_start = datetime.utcnow()
            
            prompt = build_rag_prompt(question, context)
            answer = await self.llm.generate_async(
                prompt=prompt,
                system_prompt=LEGAL_RAG_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            generate_time = (datetime.utcnow() - generate_start).total_seconds() * 1000
            
            # Step 6: Format sources
            formatted_sources = self._format_sources(search_results[:top_k])
            
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"RAG query completed in {total_time:.0f}ms")
            
            return {
                "answer": answer,
                "sources": formatted_sources,
                "search_time_ms": search_time,
                "generate_time_ms": generate_time,
                "total_time_ms": total_time,
                "reranker_used": reranker_used,
                "num_sources": len(formatted_sources)
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            raise
    
    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            dieu = result.get("dieu_number", "")
            khoan = result.get("khoan_number", "")
            
            # Format citation
            citation = ""
            if dieu:
                citation = f"Điều {dieu}"
                if khoan:
                    citation += f", Khoản {khoan}"
            
            if citation:
                context_parts.append(f"[{i}] {citation}:\n{text}")
            else:
                context_parts.append(f"[{i}] {text}")
        
        return "\n\n".join(context_parts)
    
    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for response."""
        formatted = []
        
        for rank, result in enumerate(results, 1):
            formatted.append({
                "text": result.get("text", "")[:500],  # Truncate for response
                "source_type": result.get("source_type", "legal"),
                "dieu_number": result.get("dieu_number"),
                "khoan_number": result.get("khoan_number"),
                "file_name": result.get("file_name"),
                "score": result.get("score", 0.0),
                "rerank_score": result.get("rerank_score"),
                "rank": rank
            })
        
        return formatted
    
    async def search_only(
        self,
        query: str,
        user_id: Optional[str] = None,
        search_mode: str = "legal",
        top_k: int = 10,
        reranker_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Perform search without LLM generation.
        For /api/search endpoint.
        """
        start_time = datetime.utcnow()
        
        # Embed query
        query_vector = self.embedding.embed_query(query)
        
        # Search
        collection = "user" if search_mode == "user" else "legal"
        results = self.qdrant.hybrid_search(
            query_vector=query_vector.tolist(),
            top_k=top_k,
            collection=collection,
            user_id=user_id if search_mode == "user" else None
        )
        
        # Rerank if enabled
        if reranker_enabled and results:
            try:
                results = self.reranker.rerank(query, results, top_k=top_k)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
        
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "results": self._format_sources(results),
            "total": len(results),
            "query": query,
            "search_mode": search_mode,
            "search_time_ms": search_time
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline component status."""
        return {
            "embedding_loaded": self._embedding is not None and self._embedding.is_loaded(),
            "qdrant_connected": self._qdrant is not None and self._qdrant.check_connection(),
            "reranker_loaded": self._reranker is not None and self._reranker.is_loaded(),
            "llm_available": self._llm is not None and self._llm.check_available()
        }


def get_rag_pipeline() -> LegalRAGPipeline:
    """Get singleton RAG pipeline instance."""
    return LegalRAGPipeline()
