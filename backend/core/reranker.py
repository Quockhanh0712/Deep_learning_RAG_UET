"""
Vietnamese Cross-Encoder Reranker.
Uses AITeamVN/Vietnamese_Reranker for improved ranking accuracy.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import RerankerError

logger = get_logger("reranker")


class VietnameseReranker:
    """
    Vietnamese Reranker using Cross-Encoder architecture.
    Re-scores query-document pairs for improved ranking.
    """
    
    _instance: Optional["VietnameseReranker"] = None
    
    def __new__(cls) -> "VietnameseReranker":
        """Singleton pattern to avoid reloading model."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model_name = settings.RERANKER_MODEL
        self.device = settings.RERANKER_DEVICE
        self._model = None
        self._tokenizer = None
        self._initialized = True
        
        logger.info(f"VietnameseReranker initialized (model={self.model_name})")
    
    def _load_model(self) -> None:
        """Load the reranker model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available for reranker, falling back to CPU")
                self.device = "cpu"
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model = self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode
            
            # Enable FP16 for GPU
            if self.device == "cuda":
                self._model = self._model.half()
            
            logger.info(f"✓ Reranker model loaded on {self.device}")
            
            # Log GPU memory
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"  GPU memory allocated: {allocated:.2f} GB")
                
        except Exception as e:
            raise RerankerError(self.model_name, str(e))
    
    @property
    def model(self):
        """Lazy load reranker model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Get tokenizer (loads if needed)."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of documents with 'text' field
            top_k: Return only top_k results (None for all)
            batch_size: Batch size for inference
            
        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []
        
        try:
            import torch
            
            # Extract texts
            doc_texts = [doc.get("text", "") for doc in documents]
            
            # Prepare pairs
            pairs = [[query, text] for text in doc_texts]
            
            # Score in batches
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Get relevance scores (sigmoid for binary classification)
                    scores = torch.sigmoid(outputs.logits).squeeze(-1)
                    
                    # Handle single item case
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)
                    
                    all_scores.extend(scores.cpu().numpy().tolist())
            
            # Add scores to documents
            scored_docs = []
            for doc, score in zip(documents, all_scores):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(score)
                scored_docs.append(doc_copy)
            
            # Sort by rerank score
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top_k if specified
            if top_k is not None:
                scored_docs = scored_docs[:top_k]
            
            return scored_docs
            
        except Exception as e:
            raise RerankerError(self.model_name, str(e))
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: Search query
            document: Document text
            
        Returns:
            Relevance score between 0 and 1
        """
        result = self.rerank(query, [{"text": document}])
        if result:
            return result[0].get("rerank_score", 0.0)
        return 0.0
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded()
        }


def get_reranker() -> VietnameseReranker:
    """Get singleton reranker instance."""
    return VietnameseReranker()
