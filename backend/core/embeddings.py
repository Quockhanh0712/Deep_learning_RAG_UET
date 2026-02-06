"""
Legal Embedding Model for Vietnamese text.
Uses huyydangg/DEk21_hcmute_embedding with Vietnamese tokenization.
"""
import os
from typing import List, Optional, Union
import numpy as np

from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import EmbeddingError

logger = get_logger("embeddings")


class LegalEmbedding:
    """
    Vietnamese Legal Document Embedding.
    Uses sentence-transformers with DEk21 model optimized for Vietnamese.
    """
    
    _instance: Optional["LegalEmbedding"] = None
    
    def __new__(cls) -> "LegalEmbedding":
        """Singleton pattern to avoid reloading model."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._tokenizer = None
        self._initialized = True
        
        logger.info(f"LegalEmbedding initialized (model={self.model_name}, device={self.device})")
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Enable FP16 for GPU
            if self.device == "cuda":
                self._model = self._model.half()
            
            logger.info(f"✓ Embedding model loaded on {self.device}")
            
            # Log GPU memory if available
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"  GPU memory allocated: {allocated:.2f} GB")
                
        except Exception as e:
            raise EmbeddingError(self.model_name, str(e))
    
    def preprocess_vietnamese(self, texts: List[str]) -> List[str]:
        """
        Preprocess Vietnamese text with word segmentation.
        Uses PyVi for tokenization.
        """
        try:
            from pyvi import ViTokenizer
            
            processed = []
            for text in texts:
                # Clean and tokenize
                text = text.strip()
                if text:
                    tokenized = ViTokenizer.tokenize(text)
                    processed.append(tokenized)
                else:
                    processed.append("")
            
            return processed
            
        except ImportError:
            logger.warning("PyVi not available, skipping Vietnamese tokenization")
            return texts
        except Exception as e:
            logger.warning(f"Vietnamese preprocessing failed: {e}")
            return texts
    
    def embed_query(self, text: str, preprocess: bool = True) -> np.ndarray:
        """
        Embed a single query text.
        
        Args:
            text: Input text
            preprocess: Whether to apply Vietnamese preprocessing
            
        Returns:
            768-dimensional embedding vector
        """
        vectors = self.embed([text], preprocess=preprocess)
        return vectors[0]
    
    def embed(
        self,
        texts: Union[str, List[str]],
        preprocess: bool = True,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: Single text or list of texts
            preprocess: Whether to apply Vietnamese preprocessing
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            Array of shape (n_texts, 768)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            # Preprocess if requested
            if preprocess:
                texts = self.preprocess_vietnamese(texts)
            
            # Encode in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(self.model_name, str(e))
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (768 for DEk21)."""
        return 768
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "is_loaded": self.is_loaded(),
            "batch_size": self.batch_size
        }


def get_embedding_model() -> LegalEmbedding:
    """Get singleton embedding model instance."""
    return LegalEmbedding()
