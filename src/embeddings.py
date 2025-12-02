from sentence_transformers import SentenceTransformer
from functools import lru_cache
import os
import logging
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

# Force CUDA if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def get_model():
    logger.info(f"[EMBEDDINGS] Loading SentenceTransformer model: {DEFAULT_MODEL}")
    logger.info(f"[EMBEDDINGS] Device: {DEVICE}")
    
    # Load model on GPU if available
    model = SentenceTransformer(DEFAULT_MODEL, device=DEVICE)
    
    # Enable FP16 for faster inference on GPU
    if DEVICE == "cuda" and USE_FP16:
        model.half()
        logger.info("[EMBEDDINGS] Enabled FP16 mixed precision")
    
    # Log GPU info
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"[EMBEDDINGS] Using GPU: {gpu_name}")
    
    logger.info("[EMBEDDINGS] Model loaded OK")
    return model

def embed_texts(texts, batch_size=None):
    """
    Embed texts with GPU acceleration and batch processing.
    
    Args:
        texts: Single text or list of texts to embed
        batch_size: Batch size for processing (default from env)
    
    Returns:
        List of embedding vectors
    """
    model = get_model()
    
    # Use batch size from parameter or env
    if batch_size is None:
        batch_size = EMBEDDING_BATCH_SIZE
    
    # Convert single text to list
    if isinstance(texts, str):
        texts = [texts]
    
    # Log for performance monitoring
    logger.info(f"[EMBEDDINGS] Embedding {len(texts)} texts with batch_size={batch_size}")
    
    # Encode with GPU + batching + normalization
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        device=DEVICE,
        show_progress_bar=len(texts) > 10,
        convert_to_numpy=True
    )
    
    logger.info(f"[EMBEDDINGS] Completed embedding {len(texts)} texts")
    return embeddings.tolist()

class ChromaEmbeddingFunction:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or DEFAULT_MODEL

    def embed_documents(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input)

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input)

    def name(self):
        return self.model_name

