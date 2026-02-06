"""
Configuration settings for Legal RAG Backend.
Loads environment variables from .env file.
"""
import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = DATA_DIR / "database"

# Ensure directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{DATABASE_DIR}/legal_rag.db"
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_LEGAL_COLLECTION: str = "legal_rag_hybrid"
    QDRANT_USER_COLLECTION: str = "user_docs_private"
    
    # Embedding
    EMBEDDING_MODEL: str = "huyydangg/DEk21_hcmute_embedding"
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Reranker
    RERANKER_MODEL: str = "AITeamVN/Vietnamese_Reranker"
    RERANKER_DEVICE: str = "cuda"
    USE_RERANKER: bool = True
    
    # LLM (Ollama)
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_NUM_GPU: int = 999  # Full GPU
    
    # RAG Settings
    TOP_K: int = 10
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
