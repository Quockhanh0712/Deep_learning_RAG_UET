"""
Custom exceptions for Legal RAG Backend.
Provides clear error hierarchy for different failure scenarios.
"""
from typing import Optional, Dict, Any


class LegalRAGException(Exception):
    """Base exception for all Legal RAG errors."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


# ==================== Connection Errors ====================

class ConnectionError(LegalRAGException):
    """Base class for connection-related errors."""
    pass


class QdrantConnectionError(ConnectionError):
    """Failed to connect to Qdrant vector database."""
    
    def __init__(self, host: str, port: int, original_error: Optional[str] = None):
        message = f"Failed to connect to Qdrant at {host}:{port}"
        details = {"host": host, "port": port}
        if original_error:
            details["original_error"] = original_error
        super().__init__(message, details)


class OllamaConnectionError(ConnectionError):
    """Failed to connect to Ollama LLM server."""
    
    def __init__(self, host: str, original_error: Optional[str] = None):
        message = f"Failed to connect to Ollama at {host}"
        details = {"host": host}
        if original_error:
            details["original_error"] = original_error
        super().__init__(message, details)


class DatabaseConnectionError(ConnectionError):
    """Failed to connect to SQLite database."""
    
    def __init__(self, db_path: str, original_error: Optional[str] = None):
        message = f"Failed to connect to database: {db_path}"
        details = {"db_path": db_path}
        if original_error:
            details["original_error"] = original_error
        super().__init__(message, details)


# ==================== Model Errors ====================

class ModelError(LegalRAGException):
    """Base class for ML model errors."""
    pass


class EmbeddingError(ModelError):
    """Failed to generate embeddings."""
    
    def __init__(self, model_name: str, original_error: Optional[str] = None):
        message = f"Embedding generation failed with model: {model_name}"
        details = {"model": model_name}
        if original_error:
            details["original_error"] = original_error
        super().__init__(message, details)


class RerankerError(ModelError):
    """Failed during reranking."""
    
    def __init__(self, model_name: str, original_error: Optional[str] = None):
        message = f"Reranking failed with model: {model_name}"
        details = {"model": model_name}
        if original_error:
            details["original_error"] = original_error
        super().__init__(message, details)


class LLMError(ModelError):
    """Failed to generate response from LLM."""
    
    def __init__(self, model_name: str, original_error: Optional[str] = None):
        message = f"LLM generation failed with model: {model_name}"
        details = {"model": model_name}
        if original_error:
            details["original_error"] = original_error
        super().__init__(message, details)


# ==================== Request Errors ====================

class ValidationError(LegalRAGException):
    """Input validation failed."""
    
    def __init__(self, field: str, reason: str):
        message = f"Validation failed for '{field}': {reason}"
        details = {"field": field, "reason": reason}
        super().__init__(message, details)


class SearchError(LegalRAGException):
    """Search operation failed."""
    
    def __init__(self, query: str, original_error: Optional[str] = None):
        message = f"Search failed for query: '{query[:50]}...'"
        details = {"query_preview": query[:100]}
        if original_error:
            details["original_error"] = original_error
        super().__init__(message, details)


class DocumentNotFoundError(LegalRAGException):
    """Requested document not found."""
    
    def __init__(self, doc_id: str):
        message = f"Document not found: {doc_id}"
        details = {"doc_id": doc_id}
        super().__init__(message, details)
