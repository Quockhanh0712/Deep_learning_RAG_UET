"""
Pydantic Schemas for API Request/Response validation.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ==================== Chat Schemas ====================

class ChatRequest(BaseModel):
    """Request schema for POST /api/chat."""
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    user_id: str = Field(..., min_length=1, max_length=64, description="User identifier")
    session_id: Optional[str] = Field(None, max_length=64, description="Session ID (auto-generated if not provided)")
    search_mode: str = Field("hybrid", pattern="^(legal|user|hybrid)$", description="Search mode")
    reranker_enabled: bool = Field(True, description="Enable Vietnamese reranker")


class SourceInfo(BaseModel):
    """Source citation in response."""
    text: str
    source_type: str = "legal"
    dieu_number: Optional[str] = None
    khoan_number: Optional[str] = None
    file_name: Optional[str] = None
    score: float = 0.0
    rank: int = 0


class QualityMetrics(BaseModel):
    """Quality metrics for answer."""
    bertscore_f1: float
    hallucination_score: Optional[float] = None
    factuality_score: Optional[float] = None
    context_relevance: Optional[float] = None
    grade: str = "N/A"
    feedback: Optional[str] = None


class ChatResponse(BaseModel):
    """Response schema for POST /api/chat."""
    answer: str
    sources: List[SourceInfo] = []
    metrics: Optional[QualityMetrics] = None
    message_id: int
    session_id: str


# ==================== Search Schemas ====================

class SearchRequest(BaseModel):
    """Request schema for POST /api/search."""
    query: str = Field(..., min_length=1, max_length=1024, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results")
    user_id: Optional[str] = Field(None, description="User ID for user document search")
    search_mode: str = Field("legal", pattern="^(legal|user|hybrid)$", description="Search mode")
    reranker_enabled: bool = Field(False, description="Enable Vietnamese reranker")


class SearchResult(BaseModel):
    """Single search result."""
    text: str
    score: float
    dieu_number: Optional[str] = None
    khoan_number: Optional[str] = None
    file_name: Optional[str] = None
    source_type: str = "legal"
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Response schema for POST /api/search."""
    results: List[SearchResult]
    total: int
    query: str
    search_mode: str


# ==================== Status Schemas ====================

class ComponentStatus(BaseModel):
    """Status of a single component."""
    status: str  # "connected", "available", "loaded", "error"
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class SystemStatus(BaseModel):
    """Response schema for GET /api/status."""
    qdrant: ComponentStatus
    ollama: ComponentStatus
    embedding: ComponentStatus
    database: ComponentStatus
    gpu_available: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== History Schemas ====================

class MessageInfo(BaseModel):
    """Message info in history."""
    id: int
    role: str
    content: str
    created_at: datetime
    metrics: Optional[QualityMetrics] = None


class SessionInfo(BaseModel):
    """Session info in history."""
    session_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class HistoryResponse(BaseModel):
    """Response schema for GET /api/history."""
    sessions: List[SessionInfo]
    total: int


class SessionDetailResponse(BaseModel):
    """Response schema for GET /api/history/{session_id}."""
    session: SessionInfo
    messages: List[MessageInfo]


# ==================== Settings Schemas ====================

class RerankerSetting(BaseModel):
    """Reranker toggle setting."""
    enabled: bool


# ==================== General Schemas ====================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Dict[str, Any] = {}
