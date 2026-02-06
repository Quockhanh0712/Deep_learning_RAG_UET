"""
SQLAlchemy Models for Legal RAG Backend.
Based on architecture doc Section 9 - Database Schema.
"""
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Integer, String, Text, Float, Boolean, 
    DateTime, ForeignKey, Index
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


class ChatSession(Base):
    """Chat session model - groups messages by user session."""
    
    __tablename__ = "chat_sessions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, session_id='{self.session_id}')>"


class Message(Base):
    """Message model - stores user and assistant messages."""
    
    __tablename__ = "messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    search_mode: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # 'legal', 'user', 'hybrid'
    reranker_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")
    sources: Mapped[List["MessageSource"]] = relationship(
        "MessageSource", back_populates="message", cascade="all, delete-orphan"
    )
    metrics: Mapped[Optional["AnswerMetrics"]] = relationship(
        "AnswerMetrics", back_populates="message", uselist=False, cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_message_session", "session_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}')>"


class MessageSource(Base):
    """Source citations for assistant messages."""
    
    __tablename__ = "message_sources"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # 'legal' or 'user_document'
    dieu_number: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    khoan_number: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    file_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="sources")
    
    def __repr__(self) -> str:
        return f"<MessageSource(id={self.id}, dieu='{self.dieu_number}')>"


class AnswerMetrics(Base):
    """Quality metrics for assistant answers."""
    
    __tablename__ = "answer_metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True)
    bertscore_f1: Mapped[float] = mapped_column(Float, nullable=False)
    hallucination_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    factuality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grade: Mapped[Optional[str]] = mapped_column(String(2), nullable=True, index=True)  # A, B, C, D, F
    feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    num_sources: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="metrics")
    
    def __repr__(self) -> str:
        return f"<AnswerMetrics(id={self.id}, grade='{self.grade}')>"


class UserDocument(Base):
    """User uploaded documents metadata."""
    
    __tablename__ = "user_documents"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    file_name: Mapped[str] = mapped_column(String(256), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_chunks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    upload_status: Mapped[str] = mapped_column(String(32), default="completed")  # 'completed', 'failed'
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<UserDocument(id={self.id}, file_name='{self.file_name}')>"
