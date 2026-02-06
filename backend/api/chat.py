"""
Chat API Endpoint - RAG Generation.
"""
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.db.database import get_db
from backend.db.models import ChatSession, Message, MessageSource, AnswerMetrics
from backend.db.schemas import (
    ChatRequest, ChatResponse, 
    SourceInfo, QualityMetrics
)
from backend.core.rag_pipeline import get_rag_pipeline
from backend.utils.logger import get_logger
from backend.utils.exceptions import LLMError

logger = get_logger("api.chat")

router = APIRouter()


async def get_or_create_session(
    db: AsyncSession, 
    user_id: str, 
    session_id: Optional[str]
) -> ChatSession:
    """Get existing session or create new one."""
    if session_id:
        # Try to find existing session
        result = await db.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if session:
            return session
    
    # Create new session
    new_session_id = session_id or str(uuid.uuid4())
    session = ChatSession(
        user_id=user_id,
        session_id=new_session_id,
        title=None,  # Will be set from first message
        created_at=datetime.utcnow()
    )
    db.add(session)
    await db.flush()  # Get the ID
    
    return session


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with Legal RAG system.
    
    Flow:
    1. Get or create chat session
    2. Save user message
    3. Execute RAG pipeline (search + rerank + generate)
    4. Save assistant response with sources and metrics
    5. Return response
    
    Args:
        request: ChatRequest with message, user_id, search_mode, etc.
        
    Returns:
        ChatResponse with answer, sources, and quality metrics
    """
    logger.info(f"Chat request from user={request.user_id}, mode={request.search_mode}")
    
    try:
        # 1. Get or create session
        session = await get_or_create_session(db, request.user_id, request.session_id)
        
        # Set title from first message if not set
        if not session.title:
            session.title = request.message[:50] + "..." if len(request.message) > 50 else request.message
        
        # 2. Save user message
        user_message = Message(
            session_id=session.id,
            role="user",
            content=request.message,
            search_mode=request.search_mode,
            reranker_enabled=request.reranker_enabled,
            created_at=datetime.utcnow()
        )
        db.add(user_message)
        await db.flush()
        
        # 3. Execute RAG pipeline
        pipeline = get_rag_pipeline()
        
        result = await pipeline.query(
            question=request.message,
            user_id=request.user_id,
            session_id=session.session_id,
            search_mode=request.search_mode,
            top_k=10,
            reranker_enabled=request.reranker_enabled
        )
        
        # 4. Save assistant message
        assistant_message = Message(
            session_id=session.id,
            role="assistant",
            content=result["answer"],
            search_mode=request.search_mode,
            reranker_enabled=result.get("reranker_used", False),
            created_at=datetime.utcnow()
        )
        db.add(assistant_message)
        await db.flush()
        
        # Save sources
        sources = []
        for source_data in result.get("sources", []):
            source = MessageSource(
                message_id=assistant_message.id,
                source_text=source_data.get("text", ""),
                source_type=source_data.get("source_type", "legal"),
                dieu_number=source_data.get("dieu_number"),
                khoan_number=source_data.get("khoan_number"),
                file_name=source_data.get("file_name"),
                score=source_data.get("score", 0.0),
                rank=source_data.get("rank", 0)
            )
            db.add(source)
            
            sources.append(SourceInfo(
                text=source_data.get("text", "")[:500],
                source_type=source_data.get("source_type", "legal"),
                dieu_number=source_data.get("dieu_number"),
                khoan_number=source_data.get("khoan_number"),
                file_name=source_data.get("file_name"),
                score=source_data.get("score", 0.0),
                rank=source_data.get("rank", 0)
            ))
        
        # Save metrics (placeholder for now - quality monitor integration)
        metrics = AnswerMetrics(
            message_id=assistant_message.id,
            bertscore_f1=0.0,  # Will be computed by quality monitor
            grade="N/A",
            num_sources=len(sources),
            created_at=datetime.utcnow()
        )
        db.add(metrics)
        
        # Update session timestamp
        session.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"Chat completed: {len(sources)} sources, {result.get('total_time_ms', 0):.0f}ms")
        
        # 5. Return response
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            metrics=QualityMetrics(
                bertscore_f1=0.0,
                grade="N/A"
            ),
            message_id=assistant_message.id,
            session_id=session.session_id
        )
        
    except LLMError as e:
        logger.error(f"LLM error: {e.message}")
        raise HTTPException(status_code=503, detail=e.to_dict())
    except Exception as e:
        logger.exception(f"Chat error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail={"error": "ChatFailed", "message": str(e)})


@router.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get chat history for a session."""
    try:
        # Get session
        result = await db.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get messages
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at)
        )
        messages = result.scalars().all()
        
        return {
            "session_id": session.session_id,
            "title": session.title,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions/{user_id}")
async def list_sessions(
    user_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """List chat sessions for a user."""
    try:
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        )
        sessions = result.scalars().all()
        
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "title": s.title,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat()
                }
                for s in sessions
            ],
            "total": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a chat session and all its messages."""
    try:
        # Get session
        result = await db.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete messages (cascade will handle sources and metrics)
        await db.execute(
            Message.__table__.delete().where(Message.session_id == session.id)
        )
        
        # Delete session
        await db.delete(session)
        await db.commit()
        
        logger.info(f"Deleted session: {session_id}")
        return {"message": "Session deleted", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

