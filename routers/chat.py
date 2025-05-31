from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime
import logging

from models.database import get_db, Course, ChatSession, ChatMessage
from services.query import QueryService
from services.embedding import EmbeddingService
import sys

router = APIRouter()
logger = logging.getLogger(__name__)

# Import services from main app instead of creating new instances
query_service = None
embedding_service = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict]
    confidence: float
    timestamp: str

class MessageResponse(BaseModel):
    id: str
    content: str
    role: str
    sources: List[str] = []
    timestamp: str

@router.on_event("startup")
async def initialize_services():
    """Initialize services on startup"""
    try:
        global query_service, embedding_service
        # Get services from main app
        from main import query_service as main_query_service, embedding_service as main_embedding_service
        
        query_service = main_query_service
        embedding_service = main_embedding_service
        
        # Ensure services are initialized
        if not embedding_service or not query_service:
            logger.error("Failed to get services from main app")
            raise ValueError("Services not available from main app")
            
        logger.info("Chat services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat services: {e}")

@router.post("/{course_id}", response_model=ChatResponse)
async def chat_with_course(
    course_id: str,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """Chat with course materials"""
    try:
        # Convert string to UUID if needed
        if isinstance(course_id, str) and '-' in course_id:
            course_uuid = course_id
        else:
            course_uuid = str(course_id)
            
        # Validate course exists
        result = await db.execute(
            select(Course).where(Course.id == course_uuid)
        )
        course = result.scalar_one_or_none()
        
        if not course:
            raise HTTPException(status_code=404, detail=f"Course not found: {course_id}")
        
        # Process query
        query_result = await query_service.process_query(
            course_id=course_uuid,
            query=request.message,
            session_id=request.session_id,
            chat_history=[],
            course_name=course.name,
            db_session=db
        )
        
        # Save chat message
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create session if needed
        session = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        if not session.scalar_one_or_none():
            new_session = ChatSession(
                id=session_id,
                course_id=course_uuid,
                title=request.message[:50]
            )
            db.add(new_session)
            await db.flush()  # Make sure the session is flushed to the database
        
        # Save messages
        user_msg = ChatMessage(
            session_id=session_id,
            course_id=course_uuid,
            content=request.message,
            role="user",
            confidence=1.0
        )
        db.add(user_msg)
        
        assistant_msg = ChatMessage(
            session_id=session_id,
            course_id=course_uuid,
            content=query_result["response"],
            role="assistant",
            sources=query_result.get("sources", []),
            confidence=query_result.get("confidence", 0.0)
        )
        db.add(assistant_msg)
        
        await db.commit()
        
        return ChatResponse(
            response=query_result["response"],
            session_id=session_id,
            sources=query_result.get("sources", []),
            confidence=query_result.get("confidence", 0.0),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/{course_id}/sessions")
async def get_chat_sessions(
    course_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all chat sessions for a course"""
    try:
        # Validate course exists
        result = await db.execute(select(Course).where(Course.id == course_id))
        course = result.scalar_one_or_none()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Get sessions
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.course_id == course_id)
            .order_by(ChatSession.created_at.desc())
        )
        sessions = result.scalars().all()
        
        return [
            {
                "id": session.id,
                "title": session.title or "Chat Session",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "message_count": len(session.messages) if session.messages else 0
            }
            for session in sessions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/chat/{course_id}/sessions/{session_id}")
async def get_chat_session(
    course_id: str,
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific chat session with messages"""
    try:
        # Get session with messages
        result = await db.execute(
            select(ChatSession)
            .where(
                ChatSession.id == session_id,
                ChatSession.course_id == course_id
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Get messages
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        messages = result.scalars().all()
        
        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": msg.sources,
                    "confidence": msg.confidence,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def get_chat_history(session_id: str, db: AsyncSession) -> List[Dict]:
    """Get recent chat history for context"""
    try:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(10)  # Last 10 messages
        )
        messages = result.scalars().all()
        
        # Reverse to get chronological order
        messages = list(reversed(messages))
        
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

async def save_chat_message(
    session_id: str,
    course_id: str,
    user_message: str,
    assistant_response: str,
    sources: List[Dict],
    confidence: float,
    db: AsyncSession
):
    """Save chat messages to database"""
    try:
        # Get or create session
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            session = ChatSession(
                id=session_id,
                course_id=course_id,
                title=user_message[:50] + "..." if len(user_message) > 50 else user_message
            )
            db.add(session)
        
        # Save user message
        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=user_message
        )
        db.add(user_msg)
        
        # Save assistant response
        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
            sources=sources,
            confidence=confidence
        )
        db.add(assistant_msg)
        
        await db.commit()
        
    except Exception as e:
        logger.error(f"Error saving chat messages: {e}")
        await db.rollback()
        raise

@router.get("/{course_id}/history", response_model=List[MessageResponse])
async def get_chat_history(
    course_id: str,
    session_id: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get chat history for a course or specific session"""
    try:
        # Verify course exists
        course_result = await db.execute(
            select(Course).where(Course.id == course_id, Course.is_active == True)
        )
        
        if not course_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        # Get chat history
        messages = await query_service.get_chat_history(
            db=db,
            course_id=course_id,
            session_id=session_id,
            limit=limit
        )
        
        return [
            MessageResponse(
                id=str(msg.id),
                content=msg.content,
                role=msg.role,
                sources=msg.sources or [],
                timestamp=msg.created_at.isoformat() if msg.created_at else ""
            )
            for msg in messages
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )

@router.get("/{course_id}/sessions")
async def get_chat_sessions(course_id: str, db: AsyncSession = Depends(get_db)):
    """Get all chat sessions for a course"""
    try:
        # Verify course exists
        course_result = await db.execute(
            select(Course).where(Course.id == course_id, Course.is_active == True)
        )
        
        if not course_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        # Get sessions
        sessions_result = await db.execute(
            select(ChatSession)
            .where(ChatSession.course_id == course_id, ChatSession.is_active == True)
            .order_by(ChatSession.updated_at.desc())
        )
        
        sessions = sessions_result.scalars().all()
        
        return [
            {
                "id": str(session.id),
                "title": session.title,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            }
            for session in sessions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )

@router.delete("/{course_id}/sessions/{session_id}")
async def delete_chat_session(
    course_id: str,
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a chat session"""
    try:
        # Get session
        session_result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.course_id == course_id,
                ChatSession.is_active == True
            )
        )
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Soft delete session
        session.is_active = False
        await db.commit()
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )

@router.put("/{course_id}/sessions/{session_id}/title")
async def update_session_title(
    course_id: str,
    session_id: str,
    title_data: dict,
    db: AsyncSession = Depends(get_db)
):
    """Update chat session title"""
    try:
        new_title = title_data.get("title", "").strip()
        if not new_title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title cannot be empty"
            )
        
        # Get session
        session_result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.course_id == course_id,
                ChatSession.is_active == True
            )
        )
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Update title
        session.title = new_title
        await db.commit()
        
        return {"message": "Session title updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session title: {str(e)}"
        )

@router.get("/{course_id}/search")
async def search_course_content(
    course_id: str,
    q: str,
    limit: int = 10,
    threshold: float = 0.6,
    db: AsyncSession = Depends(get_db)
):
    """Search course content using semantic similarity"""
    try:
        # Verify course exists
        course_result = await db.execute(
            select(Course).where(Course.id == course_id, Course.is_active == True)
        )
        
        if not course_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        # Search using embedding service
        results = await query_service.embedding_service.search_similar(
            query=q,
            course_id=course_id,
            limit=limit,
            score_threshold=threshold
        )
        
        return {
            "query": q,
            "results": results,
            "total_found": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        ) 