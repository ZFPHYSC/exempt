from typing import List, Dict, Optional
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.database import Course, ChatSession, ChatMessage, DocumentChunk
from services.embedding import EmbeddingService
from services.ai import ai_service

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self):
        self.embedding_service = None
        
    async def initialize(self, embedding_service: EmbeddingService):
        """Initialize with embedding service"""
        self.embedding_service = embedding_service
        
    async def process_query(
        self, 
        course_id: str, 
        query: str, 
        session_id: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        course_name: str = "your course",
        db_session: Optional[AsyncSession] = None
    ) -> Dict:
        """Process a user query and return response with sources"""
        try:
            # Find relevant chunks using vector search
            relevant_chunks = await self.embedding_service.search_similar(
                query=query,
                course_id=course_id,
                limit=8,
                score_threshold=0.6,
                db_session=db_session
            )
            
            if not relevant_chunks:
                return {
                    "response": "I don't have enough information about this topic in your course materials. Could you try rephrasing your question or ask about something else?",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context from chunks
            context = self._prepare_context(relevant_chunks)
            
            # Generate response using AI service
            response = await self._generate_ai_response(
                query, context, chat_history, course_name
            )
            
            # Extract source information
            sources = self._extract_sources(relevant_chunks)
            
            return {
                "response": response,
                "sources": sources,
                "confidence": self._calculate_confidence(relevant_chunks),
                "chunks_used": len(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "sources": [],
                "confidence": 0.0
            }
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context string from relevant chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['content']
            metadata = chunk.get('metadata', {})
            
            # Add source information to context
            source_info = ""
            if metadata.get('filename'):
                source_info = f"[From {metadata['filename']}]"
            
            # Add section info if available
            if metadata.get('section'):
                source_info += f" [Section: {metadata['section']}]"
            
            context_parts.append(f"{source_info} {chunk_text}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_ai_response(
        self, 
        query: str, 
        context: str, 
        chat_history: Optional[List[Dict]] = None,
        course_name: str = "your course"
    ) -> str:
        """Generate response using AI service"""
        try:
            # Create structured prompt
            messages = await ai_service.create_course_assistant_prompt(
                query=query,
                context=context,
                chat_history=chat_history,
                course_name=course_name
            )
            
            # Generate response
            response = await ai_service.generate_response(
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            # Fallback to simple response if AI fails
            return await self._simple_response_generation(query, context)
    
    async def _simple_response_generation(self, query: str, context: str) -> str:
        """Fallback simple response generation without external LLM"""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Extract relevant sentences from context
        sentences = context.split('.')
        relevant_sentences = []
        
        query_words = set(query_lower.split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            # Simple relevance scoring
            overlap = len(query_words.intersection(sentence_words))
            if overlap >= 2 or any(word in sentence.lower() for word in query_words if len(word) > 4):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Take the most relevant sentences
            response = '. '.join(relevant_sentences[:3])
            if len(response) > 500:
                response = response[:500] + "..."
            return response + "\n\nThis information is based on your course materials. Would you like me to elaborate on any specific aspect?"
        else:
            return "I found some relevant course materials, but I need more specific information to answer your question accurately. Could you please rephrase your question or provide more context?"
    
    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        """Extract source filenames from chunks"""
        sources = set()
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if metadata.get('filename'):
                sources.add(metadata['filename'])
        
        return list(sources)
    
    def _calculate_confidence(self, chunks: List[Dict]) -> float:
        """Calculate confidence score based on chunk relevance"""
        if not chunks:
            return 0.0
        
        # Average the scores
        total_score = sum(chunk.get('score', 0.0) for chunk in chunks)
        avg_score = total_score / len(chunks)
        
        # Normalize to 0-1 range
        return min(avg_score, 1.0)
    
    async def save_chat_message(
        self,
        db: AsyncSession,
        course_id: str,
        session_id: str,
        content: str,
        role: str,
        sources: Optional[List[str]] = None
    ) -> ChatMessage:
        """Save a chat message to the database"""
        try:
            message = ChatMessage(
                session_id=session_id,
                course_id=course_id,
                content=content,
                role=role,
                sources=sources or []
            )
            
            db.add(message)
            await db.commit()
            await db.refresh(message)
            
            return message
            
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
            await db.rollback()
            raise
    
    async def get_or_create_session(
        self,
        db: AsyncSession,
        course_id: str,
        session_id: Optional[str] = None
    ) -> ChatSession:
        """Get existing session or create new one"""
        try:
            if session_id:
                # Try to get existing session
                result = await db.execute(
                    select(ChatSession).where(
                        ChatSession.id == session_id,
                        ChatSession.course_id == course_id,
                        ChatSession.is_active == True
                    )
                )
                session = result.scalar_one_or_none()
                
                if session:
                    return session
            
            # Create new session
            session = ChatSession(
                course_id=course_id,
                title=f"Chat Session"
            )
            
            db.add(session)
            await db.commit()
            await db.refresh(session)
            
            return session
            
        except Exception as e:
            logger.error(f"Error getting/creating session: {e}")
            await db.rollback()
            raise
    
    async def get_chat_history(
        self,
        db: AsyncSession,
        course_id: str,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ChatMessage]:
        """Get chat history for a course or session"""
        try:
            query = select(ChatMessage).where(
                ChatMessage.course_id == course_id
            )
            
            if session_id:
                query = query.where(ChatMessage.session_id == session_id)
            
            query = query.order_by(ChatMessage.created_at.desc()).limit(limit)
            
            result = await db.execute(query)
            messages = result.scalars().all()
            
            # Return in chronological order
            return list(reversed(messages))
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

# Global instance
query_service = QueryService() 