from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float
import uuid
from datetime import datetime
import os

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite+aiosqlite:///./course_assistant.db"
)

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for models
Base = declarative_base()

class Course(Base):
    __tablename__ = "courses"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    code = Column(String(50), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Metadata
    file_count = Column(Integer, default=0)
    module_count = Column(Integer, default=0)
    last_sync = Column(DateTime)
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "fileCount": self.file_count,
            "moduleCount": self.module_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None
        }

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    course_id = Column(String(36), ForeignKey("courses.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500))
    file_type = Column(String(50))
    file_size = Column(Integer)
    
    # Content
    raw_text = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    # Metadata
    doc_metadata = Column(JSON)
    chunk_count = Column(Integer, default=0)
    
    # Status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    course_id = Column(String(36), ForeignKey("courses.id"), nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Metadata
    chunk_metadata = Column(JSON)
    chunk_type = Column(String(50))  # semantic, slide, section, etc.
    
    # Vector storage reference
    vector_id = Column(String(255))  # ID in Qdrant
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    course_id = Column(String(36), ForeignKey("courses.id"), nullable=False)
    
    # Session info
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    course_id = Column(String(36), ForeignKey("courses.id"), nullable=False)
    
    # Message content
    content = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant
    
    # Context
    sources = Column(JSON)  # List of source documents/chunks
    msg_metadata = Column(JSON)
    confidence = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "sources": self.sources,
            "confidence": self.confidence,
            "timestamp": self.created_at.isoformat() if self.created_at else None
        }

class ProcessingQueue(Base):
    __tablename__ = "processing_queue"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    course_id = Column(String(36), ForeignKey("courses.id"), nullable=False)
    
    # File info
    file_path = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50))
    
    # Status
    status = Column(String(50), default="queued")  # queued, processing, completed, failed
    priority = Column(Integer, default=0)
    
    # Timestamps
    queued_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

# Dependency to get database session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() 