import os
import logging
from pathlib import Path
from typing import List, Dict
import pdfplumber
from sqlalchemy import select, update
from models.database import AsyncSessionLocal, Document, DocumentChunk, Course

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.embedding_service = None
        
    async def initialize(self, embedding_service):
        """Initialize with embedding service"""
        self.embedding_service = embedding_service
        logger.info("Ingestion service initialized")
    
    async def process_file(self, course_id: str, file_path: str, filename: str) -> bool:
        """Process a single file"""
        try:
            async with AsyncSessionLocal() as session:
                # Create document record
                document = Document(
                    course_id=course_id,
                    filename=filename,
                    original_path=file_path,
                    file_type=Path(filename).suffix.lower(),
                    file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    status="processing"
                )
                
                session.add(document)
                await session.flush()
                
                # Extract text (only PDF for now)
                text = ""
                if filename.lower().endswith('.pdf'):
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n\n"
                    except Exception as e:
                        logger.error(f"PDF extraction error: {e}")
                        text = ""
                
                if not text or len(text.strip()) < 10:
                    document.status = "failed"
                    document.error_message = "No text extracted"
                    await session.commit()
                    return False
                
                document.raw_text = text
                
                # Simple chunking
                chunks = self.simple_chunk(text, chunk_size=1000, overlap=200)
                
                # Store chunks
                chunk_records = []
                for i, chunk_text in enumerate(chunks):
                    chunk_record = DocumentChunk(
                        document_id=document.id,
                        course_id=course_id,
                        content=chunk_text,
                        chunk_index=i,
                        chunk_metadata={"filename": filename},
                        chunk_type="semantic"
                    )
                    session.add(chunk_record)
                    chunk_records.append(chunk_record)

                await session.flush()
                
                # Store embeddings
                chunk_dicts = [{"content": c, "metadata": {"filename": filename}} for c in chunks]
                vector_ids = await self.embedding_service.store_embeddings(
                    chunk_dicts, 
                    course_id, 
                    str(document.id),
                    db_session=session
                )

                # Update chunk records with their vector IDs
                for chunk_record, vector_id in zip(chunk_records, vector_ids):
                    chunk_record.vector_id = vector_id

                await session.flush()
                
                # Update document status
                document.status = "completed"
                document.chunk_count = len(chunks)
                
                # Update course file count
                await session.execute(
                    update(Course)
                    .where(Course.id == course_id)
                    .values(file_count=Course.file_count + 1)
                )
                
                await session.commit()
                logger.info(f"Processed {filename} with {len(chunks)} chunks")
                return True
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return False
    
    def simple_chunk(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple text chunking"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if c]