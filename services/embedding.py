import asyncio
from typing import List, Dict, Optional
import numpy as np
import os
import logging
import uuid
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from .file_vector_store import FileVectorStore
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.vector_store = None
        self.openai_client = None
        self.provider = os.getenv("EMBEDDING_MODEL_PROVIDER", "local")
        self.model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.vector_size = int(os.getenv("VECTOR_DIMENSION", "384"))
        
        # Set vector size based on model name for OpenAI embeddings
        if self.provider == "openai":
            if self.model_name == "text-embedding-3-small":
                self.vector_size = 1536
            elif self.model_name == "text-embedding-3-large":
                self.vector_size = 3072
            elif self.model_name == "text-embedding-ada-002":
                self.vector_size = 1536
        
    async def initialize(self):
        """Initialize the embedding model and vector storage"""
        try:
            # Initialize embedding model based on provider
            if self.provider == "local":
                logger.info(f"Loading local embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                if self.model_name == "all-MiniLM-L6-v2":
                    self.vector_size = 384
                logger.info("Local embedding model loaded successfully")
            elif self.provider == "openai":
                logger.info(f"Using OpenAI embedding model: {self.model_name}")
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided")
                
                # Initialize OpenAI client
                self.openai_client = AsyncOpenAI(api_key=api_key)
                
                # Test OpenAI connection
                try:
                    test_response = await self._openai_embed_text("test")
                    logger.info("OpenAI embedding service connected successfully")
                except Exception as e:
                    logger.error(f"Failed to connect to OpenAI: {e}")
                    raise
            
            # Initialize file vector store
            self.vector_store = FileVectorStore()
            logger.info("File vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    async def _openai_embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            # Add null checks
            if not response or not response.data or len(response.data) == 0:
                raise ValueError("OpenAI API returned empty or invalid response")
            
            embedding = response.data[0].embedding
            if not embedding:
                raise ValueError("OpenAI API returned empty embedding")
                
            return embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def _openai_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API"""
        try:
            # OpenAI has a limit on batch size, so we'll process in chunks
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                # Add null checks
                if not response or not response.data:
                    raise ValueError(f"OpenAI API returned empty response for batch {i//batch_size + 1}")
                
                if len(response.data) != len(batch):
                    raise ValueError(f"OpenAI API returned {len(response.data)} embeddings for {len(batch)} texts")
                
                batch_embeddings = []
                for item in response.data:
                    if not item or not item.embedding:
                        raise ValueError("OpenAI API returned empty embedding in batch")
                    batch_embeddings.append(item.embedding)
                
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            if self.provider == "openai":
                return await self._openai_embed_text(text)
            else:
                if not self.model:
                    raise RuntimeError("Local embedding model not initialized")
                
                # Run embedding in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, 
                    self.model.encode, 
                    text
                )
                
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if self.provider == "openai":
                return await self._openai_embed_texts(texts)
            else:
                if not self.model:
                    raise RuntimeError("Local embedding model not initialized")
                
                # Run embedding in thread pool
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    self.model.encode, 
                    texts
                )
                
                return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def store_embeddings(
        self, 
        chunks: List[Dict], 
        course_id: str,
        document_id: str,
        db_session: Optional[AsyncSession] = None
    ) -> List[str]:
        """Store document chunks with their embeddings in file storage"""
        try:
            if not chunks:
                logger.warning(f"No chunks provided for document {document_id}")
                return []
            
            # Extract texts for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks in document {document_id}")
            # Generate embeddings
            embeddings = await self.embed_texts(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings using {self.provider} provider")
            
            # Prepare vectors for storage
            vectors = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Use UUID for vector ID
                vector_id = str(uuid.uuid4())
                
                vector = {
                    "id": vector_id,
                    "vector": embedding,
                    "payload": {
                        "course_id": course_id,
                        "document_id": document_id,
                        "chunk_index": i,
                        "content": chunk['content'],
                        "metadata": chunk.get('metadata', {}),
                        "chunk_type": chunk.get('chunk_type', 'semantic')
                    }
                }
                vectors.append(vector)
            
            if not vectors:
                logger.error("No valid embeddings to store")
                return []
                
            logger.info(f"Storing {len(vectors)} vectors for document {document_id} in course {course_id}")
            # Store in file vector store
            vector_ids = await self.vector_store.store_vectors(vectors, course_id, document_id, db_session)
            
            logger.info(f"EMBEDDING SUCCESS: Stored {len(vectors)} embeddings for document {document_id}")
            return vector_ids
            
        except Exception as e:
            logger.error(f"ERROR storing embeddings: {e}")
            # More debug info
            logger.error(f"Provider: {self.provider}, Model: {self.model_name}")
            logger.error(f"Vector store type: {type(self.vector_store).__name__}")
            logger.error(f"Chunk count: {len(chunks) if chunks else 0}")
            raise
    
    async def search_similar(
        self, 
        query: str, 
        course_id: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        db_session: Optional[AsyncSession] = None
    ) -> List[Dict]:
        """Search for similar content using embeddings"""
        try:
            # Generate embedding for query
            query_embedding = await self.embed_text(query)
            
            # Search in vector store
            results = await self.vector_store.search_similar(
                query_embedding, 
                course_id, 
                limit=limit,
                score_threshold=score_threshold,
                db_session=db_session
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []
    
    async def delete_document_embeddings(self, document_id: str, course_id: str, db_session: Optional[AsyncSession] = None):
        """Delete embeddings for a document"""
        try:
            success = await self.vector_store.delete_document(document_id, course_id, db_session)
            if success:
                logger.info(f"Deleted embeddings for document {document_id}")
            else:
                logger.warning(f"No embeddings found for document {document_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting document embeddings: {e}")
            return False
    
    async def delete_course_embeddings(self, course_id: str, db_session: Optional[AsyncSession] = None):
        """Delete all embeddings for a course"""
        try:
            success = await self.vector_store.delete_course(course_id, db_session)
            if success:
                logger.info(f"Deleted all embeddings for course {course_id}")
            else:
                logger.warning(f"No embeddings found for course {course_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting course embeddings: {e}")
            return False
    
    async def get_collection_info(self) -> Dict:
        """Get information about the vector collection"""
        try:
            # For file store, we just return basic info
            return {
                "status": "active",
                "vector_count": -1,  # Unknown without scanning all files
                "storage_type": "file"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cleanup(self):
        """Clean up resources"""
        # No cleanup needed for file storage
        pass

# Global instance
embedding_service = EmbeddingService() 