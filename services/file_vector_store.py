import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session
from models.database import Course

logger = logging.getLogger(__name__)

class FileVectorStore:
    """Simple file-based vector store for embeddings"""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            # Use absolute path to project root + data/embeddings
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.storage_dir = os.path.join(base_dir, "data", "embeddings")
        else:
            self.storage_dir = storage_dir
        
        self._ensure_dir_exists()
    
    def _ensure_dir_exists(self):
        """Ensure the storage directory exists"""
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"Using file vector store at absolute path: {os.path.abspath(self.storage_dir)}")
    
    async def _get_course_name(self, course_id: str, db_session: Optional[AsyncSession] = None) -> str:
        """Get the course name from the database or return the course_id if not found"""
        try:
            if db_session:
                # Query the database for the course name
                result = await db_session.execute(
                    select(Course).where(Course.id == course_id)
                )
                course = result.scalar_one_or_none()
                if course:
                    # Sanitize the course name for use as a directory name
                    course_name = f"{course.code}_{course.name}"
                    # Replace any characters that are not valid in directory names
                    course_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in course_name)
                    return course_name
            return course_id
        except Exception as e:
            logger.error(f"Error getting course name: {e}")
            return course_id
    
    def _get_course_dir(self, course_id: str) -> str:
        """Get the directory for a specific course"""
        # Look for directories that are course directories
        for dir_name in os.listdir(self.storage_dir):
            dir_path = os.path.join(self.storage_dir, dir_name)
            if os.path.isdir(dir_path):
                # Try to find a mapping file that maps this course name to the ID
                mapping_file = os.path.join(dir_path, "course_info.json")
                if os.path.exists(mapping_file):
                    try:
                        with open(mapping_file, 'r') as f:
                            course_info = json.load(f)
                            if course_info.get("id") == course_id:
                                return dir_path
                    except:
                        pass
                
                # For backward compatibility, also check if directory starts with course ID
                if dir_name.startswith(f"{course_id}_") or dir_name == course_id:
                    return dir_path
        
        # If not found, return the course ID for backward compatibility
        course_dir = os.path.join(self.storage_dir, course_id)
        os.makedirs(course_dir, exist_ok=True)
        return course_dir
    
    def _get_document_path(self, course_id: str, document_id: str) -> str:
        """Get the file path for a specific document"""
        return os.path.join(self._get_course_dir(course_id), f"{document_id}.json")
    
    async def store_vectors(self, vectors: List[Dict[str, Any]], course_id: str, document_id: str, db_session: Optional[AsyncSession] = None) -> List[str]:
        """Store vectors for a document and return their IDs"""
        try:
            # Assign unique IDs to vectors
            for i, vector in enumerate(vectors):
                if "id" not in vector:
                    vector["id"] = f"{document_id}_{i}"
            
            # Get course name and create directory if needed
            course_name = await self._get_course_name(course_id, db_session)
            course_dir = os.path.join(self.storage_dir, course_name)
            os.makedirs(course_dir, exist_ok=True)
            
            # Create a mapping file to map course name to ID
            mapping_file = os.path.join(course_dir, "course_info.json")
            if not os.path.exists(mapping_file):
                with open(mapping_file, 'w') as f:
                    json.dump({"id": course_id, "name": course_name}, f)
            
            # Save vectors to file
            file_path = os.path.join(course_dir, f"{document_id}.json")
            with open(file_path, 'w') as f:
                json.dump(vectors, f)
            
            logger.info(f"SUCCESS: Stored {len(vectors)} vectors for document {document_id} at {file_path}")
            return [v["id"] for v in vectors]
        except Exception as e:
            logger.error(f"ERROR in store_vectors: {str(e)}")
            # Print more debug info
            logger.error(f"Course ID: {course_id}, Document ID: {document_id}")
            logger.error(f"Storage dir exists: {os.path.exists(self.storage_dir)}")
            logger.error(f"Storage dir writable: {os.access(self.storage_dir, os.W_OK)}")
            raise
    
    async def search_similar(self, query_vector: List[float], course_id: str, limit: int = 10, 
                       score_threshold: float = 0.7, db_session: Optional[AsyncSession] = None) -> List[Dict]:
        """Search for similar vectors"""
        try:
            course_dir = self._get_course_dir(course_id)
            
            if not os.path.exists(course_dir):
                logger.warning(f"Course directory does not exist: {course_dir}")
                return []
                
            # Convert query vector to numpy for faster calculations
            query_np = np.array(query_vector)
            
            results = []
            # Iterate through all document files in the course directory
            files_found = 0
            for file_name in os.listdir(course_dir):
                if not file_name.endswith('.json') or file_name == "course_info.json":
                    continue
                    
                files_found += 1
                file_path = os.path.join(course_dir, file_name)
                with open(file_path, 'r') as f:
                    vectors = json.load(f)
                
                for vector in vectors:
                    # Calculate cosine similarity
                    vec_np = np.array(vector["vector"])
                    similarity = np.dot(query_np, vec_np) / (np.linalg.norm(query_np) * np.linalg.norm(vec_np))
                    
                    if similarity >= score_threshold:
                        results.append({
                            "id": vector["id"],
                            "score": float(similarity),
                            "payload": vector["payload"]
                        })
            
            # Sort by score and limit results
            results.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Found {len(results)} similar vectors from {files_found} files in course {course_id}")
            return results[:limit]
        except Exception as e:
            logger.error(f"ERROR in search_similar: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str, course_id: str, db_session: Optional[AsyncSession] = None) -> bool:
        """Delete vectors for a document"""
        try:
            course_dir = self._get_course_dir(course_id)
            file_path = os.path.join(course_dir, f"{document_id}.json")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted vectors for document {document_id} at {file_path}")
                return True
            logger.warning(f"No vectors found at {file_path} for document {document_id}")
            return False
        except Exception as e:
            logger.error(f"ERROR in delete_document: {str(e)}")
            return False
    
    async def delete_course(self, course_id: str, db_session: Optional[AsyncSession] = None) -> bool:
        """Delete all vectors for a course"""
        try:
            course_dir = self._get_course_dir(course_id)
            if os.path.exists(course_dir):
                file_count = 0
                for file_name in os.listdir(course_dir):
                    if file_name.endswith('.json'):
                        os.remove(os.path.join(course_dir, file_name))
                        file_count += 1
                
                # Remove the directory itself
                if file_count > 0:
                    try:
                        os.rmdir(course_dir)
                    except:
                        logger.warning(f"Could not remove course directory {course_dir}, it may not be empty")
                
                logger.info(f"Deleted {file_count} vector files for course {course_id}")
                return True
            logger.warning(f"No vectors directory found for course {course_id}")
            return False
        except Exception as e:
            logger.error(f"ERROR in delete_course: {str(e)}")
            return False
    
    async def migrate_to_named_folders(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Migrate existing embeddings to named folders"""
        try:
            migrated = 0
            errors = 0
            not_needed = 0
            
            # Find all course ID directories
            for dir_name in os.listdir(self.storage_dir):
                dir_path = os.path.join(self.storage_dir, dir_name)
                
                # Skip if not a directory
                if not os.path.isdir(dir_path):
                    continue
                
                # Skip if it's already a course name folder (has course_info.json)
                if os.path.exists(os.path.join(dir_path, "course_info.json")):
                    not_needed += 1
                    continue
                
                # Try to get course ID (assuming it's a UUID in the directory name)
                course_id = dir_name
                if '_' in course_id:
                    course_id = course_id.split('_')[0]
                
                # Try to get course name from DB
                try:
                    course_name = await self._get_course_name(course_id, db_session)
                    
                    # Skip if we couldn't get a proper name (will just be the ID)
                    if course_name == course_id:
                        logger.warning(f"Could not find course name for ID {course_id}")
                        not_needed += 1
                        continue
                    
                    # Create new directory with course name
                    new_dir_path = os.path.join(self.storage_dir, course_name)
                    if os.path.exists(new_dir_path) and os.path.samefile(dir_path, new_dir_path):
                        # Same directory, just add the mapping file
                        with open(os.path.join(dir_path, "course_info.json"), 'w') as f:
                            json.dump({"id": course_id, "name": course_name}, f)
                        not_needed += 1
                        continue
                    
                    os.makedirs(new_dir_path, exist_ok=True)
                    
                    # Create mapping file
                    with open(os.path.join(new_dir_path, "course_info.json"), 'w') as f:
                        json.dump({"id": course_id, "name": course_name}, f)
                    
                    # Move all files
                    files_moved = 0
                    for file_name in os.listdir(dir_path):
                        if file_name.endswith('.json'):
                            old_file_path = os.path.join(dir_path, file_name)
                            new_file_path = os.path.join(new_dir_path, file_name)
                            # Copy instead of move to be safer
                            with open(old_file_path, 'r') as src, open(new_file_path, 'w') as dst:
                                dst.write(src.read())
                            files_moved += 1
                    
                    logger.info(f"Migrated {files_moved} files from {dir_name} to {course_name}")
                    migrated += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating course {dir_name}: {e}")
                    errors += 1
            
            return {
                "migrated": migrated,
                "errors": errors,
                "not_needed": not_needed
            }
            
        except Exception as e:
            logger.error(f"Error in migrate_to_named_folders: {e}")
            return {"migrated": 0, "errors": 1, "error_message": str(e)}

    async def reinitialize(self):
        """Remove all embeddings and initialize a fresh storage"""
        try:
            # Count files for logging
            dir_count = 0
            file_count = 0
            
            for dir_name in os.listdir(self.storage_dir):
                dir_path = os.path.join(self.storage_dir, dir_name)
                if os.path.isdir(dir_path):
                    dir_count += 1
                    # Delete all files in directory
                    for file_name in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file_name)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            file_count += 1
                    
                    # Remove directory
                    os.rmdir(dir_path)
            
            logger.info(f"Reinitialized embeddings storage: removed {file_count} files from {dir_count} directories")
            return True
        except Exception as e:
            logger.error(f"Error reinitializing embeddings storage: {e}")
            return False 