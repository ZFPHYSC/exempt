#!/usr/bin/env python3
"""
Script to reinitialize the course assistant database and embeddings.
This will remove all courses, files, and embeddings.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, exc

# Add the backend directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import AsyncSessionLocal, engine
from services.file_vector_store import FileVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def table_exists(db_session, table_name):
    """Check if a table exists in the database"""
    try:
        # Use PostgreSQL specific query to check if table exists
        query = text("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = :table_name
            )
        """)
        result = await db_session.execute(query, {"table_name": table_name})
        return result.scalar()
    except Exception as e:
        logger.error(f"Error checking if table {table_name} exists: {e}")
        return False

async def safe_delete_table(db_session, table_name):
    """Safely delete all records from a table if it exists"""
    try:
        if await table_exists(db_session, table_name):
            await db_session.execute(text(f"DELETE FROM {table_name}"))
            logger.info(f"Deleted all records from {table_name}")
            return True
        else:
            logger.warning(f"Table {table_name} does not exist, skipping")
            return False
    except Exception as e:
        logger.error(f"Error deleting from {table_name}: {e}")
        return False

async def reinitialize_database():
    """Reinitialize the database by removing all courses, files, and chat data"""
    logger.info("Starting database reinitialization")
    
    async with AsyncSessionLocal() as db_session:
        try:
            # List of tables to clear in the correct order (respect foreign key constraints)
            tables = [
                "chat_messages",
                "chat_sessions",
                "document_chunks",
                "file_modules",  # Junction table if it exists
                "files",
                "modules",
                "documents",     # Need to delete documents before courses
                "courses"
            ]
            
            # Delete from each table
            success = True
            for table in tables:
                if not await safe_delete_table(db_session, table):
                    # Continue with other tables even if one fails
                    success = False
            
            # Commit the changes
            await db_session.commit()
            
            if success:
                logger.info("Successfully reinitialized database")
            else:
                logger.warning("Database reinitialization completed with some warnings")
            
            return True
        except Exception as e:
            logger.error(f"Error reinitializing database: {e}")
            try:
                await db_session.rollback()
            except:
                pass
            return False

async def reinitialize_embeddings():
    """Reinitialize embeddings by removing all embedding files"""
    logger.info("Starting embeddings reinitialization")
    
    # Initialize the file vector store
    vector_store = FileVectorStore()
    
    # Run the reinitialization
    success = await vector_store.reinitialize()
    
    if success:
        logger.info("Successfully reinitialized embeddings storage")
    else:
        logger.error("Failed to reinitialize embeddings storage")
    
    return success

async def main():
    """Main function to run the reinitialization"""
    logger.info("Starting complete system reinitialization")
    
    # Reinitialize embeddings
    embeddings_success = await reinitialize_embeddings()
    
    # Reinitialize database
    db_success = await reinitialize_database()
    
    if embeddings_success and db_success:
        logger.info("Successfully reinitialized the entire system")
    else:
        logger.error("Reinitialization completed with errors")
        if not embeddings_success:
            logger.error("Failed to reinitialize embeddings")
        if not db_success:
            logger.error("Failed to reinitialize database")

if __name__ == "__main__":
    asyncio.run(main()) 