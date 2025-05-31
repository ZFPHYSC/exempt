#!/usr/bin/env python3
"""
Script to migrate embeddings to use course name-based folders.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

# Add the backend directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import AsyncSessionLocal
from services.file_vector_store import FileVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def migrate_embeddings():
    """Migrate embeddings to use course name-based folders"""
    logger.info("Starting embedding migration to course name-based folders")
    
    async with AsyncSessionLocal() as db_session:
        # Initialize the file vector store
        vector_store = FileVectorStore()
        
        # Run the migration
        result = await vector_store.migrate_to_named_folders(db_session)
        
        logger.info(f"Migration completed:")
        logger.info(f"  - Courses migrated: {result['migrated']}")
        logger.info(f"  - Courses not needing migration: {result['not_needed']}")
        logger.info(f"  - Errors: {result['errors']}")
        
        if result['errors'] > 0 and 'error_message' in result:
            logger.error(f"Migration error: {result['error_message']}")
        
        # Report summary
        if result['migrated'] > 0:
            logger.info("Migration successful! Embeddings are now organized by course name.")
        elif result['not_needed'] > 0 and result['errors'] == 0:
            logger.info("No migration needed. Embeddings were already properly organized.")
        else:
            logger.warning("No changes were made during migration.")

if __name__ == "__main__":
    asyncio.run(migrate_embeddings()) 