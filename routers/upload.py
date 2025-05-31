from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil
import asyncio
import logging
from pathlib import Path
import uuid
from datetime import datetime

from services.ingestion import IngestionService
from services.websocket import websocket_manager
from models.database import get_db, Course
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

router = APIRouter()
logger = logging.getLogger(__name__)

# Storage configuration
STORAGE_DIR = Path(os.getenv("PERSISTENT_STORAGE_DIR", "./storage"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "./temp"))

# Ensure directories exist
STORAGE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

class FileProcessor:
    def __init__(self):
        self.ingestion_service = IngestionService()
        
    async def process_files_with_progress(
        self, 
        files: List[UploadFile], 
        course_id: str,
        db: AsyncSession
    ):
        """Process files with real-time progress updates"""
        total_files = len(files)
        processed_files = 0
        failed_files = []
        
        try:
            # Initialize ingestion service
            await self.ingestion_service.initialize()
            
            # Send initial progress
            await websocket_manager.send_progress(0, f"Starting to process {total_files} files...")
            
            for i, file in enumerate(files):
                try:
                    # Update progress
                    progress = int((i / total_files) * 90)  # Reserve 10% for final steps
                    await websocket_manager.send_progress(
                        progress, 
                        f"Processing {file.filename} ({i+1}/{total_files})"
                    )
                    
                    # Save file to persistent storage
                    file_path = await self._save_file_permanently(file, course_id)
                    
                    # Process the file
                    await self.ingestion_service.process_file(
                        file_path=str(file_path),
                        course_id=course_id,
                        filename=file.filename
                    )
                    
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    failed_files.append({
                        "filename": file.filename,
                        "error": str(e)
                    })
            
            # Update course file count
            await self._update_course_file_count(course_id, db)
            
            # Send completion progress
            await websocket_manager.send_progress(
                100, 
                f"Completed! Processed {processed_files}/{total_files} files successfully"
            )
            
            # Send final status
            await websocket_manager.send_status("completed")
            
            return {
                "processed": processed_files,
                "failed": len(failed_files),
                "failures": failed_files,
                "total": total_files
            }
            
        except Exception as e:
            logger.error(f"Error in file processing: {e}")
            await websocket_manager.send_status("error")
            raise
    
    async def _save_file_permanently(self, file: UploadFile, course_id: str) -> Path:
        """Save uploaded file to persistent storage"""
        # Create course-specific directory
        course_dir = STORAGE_DIR / course_id
        course_dir.mkdir(exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        safe_filename = f"{timestamp}_{file_id}_{file.filename}"
        
        file_path = course_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Reset file pointer for potential reuse
        await file.seek(0)
        
        logger.info(f"Saved file {file.filename} to {file_path}")
        return file_path
    
    async def _update_course_file_count(self, course_id: str, db: AsyncSession):
        """Update the file count for a course"""
        try:
            # Count files in storage directory
            course_dir = STORAGE_DIR / course_id
            if course_dir.exists():
                file_count = len([f for f in course_dir.iterdir() if f.is_file()])
            else:
                file_count = 0
            
            # Update database
            await db.execute(
                update(Course)
                .where(Course.id == course_id)
                .values(file_count=file_count)
            )
            await db.commit()
            
            logger.info(f"Updated file count for course {course_id}: {file_count}")
            
        except Exception as e:
            logger.error(f"Error updating file count: {e}")

# Global processor instance
file_processor = FileProcessor()

@router.post("/upload/{course_id}")
async def upload_files(
    course_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process files for a course"""
    try:
        # Validate course exists
        result = await db.execute(select(Course).where(Course.id == course_id))
        course = result.scalar_one_or_none()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Validate files
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Check file types
        allowed_extensions = os.getenv("ALLOWED_FILE_TYPES", ".pdf,.docx,.pptx,.xlsx,.txt,.jpg,.jpeg,.png,.gif,.bmp,.tiff,.webp").split(",")
        for file in files:
            if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type not supported: {file.filename}"
                )
        
        # Start processing in background
        background_tasks.add_task(
            file_processor.process_files_with_progress,
            files,
            course_id,
            db
        )
        
        # Send initial status
        await websocket_manager.send_status("processing")
        
        return JSONResponse(
            status_code=202,
            content={
                "message": f"Started processing {len(files)} files",
                "files": [f.filename for f in files],
                "status": "processing"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{course_id}")
async def list_course_files(
    course_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List all files for a course"""
    try:
        # Validate course exists
        result = await db.execute(select(Course).where(Course.id == course_id))
        course = result.scalar_one_or_none()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # List files in storage directory
        course_dir = STORAGE_DIR / course_id
        files = []
        
        if course_dir.exists():
            for file_path in course_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "path": str(file_path.relative_to(STORAGE_DIR))
                    })
        
        return {
            "course_id": course_id,
            "files": files,
            "total_files": len(files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{course_id}/{filename}")
async def delete_file(
    course_id: str,
    filename: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific file"""
    try:
        # Validate course exists
        result = await db.execute(select(Course).where(Course.id == course_id))
        course = result.scalar_one_or_none()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Find and delete file
        course_dir = STORAGE_DIR / course_id
        file_path = course_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path.unlink()
        
        # Update file count
        await file_processor._update_course_file_count(course_id, db)
        
        return {"message": f"File {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 