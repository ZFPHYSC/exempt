from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import asyncio

from models.database import get_db, Course, Document, ProcessingQueue
from services.ingestion import IngestionService

router = APIRouter()

# Initialize ingestion service
ingestion_service = IngestionService()

# Pydantic models
class CourseData(BaseModel):
    id: str
    name: str
    code: str
    description: str = ""
    modules: List[Dict] = []

class FileData(BaseModel):
    courseId: str
    filename: str
    path: str
    downloadUrl: str = ""

class SyncStatus(BaseModel):
    status: str
    message: str = ""
    courses_found: int = 0
    files_processed: int = 0

@router.post("/course")
async def sync_course_from_extension(
    course_data: CourseData,
    db: AsyncSession = Depends(get_db)
):
    """Receive course data from extension"""
    try:
        # Check if course already exists
        existing_course = await db.execute(
            select(Course).where(
                Course.code == course_data.code,
                Course.is_active == True
            )
        )
        course = existing_course.scalar_one_or_none()
        
        if course:
            # Update existing course
            course.name = course_data.name
            course.description = course_data.description
            course.last_sync = datetime.utcnow()
            
            # Update module count
            course.module_count = len(course_data.modules)
        else:
            # Create new course
            course = Course(
                name=course_data.name,
                code=course_data.code,
                description=course_data.description,
                module_count=len(course_data.modules),
                last_sync=datetime.utcnow()
            )
            db.add(course)
        
        await db.commit()
        await db.refresh(course)
        
        return {
            "success": True,
            "course_id": str(course.id),
            "message": f"Course '{course.name}' synced successfully",
            "modules_found": len(course_data.modules)
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync course: {str(e)}"
        )

@router.post("/file-ready")
async def file_ready_for_processing(
    file_data: FileData,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Notification that a file is ready for processing"""
    try:
        # Verify course exists
        course_result = await db.execute(
            select(Course).where(Course.id == file_data.courseId, Course.is_active == True)
        )
        course = course_result.scalar_one_or_none()
        
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        # Queue file for processing
        await ingestion_service.queue_file(
            course_id=file_data.courseId,
            file_path=file_data.path,
            filename=file_data.filename
        )
        
        # Start processing in background
        background_tasks.add_task(
            process_single_file,
            file_data.courseId,
            file_data.path,
            file_data.filename
        )
        
        return {
            "success": True,
            "message": f"File '{file_data.filename}' queued for processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue file: {str(e)}"
        )

@router.get("/status/{course_id}")
async def get_sync_status(course_id: str, db: AsyncSession = Depends(get_db)):
    """Get synchronization status for a course"""
    try:
        # Get course
        course_result = await db.execute(
            select(Course).where(Course.id == course_id, Course.is_active == True)
        )
        course = course_result.scalar_one_or_none()
        
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        # Get processing queue status
        queue_result = await db.execute(
            select(ProcessingQueue).where(ProcessingQueue.course_id == course_id)
        )
        queue_items = queue_result.scalars().all()
        
        # Get document count
        doc_result = await db.execute(
            select(Document).where(Document.course_id == course_id)
        )
        documents = doc_result.scalars().all()
        
        # Calculate status
        total_queued = len(queue_items)
        completed = len([item for item in queue_items if item.status == "completed"])
        processing = len([item for item in queue_items if item.status == "processing"])
        failed = len([item for item in queue_items if item.status == "failed"])
        
        status_text = "idle"
        if processing > 0:
            status_text = "processing"
        elif total_queued > completed:
            status_text = "queued"
        
        return {
            "course_id": course_id,
            "status": status_text,
            "last_sync": course.last_sync.isoformat() if course.last_sync else None,
            "total_documents": len(documents),
            "queue_status": {
                "total": total_queued,
                "completed": completed,
                "processing": processing,
                "failed": failed
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )

@router.post("/bulk-files")
async def process_bulk_files(
    files_data: List[FileData],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Process multiple files from extension sync"""
    try:
        results = []
        
        for file_data in files_data:
            # Verify course exists
            course_result = await db.execute(
                select(Course).where(Course.id == file_data.courseId, Course.is_active == True)
            )
            
            if not course_result.scalar_one_or_none():
                results.append({
                    "filename": file_data.filename,
                    "status": "error",
                    "message": "Course not found"
                })
                continue
            
            # Queue for processing
            try:
                await ingestion_service.queue_file(
                    course_id=file_data.courseId,
                    file_path=file_data.path,
                    filename=file_data.filename
                )
                
                results.append({
                    "filename": file_data.filename,
                    "status": "queued",
                    "message": "Successfully queued for processing"
                })
                
            except Exception as e:
                results.append({
                    "filename": file_data.filename,
                    "status": "error",
                    "message": str(e)
                })
        
        # Start batch processing in background
        successful_files = [r for r in results if r["status"] == "queued"]
        if successful_files:
            background_tasks.add_task(
                process_batch_files,
                files_data
            )
        
        return {
            "success": True,
            "total_files": len(files_data),
            "queued": len(successful_files),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk processing failed: {str(e)}"
        )

@router.delete("/queue/{course_id}")
async def clear_processing_queue(course_id: str, db: AsyncSession = Depends(get_db)):
    """Clear processing queue for a course"""
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
        
        # Delete queue items that are not currently processing
        await db.execute(
            "DELETE FROM processing_queue WHERE course_id = :course_id AND status != 'processing'",
            {"course_id": course_id}
        )
        
        await db.commit()
        
        return {"message": "Processing queue cleared"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear queue: {str(e)}"
        )

@router.post("/retry-failed/{course_id}")
async def retry_failed_files(
    course_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Retry processing failed files for a course"""
    try:
        # Get failed queue items
        failed_result = await db.execute(
            select(ProcessingQueue).where(
                ProcessingQueue.course_id == course_id,
                ProcessingQueue.status == "failed"
            )
        )
        failed_items = failed_result.scalars().all()
        
        if not failed_items:
            return {"message": "No failed files to retry"}
        
        # Reset status to queued
        for item in failed_items:
            item.status = "queued"
            item.retry_count += 1
        
        await db.commit()
        
        # Start processing in background
        background_tasks.add_task(
            retry_failed_processing,
            course_id,
            [item.file_path for item in failed_items]
        )
        
        return {
            "message": f"Retrying {len(failed_items)} failed files",
            "files": [item.filename for item in failed_items]
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry files: {str(e)}"
        )

# Background task functions
async def process_single_file(course_id: str, file_path: str, filename: str):
    """Background task to process a single file"""
    try:
        success = await ingestion_service.process_file(course_id, file_path, filename)
        if success:
            print(f"Successfully processed {filename}")
        else:
            print(f"Failed to process {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

async def process_batch_files(files_data: List[FileData]):
    """Background task to process multiple files"""
    for file_data in files_data:
        try:
            await process_single_file(
                file_data.courseId,
                file_data.path,
                file_data.filename
            )
            # Small delay between files to prevent overwhelming the system
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in batch processing {file_data.filename}: {e}")

async def retry_failed_processing(course_id: str, file_paths: List[str]):
    """Background task to retry failed file processing"""
    for file_path in file_paths:
        try:
            filename = file_path.split('/')[-1]
            await process_single_file(course_id, file_path, filename)
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error retrying {file_path}: {e}") 