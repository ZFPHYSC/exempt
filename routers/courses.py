from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime
import uuid

from models.database import get_db, Course, Document
from pydantic import BaseModel

router = APIRouter()

# Pydantic models for request/response
class CourseCreate(BaseModel):
    name: str
    code: str
    description: str = ""

class CourseUpdate(BaseModel):
    name: str = None
    code: str = None
    description: str = None

class CourseResponse(BaseModel):
    id: str
    name: str
    code: str
    description: str = ""
    fileCount: int = 0
    moduleCount: int = 0
    created_at: Optional[str] = None
    last_sync: Optional[str] = None

@router.get("/", response_model=List[CourseResponse])
async def get_courses(db: AsyncSession = Depends(get_db)):
    """Get all courses with file counts"""
    try:
        # Get courses with file counts
        result = await db.execute(
            select(
                Course,
                func.count(Document.id).label('file_count')
            )
            .outerjoin(Document, Course.id == Document.course_id)
            .where(Course.is_active == True)
            .group_by(Course.id)
            .order_by(Course.created_at.desc())
        )
        
        courses_data = result.all()
        
        courses = []
        for course_row, file_count in courses_data:
            course_dict = course_row.to_dict()
            course_dict['fileCount'] = file_count or 0
            courses.append(CourseResponse(**course_dict))
        
        return courses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve courses: {str(e)}"
        )

@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(course_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific course by ID"""
    try:
        # Get course with file count
        result = await db.execute(
            select(
                Course,
                func.count(Document.id).label('file_count')
            )
            .outerjoin(Document, Course.id == Document.course_id)
            .where(Course.id == course_id, Course.is_active == True)
            .group_by(Course.id)
        )
        
        course_data = result.first()
        
        if not course_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        course_row, file_count = course_data
        course_dict = course_row.to_dict()
        course_dict['fileCount'] = file_count or 0
        
        return CourseResponse(**course_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve course: {str(e)}"
        )

@router.post("/", response_model=CourseResponse)
async def create_course(course_data: CourseCreate, db: AsyncSession = Depends(get_db)):
    """Create a new course"""
    try:
        # Check if course code already exists
        existing_course = await db.execute(
            select(Course).where(
                Course.code == course_data.code,
                Course.is_active == True
            )
        )
        
        if existing_course.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Course with code '{course_data.code}' already exists"
            )
        
        # Create new course
        new_course = Course(
            name=course_data.name,
            code=course_data.code,
            description=course_data.description
        )
        
        db.add(new_course)
        await db.commit()
        await db.refresh(new_course)
        
        # Return course data
        course_dict = new_course.to_dict()
        course_dict['fileCount'] = 0
        
        return CourseResponse(**course_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create course: {str(e)}"
        )

@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: str, 
    course_data: CourseUpdate, 
    db: AsyncSession = Depends(get_db)
):
    """Update an existing course"""
    try:
        # Get existing course
        result = await db.execute(
            select(Course).where(Course.id == course_id, Course.is_active == True)
        )
        course = result.scalar_one_or_none()
        
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        # Update fields
        update_data = course_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(course, field, value)
        
        course.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(course)
        
        # Get file count
        file_count_result = await db.execute(
            select(func.count(Document.id)).where(Document.course_id == course_id)
        )
        file_count = file_count_result.scalar() or 0
        
        course_dict = course.to_dict()
        course_dict['fileCount'] = file_count
        
        return CourseResponse(**course_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update course: {str(e)}"
        )

@router.delete("/{course_id}")
async def delete_course(course_id: str, db: AsyncSession = Depends(get_db)):
    """Soft delete a course"""
    try:
        # Get existing course
        result = await db.execute(
            select(Course).where(Course.id == course_id, Course.is_active == True)
        )
        course = result.scalar_one_or_none()
        
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        # Soft delete
        course.is_active = False
        course.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {"message": "Course deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete course: {str(e)}"
        )

@router.get("/{course_id}/documents")
async def get_course_documents(course_id: str, db: AsyncSession = Depends(get_db)):
    """Get all documents for a course"""
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
        
        # Get documents
        documents_result = await db.execute(
            select(Document)
            .where(Document.course_id == course_id)
            .order_by(Document.created_at.desc())
        )
        
        documents = documents_result.scalars().all()
        
        return [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "status": doc.status,
                "chunk_count": doc.chunk_count,
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            }
            for doc in documents
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@router.post("/{course_id}/sync")
async def update_course_sync(course_id: str, db: AsyncSession = Depends(get_db)):
    """Update course last sync timestamp"""
    try:
        result = await db.execute(
            select(Course).where(Course.id == course_id, Course.is_active == True)
        )
        course = result.scalar_one_or_none()
        
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        
        course.last_sync = datetime.utcnow()
        await db.commit()
        
        return {"message": "Sync timestamp updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update sync: {str(e)}"
        ) 