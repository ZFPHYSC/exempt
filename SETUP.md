# Course Assistant Critical Fixes - Setup Guide

This document contains all the critical fixes that have been applied to resolve database schema issues, service initialization problems, and API communication errors.

## Quick Start

1. **Copy environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start services and run migrations:**
   ```bash
   ./reset_and_start.sh
   ```

## Critical Fixes Applied

### 1. Database Schema Fixes ✅

**Problem:** Missing `confidence` column in `ChatMessage` model
**Solution:** Added `confidence = Column(Float, default=0.0)` to the model

**Files Modified:**
- `models/database.py` - Added Float import and confidence column
- `alembic/versions/001_add_missing_columns.py` - Migration script

### 2. Service Initialization Fixes ✅

**Problem:** IngestionService not properly initialized with EmbeddingService
**Solution:** Added `initialize()` method and proper service dependency injection

**Files Modified:**
- `main.py` - Updated lifespan to initialize ingestion service
- `services/ingestion.py` - Added initialize method and simplified processing

### 3. API Communication Fixes ✅

**Problem:** Chat API UUID handling and request format issues
**Solution:** Fixed UUID conversion and request/response format

**Files Modified:**
- `routers/chat.py` - Fixed UUID handling and confidence field
- `frontend/app/courses/[id]/page.tsx` - Updated request format

### 4. File Upload Simplification ✅

**Problem:** Complex queuing system causing processing failures
**Solution:** Simplified to immediate processing with better error handling

**Files Modified:**
- `main.py` - Simplified upload endpoint
- `services/ingestion.py` - Added simple chunking method

### 5. Database Migration Setup ✅

**Problem:** No migration system for schema changes
**Solution:** Added Alembic configuration and migration scripts

**Files Added:**
- `alembic.ini` - Alembic configuration
- `alembic/env.py` - Migration environment
- `alembic/script.py.mako` - Migration template
- `alembic/versions/001_add_missing_columns.py` - Initial migration

## Environment Variables

Make sure your `.env` file contains:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/course_assistant
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_actual_openai_key
OPENROUTER_API_KEY=your_actual_openrouter_key
CHAT_MODEL_PROVIDER=openrouter
CHAT_MODEL=google/gemini-2.5-flash-preview-05-20:thinking
EMBEDDING_MODEL_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIMENSION=1536
```

## Manual Setup (Alternative to reset_and_start.sh)

1. **Start Docker services:**
   ```bash
   docker-compose up -d postgres qdrant minio redis
   ```

2. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

3. **Start the backend:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Start the frontend (in another terminal):**
   ```bash
   cd ../frontend
   npm run dev
   ```

## Testing the Fixes

1. **Test file upload:**
   - Go to http://localhost:3000
   - Create a course
   - Upload a PDF file
   - Verify it processes successfully

2. **Test chat functionality:**
   - Navigate to a course page
   - Send a message
   - Verify you get a response with sources

3. **Test database schema:**
   - Check that chat messages are saved with confidence scores
   - Verify no database errors in logs

## Common Issues and Solutions

### Issue: "confidence column doesn't exist"
**Solution:** Run the migration: `alembic upgrade head`

### Issue: "EmbeddingService not initialized"
**Solution:** Ensure services are initialized in the correct order in `main.py`

### Issue: "UUID conversion error"
**Solution:** The chat router now handles both string and UUID formats

### Issue: "File processing fails"
**Solution:** Check that all required packages are installed: `pip install -r requirements.txt`

## Dependencies Added

- `pdfplumber==0.10.3` - For better PDF text extraction
- `alembic==1.12.1` - For database migrations (already in requirements)

## Architecture Improvements

1. **Simplified Processing Pipeline:**
   - Direct file processing instead of queuing
   - Better error handling and logging
   - Immediate feedback to users

2. **Robust Service Initialization:**
   - Proper dependency injection
   - Clear initialization order
   - Better error reporting

3. **Improved API Design:**
   - Consistent UUID handling
   - Proper error responses
   - Better request/response formats

## Next Steps

1. Add your actual API keys to `.env`
2. Test the complete workflow
3. Monitor logs for any remaining issues
4. Consider adding more comprehensive error handling as needed

 