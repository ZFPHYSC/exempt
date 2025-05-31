from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from typing import List
import json
import os
from dotenv import load_dotenv

from services.embedding import EmbeddingService
from services.ingestion import IngestionService
from services.query import QueryService
from routers import courses, chat, sync
from models.database import engine, Base

# Load environment variables
load_dotenv()

# Initialize services globally
embedding_service = EmbeddingService()
ingestion_service = IngestionService()
query_service = QueryService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up Course Assistant backend...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize services
    await embedding_service.initialize()
    await query_service.initialize(embedding_service)
    await ingestion_service.initialize(embedding_service)
    
    print("Services initialized successfully!")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await embedding_service.cleanup()

app = FastAPI(
    title="Course Assistant API",
    description="AI-powered course assistant backend",
    version="1.0.0",
    lifespan=lifespan
)

# Update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(courses.router, prefix="/api/courses", tags=["courses"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(sync.router, prefix="/api/sync", tags=["sync"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "heartbeat", "data": data})
    except:
        manager.disconnect(websocket)

# Simple file upload endpoint
@app.post("/api/upload")
async def upload_files(
    courseId: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Handle file uploads"""
    try:
        results = []
        
        for file in files:
            # Save file
            file_path = f"temp/{courseId}_{file.filename}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process immediately
            success = await ingestion_service.process_file(courseId, file_path, file.filename)
            
            results.append({
                "filename": file.filename,
                "status": "success" if success else "failed"
            })
            
            # Send progress update
            await manager.broadcast({
                "type": "file_processed",
                "filename": file.filename,
                "courseId": courseId
            })
        
        return {
            "success": True,
            "uploaded": len([r for r in results if r["status"] == "success"]),
            "results": results
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Course Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}