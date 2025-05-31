# Course Assistant

A comprehensive AI-powered course assistant that automatically syncs content from your LMS (Learning Management System) and provides intelligent chat functionality with your course materials.

## Features

- **🤖 AI Chat Interface**: Ask questions about your course content with context-aware responses
- **📚 Smart Document Processing**: Automatically processes PDFs, DOCX, PPTX, XLSX, and TXT files
- **🔄 LMS Integration**: Chrome extension syncs courses from Brightspace, Canvas, Moodle, and more
- **🎨 Beautiful UI**: Modern, responsive interface with dark/light mode
- **⚡ Real-time Updates**: WebSocket-powered progress tracking and live chat
- **🔍 Vector Search**: Semantic search using embeddings for relevant content retrieval

## Architecture

- **Frontend**: Next.js 14 with TypeScript, Tailwind CSS, Framer Motion
- **Backend**: FastAPI with async SQLAlchemy, WebSocket support
- **Vector Database**: Qdrant for semantic search
- **File Storage**: MinIO (S3-compatible) for document storage
- **Database**: PostgreSQL for metadata
- **Cache**: Redis for session management
- **Extension**: Chrome extension for LMS scraping

## Quick Setup

### 1. Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Docker and Docker Compose

### 2. Clone and Setup

```bash
git clone <your-repo>
cd course-assistant

# Start services
docker-compose up -d

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy environment file
cp env.example .env
# Edit .env with your configuration

# Setup frontend
cd ../frontend
npm install
```

### 3. Start Development Servers

```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### 4. Install Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `extension` folder
4. Pin the extension to your toolbar

## Usage

### Adding Courses

**Method 1: Chrome Extension (Recommended)**
1. Navigate to your LMS course page
2. Click the "🤖 Sync with Course Assistant" button that appears
3. The extension will automatically extract course info and files

**Method 2: Manual Upload**
1. Open the Course Assistant web app
2. Click "Add Course"
3. Enter course details and upload files manually

### Chatting with Your Course

1. Click on any course card
2. Start asking questions about your course content
3. The AI will provide context-aware responses with source citations

### Supported LMS Platforms

- **Brightspace** (D2L) - e.g., OnQ at Queen's University
- **Canvas** - Instructure Canvas
- **Moodle** - Including eClass variants
- **Learn** - University of Waterloo Learn

## Configuration

### Backend Environment (.env)

```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/course_assistant

# Vector Database
QDRANT_URL=http://localhost:6333

# Redis
REDIS_URL=redis://localhost:6379

# MinIO Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Optional: OpenAI for advanced features
OPENAI_API_KEY=your_key_here
```

### Frontend Environment (.env.local)

```bash
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Development

### Project Structure

```
course-assistant/
├── frontend/                 # Next.js 14 App
│   ├── app/                 # App router pages
│   ├── components/          # React components
│   └── lib/                 # Utilities
├── backend/                 # FastAPI Backend
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   └── routers/            # API routes
├── extension/              # Chrome Extension
└── docker-compose.yml     # Services
```

### Key Components

**Frontend:**
- `CourseCard.tsx` - Course display with file upload
- `ChatInterface.tsx` - Real-time chat with WebSocket
- `AddCourseModal.tsx` - Manual course creation
- `websocket.ts` - WebSocket client utility

**Backend:**
- `models/database.py` - SQLAlchemy models
- `services/embedding.py` - Vector embeddings with Qdrant
- `services/ingestion.py` - Document processing pipeline
- `services/query.py` - RAG query processing
- `routers/` - API endpoints for courses, chat, sync

**Extension:**
- `content.js` - LMS data extraction
- `background.js` - Extension background tasks
- `manifest.json` - Chrome extension configuration

### Adding New LMS Support

To add support for a new LMS:

1. Update `extension/content.js`:
   - Add detection logic in `detectLMS()`
   - Create extraction method like `extractNewLMSData()`
   - Add selectors for course info and files

2. Update `extension/manifest.json`:
   - Add new LMS URLs to `host_permissions`
   - Add to `content_scripts` matches

## API Documentation

Once running, visit:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001

### Key Endpoints

- `GET /api/courses` - List all courses
- `POST /api/courses` - Create new course
- `POST /api/chat/{course_id}` - Send chat message
- `GET /api/chat/{course_id}/history` - Get chat history
- `POST /api/upload` - Upload course files
- `POST /api/sync/course` - Sync course from extension

## Troubleshooting

### Common Issues

1. **Extension not detecting courses**: Check if you're on a supported LMS and the page contains course information
2. **File processing fails**: Ensure file formats are supported (.pdf, .docx, .pptx, .xlsx, .txt)
3. **Chat responses are poor**: Make sure documents are properly processed and indexed
4. **WebSocket connection fails**: Check if backend is running and ports are correct

### Logs

- Backend logs: Check terminal running `python main.py`
- Frontend logs: Check browser console
- Extension logs: Chrome DevTools > Extensions > Course Assistant > background page

## Production Deployment

### Docker Production

```bash
# Update docker-compose.yml with production settings
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables

Set these for production:
- `SECRET_KEY` - Random secure key
- `DATABASE_URL` - Production database
- `OPENAI_API_KEY` - For advanced AI features
- `DEBUG=false`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Open an issue on GitHub 