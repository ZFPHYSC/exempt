#!/bin/bash

# Remove old database file if it exists
if [ -f "course_assistant.db" ]; then
    echo "Removing old database..."
    rm course_assistant.db
fi

# Run migrations
echo "Running migrations..."
alembic upgrade head

# Start backend
echo "Starting backend..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 