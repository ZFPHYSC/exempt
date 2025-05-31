#!/bin/bash

# Set up Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-fixed.txt

# Check if embeddings directory exists
EMBEDDINGS_DIR="../data/embeddings"
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "Creating embeddings directory..."
    mkdir -p "$EMBEDDINGS_DIR"
fi

# Run the check_embeddings script
echo "Checking embeddings..."
python check_embeddings.py

# Start the server
echo "Starting server..."
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 