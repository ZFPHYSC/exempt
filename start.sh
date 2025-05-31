#!/bin/bash

# Course Assistant Backend Startup Script
echo "Starting Course Assistant Backend..."

# Navigate to backend directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Start the application
python main.py 