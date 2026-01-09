#!/bin/bash
# Start app with V2 analyzer

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Kill any process using port 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "⚠️  Port 8000 is in use, killing process..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

export USE_V2_ANALYZER=true
echo "🔵 Starting V2 analyzer..."
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
