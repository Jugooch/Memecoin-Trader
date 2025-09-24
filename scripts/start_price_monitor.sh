#!/bin/bash

# Memecoin Price Monitor Service Launcher
# Runs independently from the main trading bot

echo "========================================="
echo "Starting Price Monitor Service"
echo "========================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Navigate to project root
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Log file with timestamp
LOG_FILE="logs/price_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Price Monitor service..."
echo "Logs will be written to: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Run the price monitor with logging
python3 start_price_monitor.py 2>&1 | tee "$LOG_FILE"