#!/bin/bash
# Start script for memecoin trading bot on Oracle Cloud VM
# This script starts both the main trading bot and alpha discovery scheduler

set -e

# Change to project directory
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d "bot_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv bot_env
fi

# Activate virtual environment
source bot_env/bin/activate

# Install/update dependencies
echo "Updating dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p logs

# Check if config exists
if [ ! -f "config/config.yml" ]; then
    echo "ERROR: config/config.yml not found!"
    echo "Please copy config/config.yml.example to config/config.yml and configure your API keys"
    exit 1
fi

# Start alpha discovery in background (runs every 6 hours)
echo "Starting alpha wallet discovery scheduler..."
nohup python3 scripts/alpha_discovery_scheduler.py > logs/alpha_discovery_scheduler.log 2>&1 &
ALPHA_PID=$!
echo "Alpha discovery scheduler started with PID: $ALPHA_PID"

# Wait a moment for alpha discovery to initialize
sleep 5

# Start main trading bot
echo "Starting main trading bot..."
python3 start_bot.py

# If bot exits, kill alpha discovery
kill $ALPHA_PID 2>/dev/null || true