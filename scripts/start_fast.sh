#!/bin/bash
# Fast Execution Bot Startup Script for Memecoin Trading Bot
# Uses LaserStream (Geyser) + Jito bundles for sub-2-second execution

set -e

# Define full project path and Python environment
PROJECT_DIR="/home/opc/Memecoin-Trader"
VENV="$PROJECT_DIR/bot_env"
PYTHON="$VENV/bin/python3"

cd "$PROJECT_DIR"

# Create venv if needed
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

# Install dependencies
"$PYTHON" -m pip install -q -r requirements.txt

# Ensure logs folder exists
mkdir -p logs

# Check fast config
if [ ! -f "config/config_fast.yml" ]; then
    echo "ERROR: config/config_fast.yml not found!"
    echo "Please create the fast execution configuration first."
    exit 1
fi

echo "⚡ Starting FAST EXECUTION Memecoin Trading Bot..."
echo "=============================================="
echo "📱 Discord: Fast execution webhook channel"
echo "📊 Log: logs/trading_fast.log"
echo "🚀 Strategy: LaserStream + Jito bundles"
echo "🎯 Latency: Sub-2-second execution"
echo "=============================================="

# Start fast execution trading bot
"$PYTHON" start_bot_fast.py
