#!/bin/bash
# Safe startup for Memecoin Trading Bot under systemd

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

# Check config
if [ ! -f "config/config.yml" ]; then
    echo "ERROR: config/config.yml not found!"
    exit 1
fi

# Start alpha accumulator (continuous discovery) in background
nohup "$PYTHON" scripts/run_alpha_accumulator.py --loop > logs/alpha_accumulator.log 2>&1 &

# Start main trading bot (will keep the systemd service alive)
"$PYTHON" start_bot.py
