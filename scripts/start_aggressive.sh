#!/bin/bash
# Aggressive Bot Only Startup Script for Memecoin Trading Bot
# Based on the original start_all.sh pattern

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

# Check aggressive config
if [ ! -f "config/config_aggressive.yml" ]; then
    echo "ERROR: config/config_aggressive.yml not found!"
    echo "Please create the aggressive configuration first."
    exit 1
fi

echo "⚡ Starting AGGRESSIVE Memecoin Trading Bot..."
echo "=============================================="
echo "📱 Discord: Aggressive webhook channel"
echo "📊 Log: logs/trading_aggressive.log"
echo "🎯 Strategy: Fast execution, single wallet triggers"
echo "=============================================="

# Start aggressive trading bot
"$PYTHON" main.py --config config_aggressive.yml --bot-name AGGRESSIVE