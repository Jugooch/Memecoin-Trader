#!/bin/bash
# Enhanced startup script for Memecoin Trading Bot
# Supports single bot (original) or dual bot mode

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

# Function to show menu
show_menu() {
    echo "🤖 Memecoin Trading Bot Startup"
    echo "======================================"
    echo "1. Conservative Bot Only (Original)"
    echo "2. Aggressive Bot Only"
    echo "3. Both Bots (Dual Mode)"
    echo "4. Exit"
    echo ""
    echo -n "Choose option (1-4): "
}

# Function to start single bot
start_single_bot() {
    local config_file=$1
    local bot_name=$2
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: $config_file not found!"
        exit 1
    fi
    
    echo "🚀 Starting $bot_name Bot..."
    "$PYTHON" main.py --config "$config_file" --bot-name "$bot_name"
}

# Function to start dual bots
start_dual_bots() {
    # Check both configs exist
    if [ ! -f "config.yml" ]; then
        echo "ERROR: config.yml not found!"
        exit 1
    fi
    
    if [ ! -f "config_aggressive.yml" ]; then
        echo "ERROR: config_aggressive.yml not found!"
        echo "Please create aggressive config first."
        exit 1
    fi
    
    # Use the dual bot startup script
    exec "./scripts/start_dual_bots.sh"
}

# If script called with argument, run directly
if [ $# -eq 1 ]; then
    case $1 in
        "conservative"|"1")
            start_single_bot "config.yml" "CONSERVATIVE"
            ;;
        "aggressive"|"2")
            start_single_bot "config_aggressive.yml" "AGGRESSIVE"
            ;;
        "dual"|"both"|"3")
            start_dual_bots
            ;;
        *)
            echo "Usage: $0 [conservative|aggressive|dual]"
            exit 1
            ;;
    esac
    exit 0
fi

# Interactive mode
while true; do
    show_menu
    read -r choice
    
    case $choice in
        1)
            start_single_bot "config.yml" "CONSERVATIVE"
            break
            ;;
        2)
            start_single_bot "config_aggressive.yml" "AGGRESSIVE"
            break
            ;;
        3)
            start_dual_bots
            break
            ;;
        4)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option. Please choose 1-4."
            echo ""
            ;;
    esac
done