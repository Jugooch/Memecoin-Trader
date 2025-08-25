#!/bin/bash
# Dual Bot Startup Script for Memecoin Trading Bot
# Runs both Conservative and Aggressive bots in separate tmux sessions

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

# Check configs
if [ ! -f "config.yml" ]; then
    echo "ERROR: config.yml not found!"
    exit 1
fi

if [ ! -f "config_aggressive.yml" ]; then
    echo "ERROR: config_aggressive.yml not found!"
    echo "Please run the setup to create aggressive config first."
    exit 1
fi

echo "🤖 Starting Dual Bot System..."
echo "=============================================="

# Function to start a bot in tmux
start_bot() {
    local bot_name=$1
    local config_file=$2
    local session_name="memecoin-${bot_name,,}"
    
    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # Start new tmux session
    echo "▶️  Starting $bot_name Bot in tmux session: $session_name"
    tmux new-session -d -s "$session_name" -c "$PROJECT_DIR" \
        "$PYTHON main.py --config $config_file --bot-name $bot_name"
    
    # Wait a moment for startup
    sleep 2
    
    # Check if session is still running
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "✅ $bot_name Bot started successfully"
    else
        echo "❌ Failed to start $bot_name Bot"
        return 1
    fi
}

# Start Conservative Bot
start_bot "CONSERVATIVE" "config.yml"

# Start Aggressive Bot  
start_bot "AGGRESSIVE" "config_aggressive.yml"

echo ""
echo "🎉 Both bots are now running!"
echo "=============================================="
echo "📱 Discord Channels:"
echo "   • Conservative: Original webhook channel"
echo "   • Aggressive: Aggressive webhook channel"
echo ""
echo "🔧 Management Commands:"
echo "   • List sessions:     tmux list-sessions"
echo "   • Attach Conservative: tmux attach -t memecoin-conservative"
echo "   • Attach Aggressive:   tmux attach -t memecoin-aggressive"
echo "   • Stop Conservative:   tmux kill-session -t memecoin-conservative"
echo "   • Stop Aggressive:     tmux kill-session -t memecoin-aggressive"
echo "   • Stop Both:          tmux kill-server"
echo ""
echo "📊 Log Files:"
echo "   • Conservative: logs/trading.log"
echo "   • Aggressive:   logs/trading_aggressive.log"
echo ""
echo "✨ Happy Trading!"