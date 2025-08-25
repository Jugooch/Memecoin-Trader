#!/bin/bash
# Stop script for Memecoin Trading Bots

echo "🛑 Stopping Memecoin Trading Bots..."
echo "===================================="

# Function to stop a specific bot
stop_bot() {
    local bot_name=$1
    local session_name="memecoin-${bot_name,,}"
    
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "🔴 Stopping $bot_name Bot..."
        tmux kill-session -t "$session_name"
        echo "✅ $bot_name Bot stopped"
    else
        echo "ℹ️  $bot_name Bot is not running"
    fi
}

# Show menu if no arguments
if [ $# -eq 0 ]; then
    echo "Which bots would you like to stop?"
    echo "1. Conservative Bot Only"
    echo "2. Aggressive Bot Only"
    echo "3. Both Bots"
    echo "4. All tmux sessions (nuclear option)"
    echo ""
    echo -n "Choose option (1-4): "
    read -r choice
    
    case $choice in
        1)
            stop_bot "CONSERVATIVE"
            ;;
        2)
            stop_bot "AGGRESSIVE"
            ;;
        3)
            stop_bot "CONSERVATIVE"
            stop_bot "AGGRESSIVE"
            ;;
        4)
            echo "💥 Killing all tmux sessions..."
            tmux kill-server 2>/dev/null || echo "ℹ️  No tmux sessions to kill"
            ;;
        *)
            echo "❌ Invalid option"
            exit 1
            ;;
    esac
else
    # Handle command line arguments
    case $1 in
        "conservative")
            stop_bot "CONSERVATIVE"
            ;;
        "aggressive")
            stop_bot "AGGRESSIVE"
            ;;
        "both"|"all")
            stop_bot "CONSERVATIVE"
            stop_bot "AGGRESSIVE"
            ;;
        *)
            echo "Usage: $0 [conservative|aggressive|both]"
            exit 1
            ;;
    esac
fi

echo ""
echo "🔍 Remaining tmux sessions:"
tmux list-sessions 2>/dev/null || echo "   No active sessions"