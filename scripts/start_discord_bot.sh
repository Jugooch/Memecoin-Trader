#!/bin/bash

# Start Discord Wallet Proof Bot
# Completely isolated from main trading bot

echo "🤖 Starting Discord Wallet Proof Bot..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "bot_env" ]; then
    source bot_env/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the Discord bot
python -m src.services.discord_wallet_bot

echo "Discord bot stopped"