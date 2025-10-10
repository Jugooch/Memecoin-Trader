#!/bin/bash
# Start Auto-Update Watchlist Scheduler (Linux/Mac)
#
# Usage:
#   # Run in foreground
#   ./start_auto_update.sh
#
#   # Run in background
#   nohup ./start_auto_update.sh > ../logs/auto_update.log 2>&1 &
#
#   # View logs
#   tail -f ../logs/auto_update.log

echo "===================================="
echo "  Dev Watchlist Auto-Updater"
echo "===================================="
echo ""
echo "This will update your dev watchlist every 8 hours"
echo "Press Ctrl+C to stop"
echo ""

# Activate virtual environment if it exists
if [ -f "../bot_env/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../bot_env/bin/activate
fi

# Create logs directory if it doesn't exist
mkdir -p ../logs

# Run the auto-updater
echo "Starting auto-updater..."
echo ""
python auto_update_watchlist.py --interval 8 --max-pages 500 --min-buy-rate 70
