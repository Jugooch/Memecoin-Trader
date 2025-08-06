#!/bin/bash
# Install systemd service for automatic startup on Oracle Cloud VM

set -e

echo "Installing Memecoin Trading Bot as system service..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
   echo "Please run as root or with sudo"
   exit 1
fi

# Get the actual user (not root if using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Update service file with correct paths and user
sed -i "s|User=ubuntu|User=$ACTUAL_USER|g" scripts/memecoin-bot.service
sed -i "s|/home/ubuntu/memecoin-trader|$PROJECT_DIR|g" scripts/memecoin-bot.service

# Copy service file to systemd
cp scripts/memecoin-bot.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable service to start on boot
systemctl enable memecoin-bot.service

echo "Service installed successfully!"
echo ""
echo "To start the bot: sudo systemctl start memecoin-bot"
echo "To stop the bot: sudo systemctl stop memecoin-bot"
echo "To check status: sudo systemctl status memecoin-bot"
echo "To view logs: sudo journalctl -u memecoin-bot -f"
echo ""
echo "The bot will automatically start on system boot."