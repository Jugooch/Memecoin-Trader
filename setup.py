#!/usr/bin/env python3
"""
Setup script for Memecoin Trading Bot
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"OK - Python {sys.version.split()[0]} detected")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("OK - Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = ["logs", "data", "exports"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def setup_config():
    """Setup configuration file"""
    config_file = "config.yml"
    example_file = "config.example.yml"
    
    if os.path.exists(config_file):
        print(f"WARNING: {config_file} already exists, skipping...")
        return True
    
    if not os.path.exists(example_file):
        print(f"ERROR: {example_file} not found")
        return False
    
    try:
        shutil.copy(example_file, config_file)
        print(f"OK - Created {config_file} from example")
        print(f"Please edit {config_file} and add your API keys")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create config file: {e}")
        return False


def check_api_requirements():
    """Check API requirements"""
    print("\nAPI REQUIREMENTS:")
    print("=" * 30)
    print("You will need API keys from the following services:")
    print()
    print("1. Bitquery (bitquery.io)")
    print("   - GraphQL API for monitoring token launches")
    print("   - Free tier available")
    print()
    print("2. Moralis (moralis.io)")
    print("   - Solana API for token data and analytics")
    print("   - Free tier available")
    print()
    print("3. QuickNode (quicknode.com)")
    print("   - Solana RPC with Pump.fun marketplace add-on")
    print("   - Required for trade execution")
    print()
    print("Add these API keys to config.yml before starting the bot")
    print()


def display_next_steps():
    """Display next steps"""
    print("\nSETUP COMPLETE!")
    print("=" * 20)
    print("Next steps:")
    print()
    print("1. Edit config.yml and add your API keys")
    print("2. 🧪 Start with paper trading mode (paper_mode: true)")
    print("3. Monitor performance using the dashboard")
    print("4. Add proven alpha wallets to watched_wallets list")
    print()
    print("Commands:")
    print("  python start_bot.py    # Start the trading bot")
    print("  python dashboard.py    # View performance dashboard")
    print()
    print("WARNING: Test thoroughly with paper trading before going live!")


def main():
    """Main setup function"""
    print("🤖 MEMECOIN TRADING BOT SETUP")
    print("=" * 35)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup config
    if not setup_config():
        sys.exit(1)
    
    # Show API requirements
    check_api_requirements()
    
    # Display next steps
    display_next_steps()


if __name__ == "__main__":
    main()