#!/usr/bin/env python3
"""
Startup script for the Aggressive Memecoin Trading Bot
Based on start_bot.py but uses aggressive config
"""

import asyncio
import signal
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from main import MemecoinTradingBot
from src.utils.monitoring import PerformanceMonitor, SystemMonitor
from src.utils.logger_setup import setup_logging
from src.utils.config_loader import load_config


class BotManager:
    def __init__(self):
        self.bot = None
        self.performance_monitor = None
        self.system_monitor = None
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown_event.set()
    
    async def start_bot(self, config_file: str = "config_aggressive.yml"):
        """Start the aggressive trading bot"""
        try:
            print("⚡ Starting AGGRESSIVE Memecoin Trading Bot...")
            print("=" * 50)
            
            # Initialize bot with aggressive config
            self.bot = MemecoinTradingBot(config_path=config_file)
            
            # Initialize monitoring
            self.performance_monitor = PerformanceMonitor()
            self.system_monitor = SystemMonitor()
            
            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self.performance_monitor.start()),
                asyncio.create_task(self.system_monitor.start())
            ]
            
            # Start bot
            bot_task = asyncio.create_task(self.bot.start())
            
            print("🚀 AGGRESSIVE Bot is now running!")
            print("📱 Discord: Aggressive webhook channel")
            print("📊 Log: logs/trading_aggressive.log")
            print("🎯 Strategy: Fast execution, single wallet triggers")
            print("=" * 50)
            
            # Wait for shutdown signal or bot completion
            done, pending = await asyncio.wait(
                [bot_task, asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Stop monitoring
            for task in monitoring_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            print(f"❌ Bot startup failed: {e}")
            raise
        finally:
            if self.bot:
                try:
                    await self.bot.stop()
                    print("✅ AGGRESSIVE Bot stopped cleanly")
                except Exception as e:
                    print(f"⚠️  Error during bot shutdown: {e}")


async def main():
    """Main entry point"""
    print("⚡ AGGRESSIVE Memecoin Trading Bot")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check aggressive config file exists
    try:
        config = load_config("config_aggressive.yml")
        config_file = "config_aggressive.yml"
        print(f"✅ Loaded aggressive config: {config_file}")
    except FileNotFoundError:
        print(f"❌ Aggressive config file not found")
        print("Please ensure config/config_aggressive.yml exists")
        sys.exit(1)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Start bot
    manager = BotManager()
    
    try:
        await manager.start_bot(config_file)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    
    print("👋 AGGRESSIVE Bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())