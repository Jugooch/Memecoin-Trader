#!/usr/bin/env python3
"""
Dual Bot Startup Script
Runs both conservative and aggressive trading bots simultaneously for A/B testing
"""

import asyncio
import subprocess
import signal
import sys
import time
from pathlib import Path

class DualBotManager:
    def __init__(self):
        self.conservative_process = None
        self.aggressive_process = None
        self.running = False
        
    async def start_bots(self):
        """Start both bots simultaneously"""
        print("🚀 Starting Dual Bot Trading System...")
        print("=" * 60)
        
        try:
            # Start Conservative Bot
            print("📊 Starting CONSERVATIVE Bot...")
            self.conservative_process = await asyncio.create_subprocess_exec(
                "python", "main.py", 
                "--config", "config.yml", 
                "--bot-name", "CONSERVATIVE",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Give it a moment to start
            await asyncio.sleep(2)
            
            # Start Aggressive Bot
            print("⚡ Starting AGGRESSIVE Bot...")
            self.aggressive_process = await asyncio.create_subprocess_exec(
                "python", "main.py", 
                "--config", "config_aggressive.yml", 
                "--bot-name", "AGGRESSIVE",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            print("✅ Both bots started successfully!")
            print("📱 Check your Discord channels for trade notifications:")
            print("   • Conservative: Original webhook")
            print("   • Aggressive: Aggressive webhook")
            print("")
            print("💡 Tips:")
            print("   • Use 'tmux list-sessions' to see running sessions")
            print("   • Press Ctrl+C to stop both bots gracefully")
            print("   • Check logs/trading.log and logs/trading_aggressive.log")
            print("=" * 60)
            
            self.running = True
            
            # Wait for both processes
            await self._monitor_processes()
            
        except Exception as e:
            print(f"❌ Error starting bots: {e}")
            await self.stop_bots()
    
    async def _monitor_processes(self):
        """Monitor both bot processes and handle their lifecycle"""
        while self.running:
            try:
                # Check if processes are still running
                conservative_running = (self.conservative_process and 
                                      self.conservative_process.returncode is None)
                aggressive_running = (self.aggressive_process and 
                                    self.aggressive_process.returncode is None)
                
                if not conservative_running and not aggressive_running:
                    print("⚠️  Both bots have stopped")
                    break
                elif not conservative_running:
                    print("⚠️  Conservative bot has stopped")
                elif not aggressive_running:
                    print("⚠️  Aggressive bot has stopped")
                
                # Wait a bit before checking again
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"❌ Error monitoring processes: {e}")
                break
        
        await self.stop_bots()
    
    async def stop_bots(self):
        """Stop both bots gracefully"""
        print("🛑 Stopping bots...")
        self.running = False
        
        # Stop conservative bot
        if self.conservative_process:
            try:
                self.conservative_process.terminate()
                await asyncio.wait_for(self.conservative_process.wait(), timeout=10)
                print("✅ Conservative bot stopped")
            except asyncio.TimeoutError:
                print("⚠️  Force killing conservative bot...")
                self.conservative_process.kill()
            except Exception as e:
                print(f"❌ Error stopping conservative bot: {e}")
        
        # Stop aggressive bot
        if self.aggressive_process:
            try:
                self.aggressive_process.terminate()
                await asyncio.wait_for(self.aggressive_process.wait(), timeout=10)
                print("✅ Aggressive bot stopped")
            except asyncio.TimeoutError:
                print("⚠️  Force killing aggressive bot...")
                self.aggressive_process.kill()
            except Exception as e:
                print(f"❌ Error stopping aggressive bot: {e}")
        
        print("🏁 Both bots stopped")


# Alternative single-process approach for better resource sharing
class SingleProcessDualBot:
    """Run both bot strategies in a single process to share resources"""
    
    def __init__(self):
        self.conservative_bot = None
        self.aggressive_bot = None
        self.running = False
        
    async def start_both(self):
        """Start both bots in the same process"""
        print("🚀 Starting Dual Bot System (Single Process Mode)...")
        print("=" * 60)
        
        try:
            # Import here to avoid circular imports
            from main import MemecoinTradingBot
            
            print("📊 Initializing CONSERVATIVE Bot...")
            self.conservative_bot = MemecoinTradingBot(config_path="config.yml")
            
            print("⚡ Initializing AGGRESSIVE Bot...")  
            self.aggressive_bot = MemecoinTradingBot(config_path="config_aggressive.yml")
            
            print("✅ Both bots initialized successfully!")
            print("")
            print("🔄 Starting trading loops...")
            
            self.running = True
            
            # Run both bots concurrently
            await asyncio.gather(
                self._run_bot(self.conservative_bot, "CONSERVATIVE"),
                self._run_bot(self.aggressive_bot, "AGGRESSIVE"),
                return_exceptions=True
            )
            
        except Exception as e:
            print(f"❌ Error in dual bot system: {e}")
            await self.stop_both()
    
    async def _run_bot(self, bot, name):
        """Run a single bot with error handling"""
        try:
            print(f"▶️  Starting {name} bot...")
            await bot.start()
        except Exception as e:
            print(f"❌ {name} bot error: {e}")
            raise
    
    async def stop_both(self):
        """Stop both bots gracefully"""
        print("🛑 Stopping both bots...")
        self.running = False
        
        tasks = []
        
        if self.conservative_bot:
            tasks.append(self._stop_bot(self.conservative_bot, "CONSERVATIVE"))
            
        if self.aggressive_bot:
            tasks.append(self._stop_bot(self.aggressive_bot, "AGGRESSIVE"))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        print("🏁 Both bots stopped")
    
    async def _stop_bot(self, bot, name):
        """Stop a single bot with error handling"""
        try:
            await bot.stop()
            print(f"✅ {name} bot stopped")
        except Exception as e:
            print(f"❌ Error stopping {name} bot: {e}")


async def main():
    """Main entry point - choose execution mode"""
    
    # Check if config files exist
    if not Path("config.yml").exists():
        print("❌ config.yml not found! Please create it from config.yml.example")
        return
        
    if not Path("config_aggressive.yml").exists():
        print("❌ config_aggressive.yml not found! Please run the setup first.")
        return
    
    print("🤖 Memecoin Dual Bot System")
    print("=" * 40)
    print("Choose execution mode:")
    print("1. Separate Processes (better isolation)")
    print("2. Single Process (better resource sharing)")
    print("3. Conservative Only")
    print("4. Aggressive Only")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    # Set up signal handlers
    manager = None
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, stopping...")
        if manager:
            asyncio.create_task(manager.stop_bots() if hasattr(manager, 'stop_bots') else manager.stop_both())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if choice == "1":
            manager = DualBotManager()
            await manager.start_bots()
        elif choice == "2":
            manager = SingleProcessDualBot()
            await manager.start_both()
        elif choice == "3":
            print("📊 Starting CONSERVATIVE Bot only...")
            process = await asyncio.create_subprocess_exec(
                "python", "main.py", "--config", "config.yml"
            )
            await process.wait()
        elif choice == "4":
            print("⚡ Starting AGGRESSIVE Bot only...")
            process = await asyncio.create_subprocess_exec(
                "python", "main.py", "--config", "config_aggressive.yml"
            )
            await process.wait()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if manager:
            await (manager.stop_bots() if hasattr(manager, 'stop_bots') else manager.stop_both())


if __name__ == "__main__":
    asyncio.run(main())