#!/usr/bin/env python3
"""
Startup script for the Memecoin Trading Bot
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
        self.running = False
        
    async def start_bot(self, config_file: str = "config/config.yml"):
        """Start the trading bot with monitoring"""
        print("Starting Memecoin Trading Bot...")
        
        try:
            # Initialize bot
            self.bot = MemecoinTradingBot(config_file)
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start monitoring
            monitor = PerformanceMonitor(self.bot.database)
            system_monitor = SystemMonitor()
            
            # Start monitoring tasks
            monitoring_tasks = [
                self._periodic_health_check(system_monitor),
                self._periodic_metrics_save(monitor),
                self._periodic_alerts_check(monitor)
            ]
            
            print("Bot initialized successfully")
            print(f"Paper mode: {self.bot.config.paper_mode} | Capital: ${self.bot.config.initial_capital} | Max trades/day: {self.bot.config.max_trades_per_day}")
            print("Starting monitoring (detailed logs in logs/trading.log)...")
            
            self.running = True
            
            # Start bot and monitoring
            await asyncio.gather(
                self.bot.start(),
                *monitoring_tasks,
                return_exceptions=True
            )
            
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
            await self.shutdown()
        except Exception as e:
            print(f"Bot startup failed: {e}")
            sys.exit(1)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}")
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """Gracefully shutdown the bot"""
        print("Shutting down bot...")
        self.running = False
        
        if self.bot:
            await self.bot.stop()
            
            # Close all positions in paper mode
            if self.bot.config.paper_mode:
                print("Closing all paper positions...")
                await self.bot.trading_engine.close_all_positions()
            
            # Generate final report
            print("Generating final performance report...")
            monitor = PerformanceMonitor(self.bot.database)
            report = await monitor.generate_performance_report(7)
            
            print("\n" + "="*50)
            print("FINAL PERFORMANCE REPORT")
            print("="*50)
            print(f"Total Trades: {report.get('total_trades', 0)}")
            print(f"Profitable Trades: {report.get('profitable_trades', 0)}")
            print(f"Win Rate: {report.get('win_rate', 0):.1f}%")
            print(f"Total Profit: ${report.get('total_profit', 0):.2f}")
            print(f"Max Drawdown: {report.get('max_drawdown', 0):.1f}%")
            print("="*50)
            
            # Cleanup
            await self.bot.trading_engine.cleanup()
            
        print("Bot shutdown complete")
        sys.exit(0)

    async def _periodic_health_check(self, system_monitor: SystemMonitor):
        """Periodic system health checks"""
        while self.running:
            try:
                health = await system_monitor.check_system_health()
                
                # Log warnings for high resource usage
                if health['memory_used_pct'] > 80:
                    print(f"WARNING: High memory usage: {health['memory_used_pct']:.1f}%")
                
                if health['cpu_used_pct'] > 80:
                    print(f"WARNING: High CPU usage: {health['cpu_used_pct']:.1f}%")
                
                # Check network connectivity (only show if critical services are down)
                connectivity = await system_monitor.check_network_connectivity()
                critical_offline = [name for name, status in connectivity.items() 
                                  if status['status'] == 'offline' and name in ['bitquery', 'quicknode']]
                
                if critical_offline:
                    print(f"Critical services offline: {', '.join(critical_offline)}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"Health check error: {e}")
                await asyncio.sleep(60)

    async def _periodic_metrics_save(self, monitor: PerformanceMonitor):
        """Periodic metrics saving"""
        while self.running:
            try:
                await monitor.save_daily_metrics()
                await asyncio.sleep(3600)  # Save every hour
            except Exception as e:
                print(f"Metrics save error: {e}")
                await asyncio.sleep(300)

    async def _periodic_alerts_check(self, monitor: PerformanceMonitor):
        """Periodic alerts checking"""
        while self.running:
            try:
                alerts = await monitor.check_performance_alerts()
                
                for alert in alerts:
                    icon = "[CRITICAL]" if alert['type'] == 'critical' else "[WARNING]" if alert['type'] == 'warning' else "[INFO]"
                    print(f"{icon} {alert['message']}")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                print(f"Alerts check error: {e}")
                await asyncio.sleep(300)


def main():
    """Main entry point"""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check config file exists using shared loader
    try:
        config = load_config("config/config.yml")
        config_file = "config/config.yml"  # Will be found by shared loader
    except FileNotFoundError:
        print(f"Config file not found")
        print("Please copy config/config.yml.example to config/config.yml and configure your API keys")
        sys.exit(1)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Start bot
    manager = BotManager()
    
    try:
        asyncio.run(manager.start_bot(config_file))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()