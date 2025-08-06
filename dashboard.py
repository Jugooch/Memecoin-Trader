#!/usr/bin/env python3
"""
Simple CLI dashboard for monitoring bot performance
"""

import asyncio
import os
import time
from datetime import datetime
import json

from database import Database
from monitoring import PerformanceMonitor


class Dashboard:
    def __init__(self):
        self.db = Database()
        self.monitor = PerformanceMonitor(self.db)

    async def initialize(self):
        """Initialize database connection"""
        await self.db.initialize()

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    async def display_dashboard(self):
        """Display the main dashboard"""
        while True:
            self.clear_screen()
            
            print("MEMECOIN TRADING BOT DASHBOARD")
            print("=" * 50)
            print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Portfolio Summary
            await self.display_portfolio_summary()
            print()
            
            # Recent Trades
            await self.display_recent_trades()
            print()
            
            # Performance Metrics
            await self.display_performance_metrics()
            print()
            
            # Active Positions
            await self.display_active_positions()
            print()
            
            print("Refreshing in 30 seconds... (Ctrl+C to exit)")
            
            try:
                await asyncio.sleep(30)
            except KeyboardInterrupt:
                break

    async def display_portfolio_summary(self):
        """Display portfolio summary"""
        print("PORTFOLIO SUMMARY")
        print("-" * 20)
        
        try:
            summary = await self.monitor.get_daily_summary()
            
            print(f"Trades Today: {summary['trades_executed']}")
            print(f"Success Rate: {summary['trade_success_rate']:.1f}%")
            print(f"Today's Profit: ${summary['total_profit']:.2f}")
            print(f"API Calls: {summary['api_calls']} (Error Rate: {summary['api_error_rate']:.1f}%)")
            print(f"Tokens Analyzed: {summary['tokens_analyzed']}")
            print(f"Alpha Signals: {summary['alpha_signals']} ({summary['alpha_signal_rate']:.1f}%)")
            print(f"⏱️  Uptime: {summary['uptime_hours']:.1f} hours")
            
        except Exception as e:
            print(f"ERROR: Error loading portfolio: {e}")

    async def display_recent_trades(self):
        """Display recent trades"""
        print("RECENT TRADES (Last 10)")
        print("-" * 25)
        
        try:
            trades = await self.db.get_trade_history(limit=10)
            
            if not trades:
                print("No trades found")
                return
            
            for trade in trades:
                timestamp = trade['timestamp'][:19] if trade['timestamp'] else 'Unknown'
                action = trade['action']
                mint = trade['mint'][:8] + '...' if trade['mint'] else 'Unknown'
                profit = trade.get('profit', 0)
                profit_pct = trade.get('profit_pct', 0)
                
                profit_str = f"${profit:.2f}" if profit else "Pending"
                pct_str = f"({profit_pct:.1f}%)" if profit_pct else ""
                
                status = "PROFIT" if profit and profit > 0 else "LOSS" if profit and profit < 0 else "PENDING"
                
                print(f"{status} {timestamp} | {action:4} | {mint} | {profit_str} {pct_str}")
                
        except Exception as e:
            print(f"ERROR: Error loading trades: {e}")

    async def display_performance_metrics(self):
        """Display performance metrics"""
        print("PERFORMANCE METRICS (Last 7 days)")
        print("-" * 35)
        
        try:
            report = await self.monitor.generate_performance_report(7)
            
            print(f"Total Trades: {report.get('total_trades', 0)}")
            print(f"Profitable Trades: {report.get('profitable_trades', 0)}")
            print(f"Win Rate: {report.get('win_rate', 0):.1f}%")
            print(f"Total Profit: ${report.get('total_profit', 0):.2f}")
            print(f"Avg Profit/Trade: ${report.get('avg_profit_per_trade', 0):.2f}")
            print(f"Max Drawdown: {report.get('max_drawdown', 0):.1f}%")
            
        except Exception as e:
            print(f"ERROR: Error loading performance: {e}")

    async def display_active_positions(self):
        """Display active positions"""
        print("ACTIVE POSITIONS")
        print("-" * 18)
        
        try:
            positions = await self.db.get_active_positions()
            
            if not positions:
                print("No active positions")
                return
            
            for pos in positions:
                mint = pos['mint'][:8] + '...' if pos['mint'] else 'Unknown'
                entry_price = pos.get('entry_price', 0)
                amount = pos.get('current_amount', 0)
                sol_invested = pos.get('sol_invested', 0)
                
                print(f"TOKEN {mint} | Entry: ${entry_price:.6f} | Amount: {amount:.0f} | Invested: {sol_invested:.3f} SOL")
                
        except Exception as e:
            print(f"ERROR: Error loading positions: {e}")

    async def show_detailed_report(self):
        """Show detailed performance report"""
        self.clear_screen()
        print("DETAILED PERFORMANCE REPORT")
        print("=" * 40)
        
        try:
            report = await self.monitor.generate_performance_report(30)
            
            print(f"Report Period: {report.get('report_period_days', 0)} days")
            print(f"Total Trades: {report.get('total_trades', 0)}")
            print(f"Profitable Trades: {report.get('profitable_trades', 0)}")
            print(f"Win Rate: {report.get('win_rate', 0):.2f}%")
            print(f"Total Profit: ${report.get('total_profit', 0):.2f}")
            print(f"Average Profit per Trade: ${report.get('avg_profit_per_trade', 0):.2f}")
            print(f"Max Drawdown: {report.get('max_drawdown', 0):.2f}%")
            print()
            
            # Daily summary
            daily = report.get('daily_summary', {})
            print("TODAY'S SUMMARY:")
            print(f"  Uptime: {daily.get('uptime_hours', 0):.1f} hours")
            print(f"  Trades: {daily.get('trades_executed', 0)}")
            print(f"  Success Rate: {daily.get('trade_success_rate', 0):.1f}%")
            print(f"  Profit: ${daily.get('total_profit', 0):.2f}")
            print(f"  API Calls: {daily.get('api_calls', 0)}")
            print(f"  Tokens Analyzed: {daily.get('tokens_analyzed', 0)}")
            print()
            
            # Performance history
            history = report.get('performance_history', [])
            if history:
                print("RECENT DAILY PERFORMANCE:")
                for day in history[:7]:  # Last 7 days
                    date = day.get('date', 'Unknown')
                    trades = day.get('total_trades', 0)
                    profit = day.get('total_profit', 0)
                    win_rate = day.get('win_rate', 0)
                    print(f"  {date}: {trades} trades, ${profit:.2f}, {win_rate:.1f}% win rate")
            
        except Exception as e:
            print(f"ERROR: Error generating report: {e}")
        
        print("\nPress Enter to return to dashboard...")
        input()

    async def export_data(self):
        """Export performance data"""
        try:
            filename = f"performance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            await self.monitor.export_metrics(filename)
            print(f"OK - Data exported to {filename}")
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
        
        input("Press Enter to continue...")

    async def run_interactive(self):
        """Run interactive dashboard"""
        await self.initialize()
        
        while True:
            self.clear_screen()
            print("MEMECOIN BOT DASHBOARD")
            print("=" * 30)
            print("1. Live Dashboard")
            print("2. Detailed Report")
            print("3. Export Data")
            print("4. Exit")
            print()
            
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                await self.display_dashboard()
            elif choice == '2':
                await self.show_detailed_report()
            elif choice == '3':
                await self.export_data()
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")
                time.sleep(1)


async def main():
    """Main entry point"""
    dashboard = Dashboard()
    try:
        await dashboard.run_interactive()
    except KeyboardInterrupt:
        print("\nDashboard closed")
    except Exception as e:
        print(f"ERROR: Dashboard error: {e}")


if __name__ == "__main__":
    asyncio.run(main())