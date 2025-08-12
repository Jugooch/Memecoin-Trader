#!/usr/bin/env python3
"""
P&L Reset Script
Resets paper trading statistics and daily summaries for fresh testing

Usage: python reset_pnl.py
"""

import os
import sys
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path

# Add parent directory to path  
sys.path.append(str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.pnl_store import PnLStore


def reset_paper_trading_stats():
    """Reset all paper trading statistics"""
    print("🔄 Resetting Paper Trading Statistics...")
    
    try:
        # Load config
        config = load_config('config.yml')
        initial_capital = config.get('trading', {}).get('initial_capital', 500.0)
        
        print(f"💰 Reset Capital: ${initial_capital}")
        
        # Reset operations
        print("\n🗑️ Clearing data...")
        
        # Clear P&L store JSON files
        pnl_files = [
            'data/pnl_state.json',    # Main P&L store
            'pnl_data.json',          # Legacy/alternative locations
            'data/pnl_data.json',
            'logs/pnl_data.json'
        ]
        
        for pnl_file in pnl_files:
            if os.path.exists(pnl_file):
                os.remove(pnl_file)
                print(f"   ✅ Removed {pnl_file}")
        
        # Create fresh P&L store with reset capital
        pnl_store = PnLStore("data/pnl_state.json", initial_capital)
        print("   ✅ Created fresh P&L store")
        
        # Clear any cached data files
        cache_files = [
            'data/token_cache.json', 
            'data/alpha_cache.json',
            'logs/daily_summary.json',
            'data/wallet_performance.json',
            'data/trade_analytics.json'
        ]
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"   ✅ Removed {cache_file}")
        
        print(f"\n✅ Paper trading reset complete!")
        print(f"📈 Starting fresh with ${initial_capital} capital")
        print(f"🕐 Reset timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show reset summary
        summary = pnl_store.get_summary()
        print(f"\n📊 Post-reset verification:")
        print(f"   Starting Capital: ${summary['starting_capital']}")
        print(f"   Current Equity: ${summary['equity']}")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Active Positions: {summary['active_positions']}")
        
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return False
    
    return True


def reset_log_files():
    """Clear relevant log files"""
    print("\n📝 Clearing log files...")
    
    log_files = [
        'logs/trading.log',
        'logs/alpha_discovery.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            # Keep the file but clear contents
            with open(log_file, 'w') as f:
                f.write(f"# Log cleared on {datetime.now().isoformat()}\n")
            print(f"   ✅ Cleared {log_file}")
    
    print("   ✅ Log files cleared")


def main():
    """Main reset function"""
    print("=" * 50)
    print("🔄 PAPER TRADING RESET UTILITY")
    print("=" * 50)
    
    # Confirm reset
    response = input("\n⚠️  This will clear ALL paper trading data. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Reset cancelled")
        return
    
    # Reset P&L data
    success = reset_paper_trading_stats()
    
    if success:
        # Reset logs
        reset_log_files()
        
        print("\n" + "=" * 50)
        print("✅ RESET COMPLETE - READY FOR FRESH TESTING")
        print("=" * 50)
        print("\n🚀 You can now start the bot with clean stats:")
        print("   python start_bot.py")
        print("\n📊 Expected trade sizes with new config:")
        print("   $500 capital × 5% = $25 max trade size")
        print("   Realistic fees: ~$0.34 per $20 trade (1.7%)")
    else:
        print("\n❌ Reset failed - check error messages above")


if __name__ == "__main__":
    main()