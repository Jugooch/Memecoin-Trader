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

from src.utils.config_loader import load_config, get_database_path
from src.core.database import Database
from src.utils.pnl_store import PnLStore


async def reset_paper_trading_stats():
    """Reset all paper trading statistics"""
    print("🔄 Resetting Paper Trading Statistics...")
    
    try:
        # Load config
        config = load_config('config.yml')
        db_path = get_database_path(config)
        initial_capital = config.get('trading', {}).get('initial_capital', 500.0)
        
        print(f"📊 Database: {db_path}")
        print(f"💰 Reset Capital: ${initial_capital}")
        
        # Initialize database
        database = Database(db_path)
        await database.initialize()
        
        # Reset operations
        print("\n🗑️ Clearing data...")
        
        # Clear all paper trading trades
        async with database.get_connection() as db:
            # Clear paper trades
            await db.execute("DELETE FROM trades WHERE paper_mode = 1")
            print("   ✅ Cleared paper trading trades")
            
            # Clear positions (they'll be recreated)
            await db.execute("DELETE FROM positions WHERE paper_mode = 1") 
            print("   ✅ Cleared paper trading positions")
            
            # Clear performance metrics
            await db.execute("DELETE FROM performance_metrics WHERE 1=1")
            print("   ✅ Cleared performance metrics")
            
            # Reset paper trading stats in the bot (if table exists)
            try:
                await db.execute("""
                    UPDATE bot_stats 
                    SET paper_capital = ?, daily_pnl = 0, total_trades = 0 
                    WHERE 1=1
                """, (initial_capital,))
                print("   ✅ Reset bot statistics")
            except:
                # Table might not exist yet
                pass
            
            await db.commit()
        
        # Clear P&L store files if they exist
        pnl_files = ['pnl_data.json', 'data/pnl_data.json', 'logs/pnl_data.json']
        for pnl_file in pnl_files:
            if os.path.exists(pnl_file):
                os.remove(pnl_file)
                print(f"   ✅ Removed {pnl_file}")
        
        # Clear any cached data files
        cache_files = [
            'data/token_cache.json', 
            'data/alpha_cache.json',
            'logs/daily_summary.json'
        ]
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"   ✅ Removed {cache_file}")
        
        print(f"\n✅ Paper trading reset complete!")
        print(f"📈 Starting fresh with ${initial_capital} capital")
        print(f"🕐 Reset timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show reset summary
        async with database.get_connection() as db:
            # Count remaining trades (should be 0)
            cursor = await db.execute("SELECT COUNT(*) FROM trades WHERE paper_mode = 1")
            trade_count = (await cursor.fetchone())[0]
            
            cursor = await db.execute("SELECT COUNT(*) FROM positions WHERE paper_mode = 1") 
            position_count = (await cursor.fetchone())[0]
            
            print(f"\n📊 Post-reset verification:")
            print(f"   Paper Trades: {trade_count}")
            print(f"   Active Positions: {position_count}")
        
        await database.close()
        
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        print(f"   This is normal if database doesn't exist yet")
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


async def main():
    """Main reset function"""
    print("=" * 50)
    print("🔄 PAPER TRADING RESET UTILITY")
    print("=" * 50)
    
    # Confirm reset
    response = input("\n⚠️  This will clear ALL paper trading data. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Reset cancelled")
        return
    
    # Reset database
    success = await reset_paper_trading_stats()
    
    if success:
        # Reset logs
        reset_log_files()
        
        print("\n" + "=" * 50)
        print("✅ RESET COMPLETE - READY FOR FRESH TESTING")
        print("=" * 50)
        print("\n🚀 You can now start the bot with clean stats:")
        print("   python start_bot.py")
    else:
        print("\n❌ Reset failed - check error messages above")


if __name__ == "__main__":
    asyncio.run(main())