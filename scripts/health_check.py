#!/usr/bin/env python3
"""
Health check script to monitor bot status
Can be used with monitoring systems or cron jobs
"""

import sys
import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_bot_health():
    """Check if bot is healthy"""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'healthy': True,
        'checks': {}
    }
    
    # Check 1: Database exists and is accessible
    try:
        db_path = 'trades.db'
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            trade_count = cursor.fetchone()[0]
            conn.close()
            
            health_status['checks']['database'] = {
                'status': 'OK',
                'total_trades': trade_count
            }
        else:
            health_status['checks']['database'] = {
                'status': 'ERROR',
                'message': 'Database not found'
            }
            health_status['healthy'] = False
    except Exception as e:
        health_status['checks']['database'] = {
            'status': 'ERROR',
            'message': str(e)
        }
        health_status['healthy'] = False
    
    # Check 2: Recent trades (bot is active)
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for trades in last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp > ?",
                (yesterday,)
            )
            recent_trades = cursor.fetchone()[0]
            conn.close()
            
            health_status['checks']['activity'] = {
                'status': 'OK' if recent_trades > 0 else 'WARNING',
                'recent_trades_24h': recent_trades
            }
            
            if recent_trades == 0:
                health_status['healthy'] = False
        else:
            health_status['checks']['activity'] = {
                'status': 'ERROR',
                'message': 'Cannot check activity'
            }
    except Exception as e:
        health_status['checks']['activity'] = {
            'status': 'ERROR',
            'message': str(e)
        }
    
    # Check 3: Log files exist and are recent
    try:
        log_file = 'logs/trading.log'
        if os.path.exists(log_file):
            # Check if log was modified in last hour
            mtime = os.path.getmtime(log_file)
            age_minutes = (datetime.now().timestamp() - mtime) / 60
            
            health_status['checks']['logging'] = {
                'status': 'OK' if age_minutes < 60 else 'WARNING',
                'last_log_minutes_ago': round(age_minutes, 1)
            }
            
            if age_minutes > 120:  # No logs for 2 hours
                health_status['healthy'] = False
        else:
            health_status['checks']['logging'] = {
                'status': 'WARNING',
                'message': 'Log file not found'
            }
    except Exception as e:
        health_status['checks']['logging'] = {
            'status': 'ERROR',
            'message': str(e)
        }
    
    # Check 4: Config file exists
    try:
        config_paths = ['config/config.yml', 'config.yml']
        config_found = any(os.path.exists(p) for p in config_paths)
        
        health_status['checks']['config'] = {
            'status': 'OK' if config_found else 'ERROR',
            'config_found': config_found
        }
        
        if not config_found:
            health_status['healthy'] = False
    except Exception as e:
        health_status['checks']['config'] = {
            'status': 'ERROR',
            'message': str(e)
        }
    
    return health_status


def main():
    """Main health check"""
    health = check_bot_health()
    
    # Print JSON output
    print(json.dumps(health, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if health['healthy'] else 1)


if __name__ == "__main__":
    main()