"""
Database module for tracking trades, PnL, and bot performance
"""

import sqlite3
import asyncio
import aiosqlite
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class Database:
    def __init__(self, db_file: str = "trades.db"):
        self.db_file = db_file
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize database tables"""
        async with aiosqlite.connect(self.db_file) as db:
            # Trades table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mint TEXT NOT NULL,
                    action TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    sol_amount REAL,
                    tokens_amount REAL,
                    timestamp DATETIME NOT NULL,
                    paper_mode BOOLEAN NOT NULL,
                    profit REAL,
                    profit_pct REAL,
                    exit_reason TEXT,
                    signature TEXT,
                    metadata TEXT
                )
            """)
            
            # Positions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mint TEXT UNIQUE NOT NULL,
                    entry_price REAL NOT NULL,
                    current_amount REAL NOT NULL,
                    sol_invested REAL NOT NULL,
                    entry_time DATETIME NOT NULL,
                    tp_price REAL,
                    sl_price REAL,
                    peak_price REAL,
                    paper_mode BOOLEAN NOT NULL,
                    status TEXT DEFAULT 'ACTIVE',
                    metadata TEXT
                )
            """)
            
            # Performance tracking
            await db.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    paper_capital REAL,
                    portfolio_value REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    metadata TEXT
                )
            """)
            
            # Alpha wallets tracking
            await db.execute("""
                CREATE TABLE IF NOT EXISTS alpha_wallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_address TEXT UNIQUE NOT NULL,
                    win_rate REAL,
                    avg_hold_time REAL,
                    avg_profit_pct REAL,
                    total_trades INTEGER DEFAULT 0,
                    last_updated DATETIME,
                    performance_score REAL,
                    metadata TEXT
                )
            """)
            
            # Token analysis
            await db.execute("""
                CREATE TABLE IF NOT EXISTS token_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mint TEXT UNIQUE NOT NULL,
                    name TEXT,
                    symbol TEXT,
                    deployer TEXT,
                    launch_time DATETIME,
                    initial_liquidity REAL,
                    max_liquidity REAL,
                    volume_24h REAL,
                    alpha_buyers INTEGER DEFAULT 0,
                    traded BOOLEAN DEFAULT FALSE,
                    performance REAL,
                    metadata TEXT
                )
            """)
            
            await db.commit()
            
        self.logger.info("Database initialized successfully")

    async def record_trade(self, trade_data: Dict):
        """Record a trade in the database"""
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute("""
                INSERT INTO trades (
                    mint, action, amount, price, sol_amount, tokens_amount,
                    timestamp, paper_mode, profit, profit_pct, exit_reason,
                    signature, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['mint'],
                trade_data['action'],
                trade_data['amount'],
                trade_data['price'],
                trade_data.get('sol_amount'),
                trade_data.get('tokens_amount'),
                trade_data['timestamp'],
                trade_data['paper_mode'],
                trade_data.get('profit'),
                trade_data.get('profit_pct'),
                trade_data.get('exit_reason'),
                trade_data.get('signature'),
                json.dumps(trade_data.get('metadata', {}))
            ))
            await db.commit()
            
        self.logger.info(f"Recorded {trade_data['action']} trade for {trade_data['mint']}")

    async def update_position(self, mint: str, position_data: Dict):
        """Update or insert position data"""
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute("""
                INSERT OR REPLACE INTO positions (
                    mint, entry_price, current_amount, sol_invested,
                    entry_time, tp_price, sl_price, peak_price,
                    paper_mode, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mint,
                position_data['entry_price'],
                position_data['current_amount'],
                position_data['sol_invested'],
                position_data['entry_time'],
                position_data.get('tp_price'),
                position_data.get('sl_price'),
                position_data.get('peak_price'),
                position_data['paper_mode'],
                position_data.get('status', 'ACTIVE'),
                json.dumps(position_data.get('metadata', {}))
            ))
            await db.commit()

    async def close_position(self, mint: str):
        """Mark position as closed"""
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute(
                "UPDATE positions SET status = 'CLOSED' WHERE mint = ?",
                (mint,)
            )
            await db.commit()

    async def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        async with aiosqlite.connect(self.db_file) as db:
            async with db.execute(
                "SELECT * FROM positions WHERE status = 'ACTIVE'"
            ) as cursor:
                rows = await cursor.fetchall()
                
                positions = []
                for row in rows:
                    positions.append({
                        'mint': row[1],
                        'entry_price': row[2],
                        'current_amount': row[3],
                        'sol_invested': row[4],
                        'entry_time': row[5],
                        'tp_price': row[6],
                        'sl_price': row[7],
                        'peak_price': row[8],
                        'paper_mode': row[9],
                        'status': row[10],
                        'metadata': json.loads(row[11] or '{}')
                    })
                
                return positions

    async def get_trade_history(self, limit: int = 100, mint: str = None) -> List[Dict]:
        """Get trade history"""
        async with aiosqlite.connect(self.db_file) as db:
            if mint:
                query = "SELECT * FROM trades WHERE mint = ? ORDER BY timestamp DESC LIMIT ?"
                params = (mint, limit)
            else:
                query = "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?"
                params = (limit,)
                
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                trades = []
                for row in rows:
                    trades.append({
                        'id': row[0],
                        'mint': row[1],
                        'action': row[2],
                        'amount': row[3],
                        'price': row[4],
                        'sol_amount': row[5],
                        'tokens_amount': row[6],
                        'timestamp': row[7],
                        'paper_mode': row[8],
                        'profit': row[9],
                        'profit_pct': row[10],
                        'exit_reason': row[11],
                        'signature': row[12],
                        'metadata': json.loads(row[13] or '{}')
                    })
                
                return trades

    async def calculate_daily_performance(self, date: datetime = None) -> Dict:
        """Calculate performance metrics for a specific date"""
        if not date:
            date = datetime.now().date()
        
        async with aiosqlite.connect(self.db_file) as db:
            # Get trades for the date
            async with db.execute("""
                SELECT action, profit, profit_pct FROM trades 
                WHERE DATE(timestamp) = ? AND paper_mode = 1
            """, (date,)) as cursor:
                trades = await cursor.fetchall()
            
            if not trades:
                return {
                    'date': date.isoformat(),
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': 0,
                    'win_rate': 0,
                    'avg_profit_per_trade': 0
                }
            
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade[1] and trade[1] > 0)
            total_profit = sum(trade[1] or 0 for trade in trades)
            win_rate = winning_trades / total_trades * 100
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            return {
                'date': date.isoformat(),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'total_profit': total_profit,
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit
            }

    async def save_daily_performance(self, performance_data: Dict):
        """Save daily performance metrics"""
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute("""
                INSERT OR REPLACE INTO performance (
                    date, total_trades, winning_trades, total_profit,
                    paper_capital, portfolio_value, win_rate, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance_data['date'],
                performance_data['total_trades'],
                performance_data['winning_trades'],
                performance_data['total_profit'],
                performance_data.get('paper_capital'),
                performance_data.get('portfolio_value'),
                performance_data['win_rate'],
                json.dumps(performance_data.get('metadata', {}))
            ))
            await db.commit()

    async def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get performance history for specified days"""
        start_date = datetime.now().date() - timedelta(days=days)
        
        async with aiosqlite.connect(self.db_file) as db:
            async with db.execute("""
                SELECT * FROM performance 
                WHERE date >= ? 
                ORDER BY date DESC
            """, (start_date,)) as cursor:
                rows = await cursor.fetchall()
                
                performance = []
                for row in rows:
                    performance.append({
                        'date': row[1],
                        'total_trades': row[2],
                        'winning_trades': row[3],
                        'total_profit': row[4],
                        'paper_capital': row[5],
                        'portfolio_value': row[6],
                        'win_rate': row[7],
                        'sharpe_ratio': row[8],
                        'max_drawdown': row[9],
                        'metadata': json.loads(row[10] or '{}')
                    })
                
                return performance

    async def update_alpha_wallet(self, wallet_address: str, performance_data: Dict):
        """Update alpha wallet performance data"""
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute("""
                INSERT OR REPLACE INTO alpha_wallets (
                    wallet_address, win_rate, avg_hold_time, avg_profit_pct,
                    total_trades, last_updated, performance_score, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wallet_address,
                performance_data.get('win_rate'),
                performance_data.get('avg_hold_time'),
                performance_data.get('avg_profit_pct'),
                performance_data.get('total_trades', 0),
                datetime.now(),
                performance_data.get('performance_score'),
                json.dumps(performance_data.get('metadata', {}))
            ))
            await db.commit()

    async def get_top_alpha_wallets(self, limit: int = 50) -> List[Dict]:
        """Get top performing alpha wallets"""
        async with aiosqlite.connect(self.db_file) as db:
            async with db.execute("""
                SELECT * FROM alpha_wallets 
                ORDER BY performance_score DESC 
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
                
                wallets = []
                for row in rows:
                    wallets.append({
                        'wallet_address': row[1],
                        'win_rate': row[2],
                        'avg_hold_time': row[3],
                        'avg_profit_pct': row[4],
                        'total_trades': row[5],
                        'last_updated': row[6],
                        'performance_score': row[7],
                        'metadata': json.loads(row[8] or '{}')
                    })
                
                return wallets

    async def record_token_analysis(self, token_data: Dict):
        """Record token analysis data"""
        async with aiosqlite.connect(self.db_file) as db:
            await db.execute("""
                INSERT OR REPLACE INTO token_analysis (
                    mint, name, symbol, deployer, launch_time,
                    initial_liquidity, max_liquidity, volume_24h,
                    alpha_buyers, traded, performance, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                token_data['mint'],
                token_data.get('name'),
                token_data.get('symbol'),
                token_data.get('deployer'),
                token_data.get('launch_time'),
                token_data.get('initial_liquidity'),
                token_data.get('max_liquidity'),
                token_data.get('volume_24h'),
                token_data.get('alpha_buyers', 0),
                token_data.get('traded', False),
                token_data.get('performance'),
                json.dumps(token_data.get('metadata', {}))
            ))
            await db.commit()


    async def close(self):
        """Close database connections"""
        pass  # aiosqlite handles connection cleanup automatically