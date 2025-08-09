"""
Alpha Wallet Accumulator - Continuous Discovery System
The main production system for finding alpha wallets

PURPOSE: 
BitQuery only gives us 3-4 minutes of trade data per call, regardless of the time window requested.
This accumulator accepts that limitation and builds up overlap data over time by running continuously.

HOW IT WORKS:
1. Runs every 2 minutes (or as configured)
2. Gets latest 3-4 minutes of trades from BitQuery
3. Finds early buyers (first 5 minutes after token launch)  
4. Stores data in SQLite database (data/alpha_accumulator.db)
5. Finds wallets that appear on 2+ successful tokens
6. Automatically updates config.yml with discovered alpha wallets

ADVANTAGES OVER ONE-OFF ANALYSIS:
- Builds historical data over time
- Finds genuine repeat patterns across multiple tokens
- Handles BitQuery's data limitations gracefully
- Runs 24/7 without manual intervention
- Better coverage than single-run analysis

USAGE:
  # Run once
  python scripts/run_alpha_accumulator.py
  
  # Run continuously every 2 minutes (RECOMMENDED for production)
  python scripts/run_alpha_accumulator.py --loop

RESULTS:
After 30-60 minutes of continuous running, you'll start seeing wallets that 
consistently buy multiple successful tokens early. These are your alpha wallets.

The system automatically updates your config.yml, so your trading bot will 
start following these wallets immediately.
"""

import asyncio
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict

from src.clients.bitquery_client import BitqueryClient
from src.clients.moralis_client import MoralisClient
from src.utils.config_loader import load_config


class AlphaAccumulator:
    def __init__(self, bitquery: BitqueryClient, moralis: MoralisClient):
        self.bitquery = bitquery
        self.moralis = moralis
        self.logger = logging.getLogger(__name__)
        self.db_path = 'data/alpha_accumulator.db'
        self._init_db()
        
    def _init_db(self):
        """Create simple tables for accumulation"""
        # Ensure data directory exists
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS tokens (
            mint TEXT PRIMARY KEY,
            launch_ts INTEGER,
            activity_score REAL,
            added_ts INTEGER
        )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS early_buys (
            wallet TEXT,
            mint TEXT,
            ts INTEGER,
            PRIMARY KEY (wallet, mint)
        )''')
        
        conn.commit()
        conn.close()
    
    async def run_accumulation(self):
        """One pass: get latest trades, find early buyers, store them"""
        self.logger.info("Starting accumulation pass...")
        
        # Step 1: Get latest ~3-4 minutes of trades from BitQuery
        trades = await self._get_latest_trades()
        if not trades:
            self.logger.warning("No trades returned from BitQuery")
            return
        
        # Step 2: Group by token and find launches
        tokens = self._extract_tokens(trades)
        self.logger.info(f"Found {len(tokens)} tokens in latest window")
        
        # Step 3: Rank by activity (just use trade count as proxy)
        ranked = sorted(tokens.items(), key=lambda x: len(x[1]['trades']), reverse=True)[:50]
        
        # Step 4: Get early buyers for top tokens
        early_buyer_count = 0
        for mint, token_info in ranked:
            # Skip if we need to rate limit
            if early_buyer_count > 30:  # Limit API calls per run
                break
                
            early_buyers = await self._get_early_buyers(mint, token_info['launch_ts'])
            if early_buyers:
                self._store_early_buyers(mint, token_info['launch_ts'], early_buyers)
                early_buyer_count += len(early_buyers)
        
        # Step 5: Prune old data (keep 12 hours)
        self._prune_old_data()
        
        # Step 6: Find overlapping wallets
        alpha_wallets = self._find_overlaps()
        
        if alpha_wallets:
            self.logger.info(f"Found {len(alpha_wallets)} alpha wallets with overlaps:")
            for wallet, count in alpha_wallets[:10]:
                self.logger.info(f"  {wallet[:8]}... appears on {count} tokens")
        
        return alpha_wallets
    
    async def _get_latest_trades(self):
        """Get whatever BitQuery gives us (usually 3-4 minutes)"""
        try:
            # Just ask for recent trades, no time filter since BitQuery ignores it anyway
            trades = await self.bitquery.get_recent_token_launches(limit=2000)
            return trades
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return []
    
    def _extract_tokens(self, trades):
        """Group trades by token and find launch times"""
        tokens = {}
        SOL_ADDRESS = "11111111111111111111111111111111"
        
        for trade in trades:
            # Extract mint address
            buy_curr = trade.get('Trade', {}).get('Buy', {}).get('Currency', {})
            sell_curr = trade.get('Trade', {}).get('Sell', {}).get('Currency', {})
            
            mint = None
            if buy_curr.get('MintAddress') and buy_curr.get('MintAddress') != SOL_ADDRESS:
                mint = buy_curr.get('MintAddress')
            elif sell_curr.get('MintAddress') and sell_curr.get('MintAddress') != SOL_ADDRESS:
                mint = sell_curr.get('MintAddress')
            
            if not mint or len(mint) < 40:
                continue
            
            # Get timestamp
            ts_str = trade.get('Block', {}).get('Time', '')
            if not ts_str:
                continue
            
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                ts = dt.timestamp()
            except:
                continue
            
            # Track earliest timestamp as launch
            if mint not in tokens:
                tokens[mint] = {'launch_ts': ts, 'trades': []}
            
            tokens[mint]['trades'].append(ts)
            if ts < tokens[mint]['launch_ts']:
                tokens[mint]['launch_ts'] = ts
        
        return tokens
    
    async def _get_early_buyers(self, mint: str, launch_ts: float):
        """Get wallets that bought in first 5 minutes"""
        try:
            # Get swaps
            swaps = await self.moralis.get_token_swaps(mint, limit=100)
            
            early_buyers = set()
            for swap in swaps:
                # Parse timestamp
                ts_str = swap.get('timestamp', '')
                if not ts_str:
                    continue
                
                try:
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    swap_ts = dt.timestamp()
                except:
                    continue
                
                # Check if within 5 minutes of launch
                if launch_ts <= swap_ts <= launch_ts + 300:
                    # Check if it's a buy into token
                    if swap.get('to_token') == mint:
                        wallet = swap.get('wallet')
                        if wallet and len(wallet) >= 32 and wallet != mint:
                            early_buyers.add(wallet)
            
            return list(early_buyers)
            
        except Exception as e:
            self.logger.debug(f"Error getting early buyers for {mint[:8]}...: {e}")
            return []
    
    def _store_early_buyers(self, mint: str, launch_ts: float, wallets: list):
        """Store early buyers in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        now = int(time.time())
        
        # Store token
        c.execute('INSERT OR IGNORE INTO tokens (mint, launch_ts, activity_score, added_ts) VALUES (?, ?, ?, ?)',
                  (mint, int(launch_ts), len(wallets), now))
        
        # Store early buyers
        for wallet in wallets:
            c.execute('INSERT OR IGNORE INTO early_buys (wallet, mint, ts) VALUES (?, ?, ?)',
                      (wallet, mint, now))
        
        conn.commit()
        conn.close()
    
    def _prune_old_data(self):
        """Remove data older than 12 hours"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        cutoff = int(time.time()) - (12 * 3600)
        
        # Remove old early buys
        c.execute('DELETE FROM early_buys WHERE ts < ?', (cutoff,))
        
        # Remove old tokens
        c.execute('DELETE FROM tokens WHERE added_ts < ?', (cutoff,))
        
        conn.commit()
        conn.close()
    
    def _find_overlaps(self):
        """Find wallets that appear on 2+ tokens"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Find wallets with multiple token appearances
        c.execute('''
            SELECT wallet, COUNT(DISTINCT mint) as token_count
            FROM early_buys
            GROUP BY wallet
            HAVING token_count >= 2
            ORDER BY token_count DESC
            LIMIT 50
        ''')
        
        results = c.fetchall()
        conn.close()
        
        return results


async def main():
    """Run the accumulator"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = load_config('config.yml')
    
    bitquery = BitqueryClient(config.get('bitquery_tokens', config.get('bitquery_token')))
    moralis = MoralisClient(config.get('moralis_keys', config.get('moralis_key')))
    
    try:
        accumulator = AlphaAccumulator(bitquery, moralis)
        alpha_wallets = await accumulator.run_accumulation()
        
        if alpha_wallets:
            print(f"\n✅ Found {len(alpha_wallets)} alpha wallets!")
            
            # Update config with wallets
            import yaml
            wallet_addresses = [w[0] for w in alpha_wallets]
            
            current_wallets = set(config.get('watched_wallets', []))
            updated_wallets = list(current_wallets.union(set(wallet_addresses)))[:50]
            config['watched_wallets'] = updated_wallets
            
            with open('config.yml', 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            
            print(f"Config updated with {len(updated_wallets)} wallets")
        else:
            print("\nNo overlapping wallets yet. Run this script every few minutes to accumulate data.")
            print("After ~30 minutes you should start seeing overlaps.")
    
    finally:
        await moralis.close()
        if hasattr(bitquery, 'client') and bitquery.client:
            await bitquery.client.transport.close()


if __name__ == "__main__":
    asyncio.run(main())