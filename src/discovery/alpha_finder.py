"""
Alpha Wallet Discovery System
Automatically finds and scores potential alpha wallets
"""

import asyncio
import logging
import time
from typing import Dict, List, Set
from datetime import datetime, timedelta
from collections import defaultdict

from src.clients.bitquery_client import BitqueryClient
from src.clients.moralis_client import MoralisClient
from src.core.database import Database


class AlphaWalletFinder:
    def __init__(self, bitquery: BitqueryClient, moralis: MoralisClient, database: Database):
        self.bitquery = bitquery
        self.moralis = moralis
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Scoring criteria
        self.min_trades = 5  # Minimum trades to be considered
        self.success_threshold = 2.0  # 2x minimum for "successful" trade
        self.early_buy_window = 300  # 5 minutes after token creation
        
        # Wallet tracking
        self.wallet_stats = defaultdict(lambda: {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0,
            'early_buys': 0,
            'tokens_traded': set(),
            'first_seen': None,
            'last_seen': None
        })

    async def analyze_recent_success_stories(self, days: int = 7) -> List[str]:
        """Find wallets that bought successful tokens early"""
        self.logger.info(f"Analyzing success stories from last {days} days")
        
        # Step 1: Get recent pump.fun trades
        successful_tokens = await self._find_successful_tokens(days)
        
        # Step 2: For each successful token, find early buyers
        alpha_candidates = set()
        
        for token_data in successful_tokens:
            mint = token_data['mint']
            launch_time = token_data['launch_time']
            
            early_buyers = await self._get_early_buyers(mint, launch_time)
            
            for buyer in early_buyers:
                alpha_candidates.add(buyer)
                self._update_wallet_stats(buyer, token_data, True)
        
        # Step 3: Score and rank wallets
        scored_wallets = await self._score_wallets(alpha_candidates)
        
        return scored_wallets

    async def _find_successful_tokens(self, days: int) -> List[Dict]:
        """Find tokens with high trading volume and activity"""
        successful_tokens = []
        
        try:
            # Get recent trades from Bitquery
            recent_trades = await self.bitquery.get_recent_token_launches(limit=50)
            
            for trade in recent_trades:
                mint = trade.get('mint')
                if not mint:
                    continue
                
                try:
                    # Get token metadata and holder stats
                    metadata = await self.moralis.get_token_metadata(mint)
                    holder_stats = await self.moralis.get_token_holders(mint)
                    
                    # Use holder count as proxy for "success"
                    holders = holder_stats.get('holder_count', 0)
                    
                    # Consider tokens with any holder activity as "interesting"
                    if holders >= 5:  # 5+ holders indicates some activity
                        successful_tokens.append({
                            'mint': mint,
                            'launch_time': trade.get('timestamp'),
                            'holders': holders,
                            'name': metadata.get('name', ''),
                            'symbol': metadata.get('symbol', '')
                        })
                        
                except Exception as e:
                    # Skip tokens that error out (very new or delisted)
                    continue
                    
                # Add small delay to avoid rate limits
                await asyncio.sleep(0.2)
            
        except Exception as e:
            self.logger.error(f"Error finding successful tokens: {e}")
        
        # Sort by holders (highest first) and return top tokens
        successful_tokens.sort(key=lambda x: x.get('holders', 0), reverse=True)
        return successful_tokens[:20]  # Top 20 by holder count

    async def _get_early_buyers(self, mint: str, launch_time: str) -> List[str]:
        """Get wallets that bought token within early window"""
        early_buyers = []
        
        try:
            # Get token swaps from Moralis (max 100 per API docs)
            swaps = await self.moralis.get_token_swaps(mint, limit=100)
            
            launch_timestamp = self._parse_timestamp(launch_time)
            early_cutoff = launch_timestamp + self.early_buy_window
            
            for swap in swaps:
                swap_time = self._parse_timestamp(swap.get('timestamp'))
                
                # Only consider early buys
                if swap_time <= early_cutoff:
                    wallet = swap.get('wallet')
                    if wallet and len(wallet) > 40:  # Valid wallet address
                        early_buyers.append(wallet)
            
        except Exception as e:
            self.logger.error(f"Error getting early buyers for {mint}: {e}")
        
        return early_buyers

    def _parse_timestamp(self, timestamp) -> float:
        """Parse timestamp to unix time"""
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        elif isinstance(timestamp, str):
            try:
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return dt.timestamp()
            except:
                pass
        return time.time()

    def _update_wallet_stats(self, wallet: str, token_data: Dict, is_early: bool):
        """Update statistics for a wallet"""
        stats = self.wallet_stats[wallet]
        
        stats['total_trades'] += 1
        stats['tokens_traded'].add(token_data['mint'])
        
        if is_early:
            stats['early_buys'] += 1
        
        # Update timestamps
        current_time = datetime.now()
        if not stats['first_seen']:
            stats['first_seen'] = current_time
        stats['last_seen'] = current_time

    async def _score_wallets(self, candidate_wallets: Set[str]) -> List[str]:
        """Score and rank wallet candidates"""
        scored_wallets = []
        
        for wallet in candidate_wallets:
            stats = self.wallet_stats[wallet]
            
            # Skip wallets with insufficient activity
            if stats['total_trades'] < self.min_trades:
                continue
            
            # Calculate score based on various factors
            score = self._calculate_wallet_score(stats)
            
            scored_wallets.append({
                'wallet': wallet,
                'score': score,
                'stats': dict(stats)
            })
        
        # Sort by score (highest first)
        scored_wallets.sort(key=lambda x: x['score'], reverse=True)
        
        # Log top performers
        self.logger.info(f"Found {len(scored_wallets)} potential alpha wallets")
        for i, wallet_data in enumerate(scored_wallets[:10]):
            wallet = wallet_data['wallet']
            score = wallet_data['score']
            stats = wallet_data['stats']
            self.logger.info(f"#{i+1}: {wallet[:8]}... Score: {score:.2f} "
                           f"(Trades: {stats['total_trades']}, Early: {stats['early_buys']})")
        
        return [w['wallet'] for w in scored_wallets[:20]]  # Return top 20

    def _calculate_wallet_score(self, stats: Dict) -> float:
        """Calculate a score for wallet performance"""
        score = 0
        
        # Base points for activity
        score += stats['total_trades'] * 1
        
        # Heavy bonus for early buying
        score += stats['early_buys'] * 5
        
        # Bonus for trading multiple tokens (diversification)
        score += len(stats['tokens_traded']) * 2
        
        # Recent activity bonus
        if stats['last_seen']:
            days_since_last = (datetime.now() - stats['last_seen']).days
            if days_since_last < 7:
                score += 10
        
        # Consistency bonus (early_buys / total_trades)
        if stats['total_trades'] > 0:
            consistency = stats['early_buys'] / stats['total_trades']
            score += consistency * 20
        
        return score

    async def save_alpha_wallets(self, wallets: List[str]):
        """Save discovered alpha wallets to database"""
        for wallet in wallets:
            stats = self.wallet_stats[wallet]
            
            await self.database.update_alpha_wallet(wallet, {
                'win_rate': 0,  # Would need price tracking to calculate
                'avg_hold_time': 0,  # Would need exit tracking
                'avg_profit_pct': 0,  # Would need PnL tracking
                'total_trades': stats['total_trades'],
                'performance_score': self._calculate_wallet_score(stats),
                'metadata': {
                    'early_buys': stats['early_buys'],
                    'tokens_traded': len(stats['tokens_traded']),
                    'discovery_method': 'retroactive_analysis',
                    'discovery_date': datetime.now().isoformat()
                }
            })
        
        self.logger.info(f"Saved {len(wallets)} alpha wallets to database")

    async def run_discovery(self) -> List[str]:
        """Run the complete alpha wallet discovery process"""
        self.logger.info("Starting alpha wallet discovery...")
        
        try:
            # Analyze recent successful trades
            alpha_wallets = await self.analyze_recent_success_stories(7)
            
            # Save to database
            await self.save_alpha_wallets(alpha_wallets)
            
            return alpha_wallets
            
        except Exception as e:
            self.logger.error(f"Alpha wallet discovery failed: {e}")
            return []


async def main():
    """Test the alpha wallet finder"""
    import yaml
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize clients
    bitquery = BitqueryClient(config['bitquery_token'])
    moralis = MoralisClient(config['moralis_key'])
    database = Database()
    
    await database.initialize()
    
    try:
        # Run alpha finder
        finder = AlphaWalletFinder(bitquery, moralis, database)
        alpha_wallets = await finder.run_discovery()
        
        print(f"\nDiscovered {len(alpha_wallets)} alpha wallets:")
        for wallet in alpha_wallets[:10]:
            print(f"  {wallet}")
            
    finally:
        # Clean up HTTP sessions
        await moralis.close()
        if hasattr(bitquery, 'client') and bitquery.client:
            await bitquery.client.transport.close()
        if hasattr(bitquery, 'ws_client') and bitquery.ws_client:
            await bitquery.ws_client.transport.close()


if __name__ == "__main__":
    asyncio.run(main())