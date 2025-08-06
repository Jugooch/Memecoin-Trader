"""
Alpha wallet tracking system for detecting smart money activity
"""

import asyncio
import logging
import time
from typing import List, Set, Dict
from src.clients.moralis_client import MoralisClient


class WalletTracker:
    def __init__(self, watched_wallets: List[str]):
        self.watched_wallets = set(watched_wallets)
        self.logger = logging.getLogger(__name__)
        
        # Cache for recent activity
        self.recent_activity = {}
        self.cache_duration = 300  # 5 minutes

    async def check_alpha_activity(self, mint_address: str, time_window_sec: int, threshold_buys: int, moralis_client=None) -> bool:
        """
        Check if enough alpha wallets have bought the token within time window
        """
        self.logger.debug(f"Checking alpha activity for {mint_address}")
        
        start_time = time.time()
        alpha_buyers = set()
        
        # Use injected moralis client
        if not moralis_client:
            self.logger.error("Moralis client not provided")
            return False
            
        moralis = moralis_client
        
        while time.time() - start_time < time_window_sec:
            try:
                # Get recent swaps for the token (max 100 per Moralis API limit)
                swaps = await moralis.get_token_swaps(mint_address, limit=100)
                
                for swap in swaps:
                    swap_time = self._parse_timestamp(swap.get('timestamp'))
                    
                    # Only consider swaps within our time window
                    if time.time() - swap_time > time_window_sec:
                        continue
                    
                    wallet = swap.get('wallet', '')
                    
                    # Check if this is a buy (to_token is our mint)
                    if swap.get('to_token') == mint_address and wallet in self.watched_wallets:
                        alpha_buyers.add(wallet)
                        self.logger.info(f"ALPHA WALLET DETECTED: {wallet[:8]}... bought {mint_address[:8]}...")
                
                # Check if we've hit the threshold
                if len(alpha_buyers) >= threshold_buys:
                    self.logger.info(f"ALPHA SIGNAL TRIGGERED! {len(alpha_buyers)} alpha wallets bought {mint_address[:8]}...")
                    return True
                
                # Wait before checking again
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error checking alpha activity: {e}")
                await asyncio.sleep(2)
        
        # Only log if we found some activity, otherwise it's just noise
        if len(alpha_buyers) > 0:
            self.logger.info(f"Alpha check complete: {len(alpha_buyers)}/{threshold_buys} alpha wallets bought {mint_address[:8]}...")
        else:
            self.logger.debug(f"No alpha wallet activity found for {mint_address[:8]}...")
        return False

    def _parse_timestamp(self, timestamp) -> float:
        """Parse timestamp to unix time"""
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        elif isinstance(timestamp, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.timestamp()
            except:
                return time.time()
        return time.time()

    async def analyze_wallet_patterns(self, wallet_address: str, moralis_client=None) -> Dict:
        """Analyze trading patterns of a specific wallet"""
        try:
            if not moralis_client:
                self.logger.error("Moralis client not provided")
                return {}
            moralis = moralis_client
            
            # Get wallet portfolio
            portfolio = await moralis.get_wallet_portfolio(wallet_address)
            
            # Analyze recent performance
            recent_trades = await self._get_recent_wallet_trades(wallet_address)
            
            win_rate = self._calculate_win_rate(recent_trades)
            avg_hold_time = self._calculate_avg_hold_time(recent_trades)
            avg_profit = self._calculate_avg_profit(recent_trades)
            
            return {
                'wallet': wallet_address,
                'portfolio_value': portfolio.get('total_value_usd', 0),
                'token_count': portfolio.get('token_count', 0),
                'win_rate': win_rate,
                'avg_hold_time_hours': avg_hold_time,
                'avg_profit_pct': avg_profit,
                'recent_trades': len(recent_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing wallet {wallet_address}: {e}")
            return {}

    async def _get_recent_wallet_trades(self, wallet_address: str, days: int = 7) -> List[Dict]:
        """Get recent trades for a wallet"""
        # This would implement logic to fetch and analyze recent trades
        # For now, return empty list
        return []

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        return winning_trades / len(trades)

    def _calculate_avg_hold_time(self, trades: List[Dict]) -> float:
        """Calculate average holding time in hours"""
        if not trades:
            return 0.0
        
        total_time = sum(trade.get('hold_time_hours', 0) for trade in trades)
        return total_time / len(trades)

    def _calculate_avg_profit(self, trades: List[Dict]) -> float:
        """Calculate average profit percentage"""
        if not trades:
            return 0.0
        
        total_profit = sum(trade.get('profit_pct', 0) for trade in trades)
        return total_profit / len(trades)

    def add_watched_wallet(self, wallet_address: str):
        """Add a wallet to the watch list"""
        self.watched_wallets.add(wallet_address)
        self.logger.info(f"Added wallet {wallet_address} to watch list")

    def remove_watched_wallet(self, wallet_address: str):
        """Remove a wallet from the watch list"""
        self.watched_wallets.discard(wallet_address)
        self.logger.info(f"Removed wallet {wallet_address} from watch list")

    def get_watched_wallets(self) -> List[str]:
        """Get list of watched wallets"""
        return list(self.watched_wallets)

    async def update_wallet_performance(self):
        """Update performance metrics for all watched wallets"""
        self.logger.info("Updating wallet performance metrics")
        
        for wallet in self.watched_wallets:
            try:
                analysis = await self.analyze_wallet_patterns(wallet)
                # Store or cache the analysis results
                self.recent_activity[wallet] = {
                    'last_updated': time.time(),
                    'analysis': analysis
                }
            except Exception as e:
                self.logger.error(f"Error updating wallet {wallet}: {e}")

    def get_wallet_performance(self, wallet_address: str) -> Dict:
        """Get cached performance data for a wallet"""
        wallet_data = self.recent_activity.get(wallet_address, {})
        
        if not wallet_data or time.time() - wallet_data.get('last_updated', 0) > self.cache_duration:
            return {}
        
        return wallet_data.get('analysis', {})