"""
Follower Harm Analyzer - Detects wallets that harm their followers
This is the most important metric for detecting farming wallets
"""

import logging
import statistics
from typing import Dict, List, Set
import numpy as np


class FollowerHarmAnalyzer:
    """
    Analyzes whether wallets harm their followers
    This is the most important metric for detecting farmers
    """
    
    def __init__(self, bitquery_client=None):
        self.bitquery = bitquery_client
        self.logger = logging.getLogger(__name__)
        self.follower_cache = {}  # Cache follower relationships
    
    async def analyze_follower_outcomes(
        self,
        wallet: str,
        trades_data: List[Dict],
        time_window: int = 60
    ) -> Dict:
        """
        Measure outcomes for wallets that follow this wallet's buys
        
        Args:
            wallet: Wallet address to analyze
            trades_data: Historical trades by this wallet
            time_window: Window to consider as "following" (seconds)
        
        Returns:
            Dict containing:
            - trap_rate: % of times followers lose >10%
            - median_follower_return: Typical follower outcome
            - follower_count: Number of unique followers
            - is_trap_wallet: Boolean determination
        """
        if not self.bitquery:
            return {
                'trap_rate': 0,
                'median_follower_return': 0,
                'total_events': 0,
                'unique_followers': 0,
                'is_trap_wallet': False,
                'error': 'No BitQuery client available'
            }
        
        harm_events = []
        total_events = 0
        unique_followers = set()
        
        for trade in trades_data:
            if trade.get('side') != 'buy':
                continue
            
            token = trade.get('token_address', trade.get('mint', ''))
            timestamp = trade.get('timestamp', 0)
            wallet_price = trade.get('price', 0)
            
            if not token or not timestamp or not wallet_price:
                continue
            
            try:
                # Get buyers who entered after this wallet
                followers = await self.bitquery.get_token_trades_in_window(
                    token=token,
                    start_time=timestamp + 5,  # 5s buffer
                    end_time=timestamp + time_window
                )
                
                # Filter to only buy trades
                follower_buys = [f for f in followers if f.get('side') == 'buy']
                
                if len(follower_buys) < 2:  # Need at least 2 followers
                    continue
                
                total_events += 1
                unique_followers.update([f.get('wallet', '') for f in follower_buys])
                
                # Calculate average follower entry price (includes slippage)
                follower_prices = [f.get('price', 0) for f in follower_buys if f.get('price', 0) > 0]
                
                if not follower_prices:
                    continue
                    
                avg_follower_price = np.mean(follower_prices)
                
                # Get price 60 seconds after wallet's buy
                price_60s_trades = await self.bitquery.get_token_trades_in_window(
                    token=token,
                    start_time=timestamp + 60,
                    end_time=timestamp + 65
                )
                
                if not price_60s_trades:
                    continue
                
                # Calculate average price at 60s
                prices_60s = [t.get('price', 0) for t in price_60s_trades if t.get('price', 0) > 0]
                if not prices_60s:
                    continue
                    
                price_60s = np.mean(prices_60s)
                
                # Calculate follower return
                follower_return = (price_60s - avg_follower_price) / avg_follower_price
                
                harm_events.append({
                    'token': token,
                    'timestamp': timestamp,
                    'wallet_entry': wallet_price,
                    'avg_follower_entry': avg_follower_price,
                    'price_60s': price_60s,
                    'follower_return': follower_return,
                    'follower_count': len(follower_buys),
                    'is_trap': follower_return < -0.10  # Lost >10%
                })
                
            except Exception as e:
                self.logger.debug(f"Error analyzing follower outcomes for {token[:8]}...: {e}")
                continue
        
        # Calculate statistics
        if not harm_events:
            return {
                'trap_rate': 0,
                'median_follower_return': 0,
                'total_events': 0,
                'unique_followers': 0,
                'is_trap_wallet': False
            }
        
        trap_count = sum(1 for e in harm_events if e['is_trap'])
        returns = [e['follower_return'] for e in harm_events]
        
        result = {
            'trap_rate': trap_count / len(harm_events),
            'median_follower_return': np.median(returns),
            'avg_follower_return': np.mean(returns),
            'total_events': total_events,
            'unique_followers': len(unique_followers),
            'harm_events': harm_events,
            'is_trap_wallet': False
        }
        
        # Determine if wallet is a trap
        # Criteria: >35% trap rate AND at least 3 events
        if result['trap_rate'] > 0.35 and len(harm_events) >= 3:
            result['is_trap_wallet'] = True
            self.logger.warning(f"TRAP WALLET DETECTED: {wallet[:8]}... "
                              f"Trap rate: {result['trap_rate']:.1%}, "
                              f"Median follower return: {result['median_follower_return']:.1%}")
        
        return result
    
    async def detect_coordination_patterns(
        self,
        wallet: str,
        related_wallets: List[str],
        time_window: int = 300
    ) -> Dict:
        """
        Detect if wallet coordinates with others to farm
        
        Uses Lead-Lag Asymmetry Score (LLAS) to find coordination
        """
        if not self.bitquery:
            return {
                'coordinated_wallets': {},
                'coordination_count': 0,
                'is_coordinated': False,
                'error': 'No BitQuery client available'
            }
        
        coordination_scores = {}
        
        for other_wallet in related_wallets:
            if other_wallet == wallet:
                continue
            
            try:
                # Get trades for both wallets
                wallet_trades = await self.bitquery.get_wallet_trades(
                    wallet, limit=50
                )
                other_trades = await self.bitquery.get_wallet_trades(
                    other_wallet, limit=50
                )
                
                # Calculate LLAS
                llas = self._calculate_llas(wallet_trades, other_trades, time_window)
                
                if abs(llas) > 0.3:  # Significant asymmetry
                    coordination_scores[other_wallet] = {
                        'llas': llas,
                        'is_leader': llas > 0,
                        'is_follower': llas < 0
                    }
                    
            except Exception as e:
                self.logger.debug(f"Error analyzing coordination between {wallet[:8]}... "
                                f"and {other_wallet[:8]}...: {e}")
                continue
        
        return {
            'coordinated_wallets': coordination_scores,
            'coordination_count': len(coordination_scores),
            'is_coordinated': len(coordination_scores) >= 2
        }
    
    def _calculate_llas(
        self,
        wallet_trades: List[Dict],
        other_trades: List[Dict],
        window: int = 60
    ) -> float:
        """
        Calculate Lead-Lag Asymmetry Score
        Positive = wallet leads, Negative = wallet follows
        """
        wallet_leads = 0
        wallet_follows = 0
        
        # Create lookup for faster matching
        other_trades_by_token = {}
        for trade in other_trades:
            token = trade.get('token_address', trade.get('mint', ''))
            if token:
                if token not in other_trades_by_token:
                    other_trades_by_token[token] = []
                other_trades_by_token[token].append(trade)
        
        for w_trade in wallet_trades:
            if w_trade.get('side') != 'buy':
                continue
            
            w_token = w_trade.get('token_address', w_trade.get('mint', ''))
            w_time = w_trade.get('timestamp', 0)
            
            if not w_token or not w_time:
                continue
            
            # Check if other wallet bought same token soon after
            if w_token in other_trades_by_token:
                for o_trade in other_trades_by_token[w_token]:
                    if o_trade.get('side') != 'buy':
                        continue
                    
                    o_time = o_trade.get('timestamp', 0)
                    time_diff = o_time - w_time
                    
                    if 0 < time_diff <= window:
                        wallet_leads += 1
                        break
                    elif -window <= time_diff < 0:
                        wallet_follows += 1
                        break
        
        total_interactions = wallet_leads + wallet_follows
        if total_interactions == 0:
            return 0.0
        
        return (wallet_leads - wallet_follows) / total_interactions
    
    async def analyze_wash_trading_patterns(
        self,
        wallet: str,
        trades_data: List[Dict]
    ) -> Dict:
        """
        Detect wash trading patterns (buying and selling to self)
        """
        wash_indicators = {
            'rapid_roundtrips': 0,
            'symmetric_trades': 0,
            'self_trading_suspected': False,
            'wash_score': 0
        }
        
        # Group trades by token
        token_trades = {}
        for trade in trades_data:
            token = trade.get('token_address', trade.get('mint', ''))
            if token:
                if token not in token_trades:
                    token_trades[token] = []
                token_trades[token].append(trade)
        
        for token, trades in token_trades.items():
            # Sort by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
            
            # Look for rapid buy-sell patterns
            for i in range(len(sorted_trades) - 1):
                curr_trade = sorted_trades[i]
                next_trade = sorted_trades[i + 1]
                
                if (curr_trade.get('side') == 'buy' and 
                    next_trade.get('side') == 'sell'):
                    
                    time_diff = next_trade.get('timestamp', 0) - curr_trade.get('timestamp', 0)
                    
                    if time_diff < 60:  # Less than 1 minute
                        wash_indicators['rapid_roundtrips'] += 1
                    
                    # Check for symmetric amounts (wash trading indicator)
                    buy_amount = curr_trade.get('amount', 0)
                    sell_amount = next_trade.get('amount', 0)
                    
                    if buy_amount > 0 and abs(buy_amount - sell_amount) / buy_amount < 0.05:
                        wash_indicators['symmetric_trades'] += 1
        
        # Calculate wash score
        total_trades = len(trades_data)
        if total_trades > 0:
            wash_indicators['wash_score'] = (
                (wash_indicators['rapid_roundtrips'] / total_trades) * 50 +
                (wash_indicators['symmetric_trades'] / total_trades) * 50
            )
            
            if wash_indicators['wash_score'] > 30:
                wash_indicators['self_trading_suspected'] = True
        
        return wash_indicators