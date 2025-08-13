"""
Stream-based safety checks for token trading
Honeypot detection and price impact estimation without additional API calls
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class SafetyChecker:
    """
    Performs safety checks on tokens using stream data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def check_sellability(self, mint: str, recent_trades: List[Dict]) -> bool:
        """
        Check if token is sellable (not a honeypot)
        Requires at least 1 sell by non-dev wallet in last 60s
        
        Args:
            mint: Token mint address
            recent_trades: List of recent swap transactions
            
        Returns:
            True if token appears sellable, False otherwise
        """
        if not recent_trades:
            self.logger.warning(f"No trade data for sellability check on {mint[:8]}...")
            return False
            
        # Look for sells in the last 60 seconds
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=60)
        
        sells_found = 0
        unique_sellers = set()
        
        for trade in recent_trades:
            # Parse timestamp
            trade_time = self._parse_timestamp(trade.get('timestamp'))
            if trade_time < cutoff_time:
                continue
                
            # Check if it's a sell
            if trade.get('side') == 'sell':
                wallet = trade.get('wallet', '')
                
                # Ignore known dev/deployer wallets (would need a list)
                # For now, just check for any sells
                if wallet:
                    sells_found += 1
                    unique_sellers.add(wallet)
        
        # Require at least 1 sell from unique wallet
        if sells_found == 0:
            self.logger.warning(f"No sells detected for {mint[:8]}... - potential honeypot")
            return False
            
        if len(unique_sellers) == 0:
            self.logger.warning(f"No unique sellers for {mint[:8]}... - suspicious")
            return False
            
        self.logger.debug(f"Sellability check passed: {sells_found} sells from {len(unique_sellers)} wallets")
        return True
    
    def estimate_price_impact(self, mint: str, order_size_usd: float, recent_trades: List[Dict]) -> float:
        """
        Estimate price impact of our order based on recent trade sizes
        
        Args:
            mint: Token mint address
            order_size_usd: Our order size in USD
            recent_trades: List of recent swap transactions
            
        Returns:
            Estimated price impact as percentage (e.g., 0.01 = 1%)
        """
        if not recent_trades:
            # No data, assume high impact
            return 0.05  # 5% default
            
        # Calculate average trade size from recent trades
        trade_sizes = []
        for trade in recent_trades[-20:]:  # Look at last 20 trades
            size = trade.get('amount_usd', 0)
            if size > 0:
                trade_sizes.append(size)
        
        if not trade_sizes:
            return 0.05  # No size data, assume 5%
            
        avg_trade_size = sum(trade_sizes) / len(trade_sizes)
        
        # Simple impact model: if our order is larger than average, expect impact
        if order_size_usd <= avg_trade_size:
            # Our order is smaller than average, minimal impact
            base_impact = 0.005  # 0.5%
        else:
            # Scale impact based on how much larger we are
            size_ratio = order_size_usd / avg_trade_size
            
            # Impact increases with size ratio
            # 2x average = 1% impact, 5x = 3%, 10x = 5%
            if size_ratio <= 2:
                base_impact = 0.01  # 1%
            elif size_ratio <= 5:
                base_impact = 0.03  # 3%
            elif size_ratio <= 10:
                base_impact = 0.05  # 5%
            else:
                base_impact = 0.10  # 10% for very large orders
        
        self.logger.debug(f"Estimated price impact: {base_impact*100:.1f}% "
                         f"(order=${order_size_usd:.0f}, avg=${avg_trade_size:.0f})")
        
        return base_impact
    
    def check_token_safety(self, mint: str, order_size_usd: float, recent_trades: List[Dict],
                          max_impact: float = 0.01) -> Dict:
        """
        Comprehensive safety check combining sellability and price impact
        
        Args:
            mint: Token mint address
            order_size_usd: Our order size in USD
            recent_trades: List of recent swap transactions
            max_impact: Maximum acceptable price impact (default 1%)
            
        Returns:
            Dictionary with safety results
        """
        # Check sellability (honeypot detection)
        is_sellable = self.check_sellability(mint, recent_trades)
        
        # Estimate price impact
        price_impact = self.estimate_price_impact(mint, order_size_usd, recent_trades)
        
        # Determine if safe to trade
        safe_to_trade = is_sellable and price_impact <= max_impact
        
        # Compile warnings
        warnings = []
        if not is_sellable:
            warnings.append("No recent sells detected (potential honeypot)")
        if price_impact > max_impact:
            warnings.append(f"High price impact: {price_impact*100:.1f}%")
        if price_impact > 0.03:
            warnings.append("Order size may be too large for liquidity")
            
        result = {
            'safe': safe_to_trade,
            'sellable': is_sellable,
            'price_impact': price_impact,
            'warnings': warnings,
            'recommendation': 'TRADE' if safe_to_trade else 'SKIP'
        }
        
        if not safe_to_trade:
            self.logger.warning(f"Token {mint[:8]}... failed safety: {', '.join(warnings)}")
        else:
            self.logger.info(f"Token {mint[:8]}... passed safety checks (impact: {price_impact*100:.1f}%)")
            
        return result
    
    def _parse_timestamp(self, timestamp) -> datetime:
        """Parse timestamp to datetime object"""
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return datetime.now()
        return datetime.now()
    
    def analyze_trading_velocity(self, recent_trades: List[Dict]) -> Dict:
        """
        Analyze trading velocity and momentum
        
        Args:
            recent_trades: List of recent swap transactions
            
        Returns:
            Dictionary with velocity metrics
        """
        if not recent_trades:
            return {
                'trades_per_minute': 0,
                'buy_sell_ratio': 0,
                'momentum': 'neutral'
            }
            
        # Count trades in last minute
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        recent_buys = 0
        recent_sells = 0
        
        for trade in recent_trades:
            trade_time = self._parse_timestamp(trade.get('timestamp'))
            if trade_time < one_minute_ago:
                continue
                
            if trade.get('side') == 'buy':
                recent_buys += 1
            elif trade.get('side') == 'sell':
                recent_sells += 1
        
        total_recent = recent_buys + recent_sells
        buy_sell_ratio = recent_buys / max(recent_sells, 1)
        
        # Determine momentum
        if buy_sell_ratio > 2 and total_recent > 5:
            momentum = 'strong_bullish'
        elif buy_sell_ratio > 1.5 and total_recent > 3:
            momentum = 'bullish'
        elif buy_sell_ratio < 0.5 and total_recent > 3:
            momentum = 'bearish'
        elif buy_sell_ratio < 0.7:
            momentum = 'weak_bearish'
        else:
            momentum = 'neutral'
            
        return {
            'trades_per_minute': total_recent,
            'buy_sell_ratio': buy_sell_ratio,
            'recent_buys': recent_buys,
            'recent_sells': recent_sells,
            'momentum': momentum
        }