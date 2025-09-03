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
    
    def __init__(self, safety_config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = safety_config or {}
        
    def check_sellability(self, mint: str, recent_trades: List[Dict]) -> Dict:
        """
        Enhanced sellability check requiring multiple distinct sellers
        
        Args:
            mint: Token mint address
            recent_trades: List of recent swap transactions
            
        Returns:
            Dictionary with sellability results
        """
        if not recent_trades:
            self.logger.warning(f"No trade data for sellability check on {mint[:8]}...")
            return {
                'is_sellable': False,
                'reason': 'No trade data available',
                'sellers_found': 0,
                'unique_sellers': 0,
                'sell_failure_rate': 0.0
            }
            
        # Look for sells in the last 90 seconds (increased from 60)
        import time
        current_timestamp = time.time()
        cutoff_timestamp = current_timestamp - 90
        
        sells_found = 0
        unique_sellers = set()
        failed_sells = 0
        total_sell_attempts = 0
        
        for trade in recent_trades:
            # Parse timestamp to unix timestamp for consistent comparison
            trade_timestamp = self._parse_timestamp_to_unix(trade.get('timestamp'))
            if trade_timestamp < cutoff_timestamp:
                continue
                
            # Check if it's a sell
            if trade.get('side') == 'sell':
                wallet = trade.get('wallet', '')
                total_sell_attempts += 1
                
                # Check if transaction failed (if this data is available)
                tx_success = trade.get('success', True)  # Default to True if not available
                if not tx_success:
                    failed_sells += 1
                    continue
                    
                if wallet:
                    sells_found += 1
                    unique_sellers.add(wallet)
        
        # Calculate failure rate
        sell_failure_rate = failed_sells / max(total_sell_attempts, 1)
        
        # Enhanced requirements - use config values
        min_sellers = self.config.get('min_distinct_sellers', 3)
        max_failure_rate = self.config.get('max_sell_failure_rate', 0.05)
        
        is_sellable = (
            len(unique_sellers) >= min_sellers and 
            sell_failure_rate <= max_failure_rate
        )
        
        reason_parts = []
        if not is_sellable:
            if len(unique_sellers) < min_sellers:
                reason_parts.append(f"Only {len(unique_sellers)} unique sellers (need {min_sellers})")
            if sell_failure_rate > max_failure_rate:
                reason_parts.append(f"High failure rate: {sell_failure_rate:.1%} (max {max_failure_rate:.1%})")
            
            reason = '; '.join(reason_parts)
            self.logger.warning(f"Enhanced sellability failed for {mint[:8]}...: {reason}")
        else:
            reason = 'Passed enhanced sellability'
            
        result = {
            'is_sellable': is_sellable,
            'reason': reason,
            'sellers_found': sells_found,
            'unique_sellers': len(unique_sellers),
            'sell_failure_rate': sell_failure_rate
        }
        
        self.logger.debug(f"Enhanced sellability check: {result}")
        return result
    
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
    
    def check_price_extension(self, mint: str, recent_trades: List[Dict], 
                             current_price: float = None) -> Dict:
        """
        Check if current price is at recent peak (extension guard)
        
        Args:
            mint: Token mint address
            recent_trades: List of recent trades
            current_price: Current token price (optional)
            
        Returns:
            Dictionary with extension check results
        """
        if not recent_trades:
            return {
                'is_extended': False,
                'reason': 'No trade data available',
                'percentile_rank': 0.0
            }
        
        # Get prices from last 60 seconds
        import time
        current_timestamp = time.time()
        cutoff_timestamp = current_timestamp - 60
        
        recent_prices = []
        for trade in recent_trades:
            trade_timestamp = self._parse_timestamp_to_unix(trade.get('timestamp'))
            if trade_timestamp >= cutoff_timestamp:
                price = trade.get('price', 0) or trade.get('price_usd', 0)
                if price > 0:
                    recent_prices.append(price)
        
        if len(recent_prices) < 5:  # Need minimum price points
            return {
                'is_extended': False,
                'reason': 'Insufficient price data',
                'percentile_rank': 0.0
            }
        
        # Use current price or latest trade price
        if current_price is None:
            current_price = recent_prices[-1] if recent_prices else 0
        
        if current_price <= 0:
            return {
                'is_extended': False,
                'reason': 'Invalid current price',
                'percentile_rank': 0.0
            }
        
        # Calculate percentile rank
        sorted_prices = sorted(recent_prices)
        p95_threshold = sorted_prices[int(len(sorted_prices) * 0.95)] if len(sorted_prices) > 1 else sorted_prices[0]
        is_extended = current_price > p95_threshold
        
        # Calculate where current price ranks
        rank = sum(1 for p in sorted_prices if p <= current_price) / len(sorted_prices)
        
        return {
            'is_extended': is_extended,
            'reason': f"Price at {rank:.1%} percentile" if is_extended else "Price not extended",
            'percentile_rank': rank,
            'p95_threshold': p95_threshold,
            'current_price': current_price,
            'recent_prices_count': len(recent_prices)
        }
    
    def check_token_safety(self, mint: str, order_size_usd: float, recent_trades: List[Dict],
                          max_impact: float = 0.01, current_price: float = None, bypass_sellability: bool = False) -> Dict:
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
        # Enhanced sellability check (skip if disabled or bypassed)
        if bypass_sellability:
            sellability_result = {'is_sellable': True, 'reason': 'Sellability check bypassed by safety hybrid'}
            is_sellable = True
        elif self.config.get('require_sellability', True):
            sellability_result = self.check_sellability(mint, recent_trades)
            is_sellable = sellability_result['is_sellable']
        else:
            sellability_result = {'is_sellable': True, 'reason': 'Sellability check disabled'}
            is_sellable = True
        
        # Estimate price impact
        price_impact = self.estimate_price_impact(mint, order_size_usd, recent_trades)
        
        # Price extension guard
        extension_result = self.check_price_extension(mint, recent_trades, current_price)
        is_not_extended = not extension_result['is_extended']
        
        # Determine if safe to trade
        safe_to_trade = is_sellable and price_impact <= max_impact and is_not_extended
        
        # Compile warnings
        warnings = []
        if not is_sellable:
            warnings.append(sellability_result['reason'])
        if price_impact > max_impact:
            warnings.append(f"High price impact: {price_impact*100:.1f}% > {max_impact*100:.1f}%")
        if price_impact > 0.03:
            warnings.append("Order size may be too large for liquidity")
        if extension_result['is_extended']:
            warnings.append(f"Price extended: {extension_result['reason']}")
        # Dump protection now runs in trading engine before safety checks
            
        result = {
            'safe': safe_to_trade,
            'safe_to_trade': safe_to_trade,  # Alias for backward compatibility
            'sellability': sellability_result,  # Enhanced sellability data
            'sellable': is_sellable,  # Backward compatibility
            'price_impact': price_impact,
            'extension_guard': extension_result,  # Price extension data
            # 'dump_protection': moved to trading engine
            'warnings': warnings,
            'recommendation': 'TRADE' if safe_to_trade else 'SKIP',
            'max_impact_threshold': max_impact
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
    
    def calculate_avwap(self, recent_trades: List[Dict], window_seconds: int = 60) -> float:
        """
        Calculate Anchored Volume Weighted Average Price from recent trades
        
        Args:
            recent_trades: List of recent swap transactions
            window_seconds: Time window for AVWAP calculation
            
        Returns:
            AVWAP price
        """
        if not recent_trades:
            return 0
            
        import time
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        total_volume_usd = 0
        total_weighted_price = 0
        
        for trade in recent_trades:
            trade_time = self._parse_timestamp_to_unix(trade.get('timestamp'))
            if trade_time < cutoff_time:
                continue
                
            price = trade.get('price', 0)
            volume_usd = trade.get('amount_usd', 0)
            
            if price > 0 and volume_usd > 0:
                total_weighted_price += price * volume_usd
                total_volume_usd += volume_usd
        
        if total_volume_usd > 0:
            return total_weighted_price / total_volume_usd
        return 0
    
    def check_dump_protection(self, mint: str, recent_trades: List[Dict], current_price: float) -> Dict:
        """
        Check if token is dumping and should be avoided
        
        Args:
            mint: Token mint address
            recent_trades: List of recent swap transactions
            current_price: Current token price
            
        Returns:
            Dictionary with dump protection results
        """
        import time
        current_time = time.time()
        
        # Check price momentum over last 30 seconds
        momentum_window = 30
        cutoff_time = current_time - momentum_window
        
        prices_in_window = []
        buy_volume = 0
        sell_volume = 0
        
        for trade in recent_trades:
            trade_time = self._parse_timestamp_to_unix(trade.get('timestamp'))
            if trade_time < cutoff_time:
                continue
                
            price = trade.get('price', 0)
            if price > 0:
                prices_in_window.append(price)
                
            # Track buy/sell volume
            if trade.get('side') == 'buy':
                buy_volume += trade.get('amount_usd', 0)
            else:
                sell_volume += trade.get('amount_usd', 0)
        
        # Calculate momentum
        if len(prices_in_window) >= 2:
            price_30s_ago = prices_in_window[0]
            momentum = (current_price - price_30s_ago) / price_30s_ago if price_30s_ago > 0 else 0
        else:
            momentum = 0
            
        # Calculate buy/sell ratio
        total_volume = buy_volume + sell_volume
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        
        # Check AVWAP recapture
        avwap = self.calculate_avwap(recent_trades, 60)
        above_avwap = current_price > avwap if avwap > 0 else True
        
        # Find recent peak (5 min window)
        peak_window = 300
        peak_cutoff = current_time - peak_window
        recent_peak = 0
        peak_time = 0
        
        for trade in recent_trades:
            trade_time = self._parse_timestamp_to_unix(trade.get('timestamp'))
            if trade_time < peak_cutoff:
                continue
            price = trade.get('price', 0)
            if price > recent_peak:
                recent_peak = price
                peak_time = trade_time
        
        # Check if we're too close to a recent peak
        time_since_peak = current_time - peak_time if peak_time > 0 else 999
        near_peak = time_since_peak < 30 and current_price > recent_peak * 0.95
        
        # Determine if dump protection should trigger
        is_dumping = (
            momentum < -0.15 or  # Price dropped >15% in 30s
            buy_ratio < 0.3 or   # Sells dominate (>70% of volume)
            (near_peak and not above_avwap)  # Near peak but below AVWAP
        )
        
        # Support bounce detection
        has_bounce = False
        if len(prices_in_window) >= 3:
            # Check for higher low pattern
            min_price = min(prices_in_window[:-1])
            recent_price = prices_in_window[-1]
            has_bounce = recent_price > min_price * 1.03  # 3% bounce from low
        
        result = {
            'is_dumping': is_dumping,
            'momentum_30s': momentum,
            'buy_ratio': buy_ratio,
            'above_avwap': above_avwap,
            'avwap': avwap,
            'near_peak': near_peak,
            'time_since_peak': time_since_peak,
            'has_support_bounce': has_bounce,
            'safe_to_enter': not is_dumping and (above_avwap or has_bounce)
        }
        
        if is_dumping:
            self.logger.warning(f"🚨 Dump protection triggered for {mint[:8]}... momentum:{momentum:.1%} buy_ratio:{buy_ratio:.1%}")
        
        return result
    
    def _parse_timestamp_to_unix(self, timestamp) -> float:
        """Parse timestamp to unix timestamp for consistent comparison"""
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        elif isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.timestamp()
            except:
                import time
                return time.time()
        import time
        return time.time()
    
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
        import time
        current_timestamp = time.time()
        one_minute_ago_timestamp = current_timestamp - 60
        
        recent_buys = 0
        recent_sells = 0
        
        for trade in recent_trades:
            trade_timestamp = self._parse_timestamp_to_unix(trade.get('timestamp'))
            if trade_timestamp < one_minute_ago_timestamp:
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