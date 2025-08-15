"""
Alpha wallet tracking system for detecting smart money activity
"""

import asyncio
import logging
import time
from typing import List, Set, Dict
from src.clients.moralis_client import MoralisClient
from src.utils.wallet_performance import WalletPerformanceTracker
from src.core.wallet_scorer import WalletScorer


class WalletTracker:
    def __init__(self, watched_wallets: List[str], config: Dict = None):
        self.watched_wallets = set(watched_wallets)
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize performance tracker
        self.performance_tracker = WalletPerformanceTracker()
        
        # Initialize Bayesian wallet scorer with Beta(3,5) prior
        prior_alpha = self.config.get('beta_prior_alpha', 3)
        prior_beta = self.config.get('beta_prior_beta', 5)
        self.wallet_scorer = WalletScorer(prior_alpha=prior_alpha, prior_beta=prior_beta)
        
        # Initialize tracking for all watched wallets
        for wallet in watched_wallets:
            self.performance_tracker.initialize_wallet(wallet)
        
        # Cache for recent activity
        self.recent_activity = {}
        self.cache_duration = 300  # 5 minutes
        
        # Wallet performance tracking
        self.wallet_performance = {}  # {wallet: {'trades': [], 'stats': {}}}
        self.wallet_tiers = {}  # {wallet: 'S', 'A', 'B', 'C'}
        self.tier_investment_multipliers = {
            'S': 1.00,   # 5.0% of capital ($25 on $500)
            'A': 0.90,   # 4.5% of capital ($22.50 on $500)  
            'B': 0.80,   # 4.0% of capital ($20 on $500)
            'C': 0.70    # 3.5% of capital ($17.50 on $500)
        }
        # Default for unknown/new wallets: 0.60 (3.0% = $15 on $500)
        
        # Track inactive wallets
        self.wallet_last_activity = {}  # {wallet: timestamp}
        self.inactive_threshold_hours = 6  # Mark as inactive after 6 hours of no activity
        self.inactive_wallets = set()  # Wallets that haven't traded recently
        
        # NEW: Correlation tracking for independence penalties
        self.wallet_correlation_cache = {}  # {(wallet1, wallet2): co_buy_count}
        self.shared_funder_cache = {}       # {wallet: funder_signature}
        self.correlation_update_interval = 3600  # Update hourly
        self.last_correlation_update = 0
        
        # Deduplication cache for alpha detections
        self.alpha_detection_cache = {}  # {"token:wallet": timestamp}
        self.dedup_cache_ttl = 75  # 75 seconds TTL for deduplication - FIXED: was 900
        self.dedupe_stats = {'total_checks': 0, 'deduped_checks': 0}  # Statistics
        
        # Real-time trade cache for PumpPortal trades
        self.realtime_trades_cache = {}  # {mint: [(wallet, timestamp), ...]}
        self.realtime_cache_ttl = 120  # Keep trades for 2 minutes

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
                    if swap.get('side') == 'buy' and wallet in self.watched_wallets:
                        # Simple deduplication for basic check
                        dedupe_key = f"{mint_address}:{wallet}"
                        current_time = time.time()
                        
                        self.dedupe_stats['total_checks'] += 1
                        
                        if dedupe_key in self.alpha_detection_cache:
                            last_seen = self.alpha_detection_cache[dedupe_key]
                            if current_time - last_seen < self.dedup_cache_ttl:
                                self.dedupe_stats['deduped_checks'] += 1
                                continue
                        
                        self.alpha_detection_cache[dedupe_key] = current_time
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
        
        # Clean up old deduplication entries
        self._cleanup_dedup_cache()
        
        # Log results with deduplication stats
        if len(alpha_buyers) > 0:
            dedup_saved = self.dedupe_stats['deduped_checks']
            self.logger.info(f"Alpha check complete: {len(alpha_buyers)}/{threshold_buys} alpha wallets bought {mint_address[:8]}... (saved {dedup_saved} duplicate API calls)")
        else:
            self.logger.debug(f"No alpha wallet activity found for {mint_address[:8]}...")
        return False
    
    async def check_alpha_activity_detailed(self, mint_address: str, time_window_sec: int, moralis_client=None, threshold_alpha_buys: int = 3) -> Dict:
        """
        Enhanced alpha activity check that returns detailed wallet information
        Returns: {
            'alpha_wallets': set of wallet addresses,
            'wallet_tiers': {wallet: tier},
            'confidence_score': 0-100,
            'investment_multiplier': float
        }
        """
        self.logger.debug(f"Checking detailed alpha activity for {mint_address}")
        
        start_time = time.time()
        
        # FIRST: Check real-time cache for immediate alpha buyers (PumpPortal trades)
        alpha_buyers = self.get_realtime_alpha_buyers(mint_address, time_window_sec)
        if alpha_buyers:
            self.logger.info(f"Found {len(alpha_buyers)} alpha buyers from real-time cache for {mint_address[:8]}...")
        
        if not moralis_client:
            self.logger.error("Moralis client not provided")
            return {'alpha_wallets': set(), 'wallet_tiers': {}, 'confidence_score': 0, 'investment_multiplier': 0.6}
            
        moralis = moralis_client
        
        # Optimized polling: max 3 polls with early abort conditions
        # Total: 5 + 20 + 40 = 65 seconds, only 3 API calls maximum
        poll_intervals = [5, 20, 40]
        poll_count = 0
        last_alpha_count = 0
        unique_buyers = 0
        buy_to_sell_ratio = 0
        
        for i, interval in enumerate(poll_intervals):
            # Check if we've exceeded our time window
            if time.time() - start_time >= time_window_sec:
                break
                
            try:
                # Use smaller limit (50 vs 100) and alpha-optimized TTL
                swaps = await moralis.get_token_swaps(mint_address, limit=100, ttl_override='swaps_alpha')
                poll_count += 1
                
                # Track market activity for early abort decisions
                total_buys = 0
                total_sells = 0
                unique_buyers_set = set()
                
                for swap in swaps:
                    swap_time = self._parse_timestamp(swap.get('timestamp'))
                    
                    if time.time() - swap_time > time_window_sec:
                        continue
                    
                    wallet = swap.get('wallet', '')
                    side = swap.get('side', '')
                    
                    # Track overall market activity
                    if side == 'buy':
                        total_buys += 1
                        unique_buyers_set.add(wallet)
                    elif side == 'sell':
                        total_sells += 1
                    
                    # Track alpha wallets
                    if side == 'buy' and wallet in self.watched_wallets:
                        # Deduplication check
                        dedupe_key = f"{mint_address}:{wallet}"
                        current_time = time.time()
                        
                        self.dedupe_stats['total_checks'] += 1
                        
                        # Check if we've already processed this wallet-token pair recently
                        if dedupe_key in self.alpha_detection_cache:
                            last_seen = self.alpha_detection_cache[dedupe_key]
                            if current_time - last_seen < self.dedup_cache_ttl:
                                self.dedupe_stats['deduped_checks'] += 1
                                continue  # Skip duplicate detection
                        
                        # New detection - add to cache and process
                        self.alpha_detection_cache[dedupe_key] = current_time
                        alpha_buyers.add(wallet)
                        
                        # Update wallet activity timestamp
                        self.update_wallet_activity(wallet)
                        
                        # Record wallet activity in performance tracker
                        self.performance_tracker.record_wallet_activity(wallet)
                        
                        self.logger.info(f"ALPHA WALLET DETECTED: {wallet[:8]}... bought {mint_address[:8]}...")
                
                # Calculate market metrics for early abort
                unique_buyers = len(unique_buyers_set)
                buy_to_sell_ratio = total_buys / max(total_sells, 1)  # Avoid division by zero
                
                # Early abort conditions after first poll
                if i == 0:  # First poll
                    if unique_buyers < 10 and buy_to_sell_ratio < 1.0:
                        self.logger.debug(f"Early abort: poor activity ({unique_buyers} buyers, ratio {buy_to_sell_ratio:.2f})")
                        break
                
                # Early exit if we reach threshold
                if len(alpha_buyers) >= threshold_alpha_buys:
                    dedup_saved = self.dedupe_stats['deduped_checks']
                    self.logger.info(f"Threshold reached: {len(alpha_buyers)}/{threshold_alpha_buys} alpha wallets after {poll_count} polls (saved {dedup_saved} API calls)")
                    break
                
                # Early exit optimization: if no alpha activity after 2 polls, stop
                if i >= 1 and len(alpha_buyers) == 0:  # No alpha wallets found after 2nd poll
                    self.logger.debug(f"No alpha activity detected after {poll_count} polls for {mint_address[:8]}...")
                    break
                
                last_alpha_count = len(alpha_buyers)
                
                # Wait for next interval (unless this was the last one)
                if poll_count < len(poll_intervals):
                    await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error checking alpha activity: {e}")
                await asyncio.sleep(2)
        
        # NEW: Filter out wallets that exit early (copy-trade bait detection)
        hold_through_check = self.config.get('safety', {}).get('hold_through_entry_check', True)
        if hold_through_check and alpha_buyers:
            signal_time = time.time()
            window_seconds = self.config.get('safety', {}).get('hold_through_window_seconds', 20)
            filtered_alpha_buyers = set()
            
            self.logger.debug(f"Checking {len(alpha_buyers)} alpha wallets for early exits...")
            
            for wallet in alpha_buyers:
                exits_early = await self._wallet_exits_early(wallet, mint_address, signal_time, moralis, window_seconds)
                if not exits_early:
                    filtered_alpha_buyers.add(wallet)
                else:
                    self.logger.info(f"Removed copy-trade bait: {wallet[:8]}... exits within {window_seconds}s")
            
            # Update alpha_buyers with filtered set
            original_count = len(alpha_buyers)
            alpha_buyers = filtered_alpha_buyers
            filtered_count = len(alpha_buyers)
            
            if filtered_count < original_count:
                self.logger.info(f"Hold-through filter: {original_count} -> {filtered_count} wallets "
                               f"({original_count - filtered_count} removed as copy-trade bait)")
        
        # Calculate weighted voting score using Bayesian scorer
        wallet_tiers = {}
        wallet_weights = {}
        total_weight = 0.0
        
        for wallet in alpha_buyers:
            # Get traditional tier for backward compatibility
            tier = self.get_wallet_tier(wallet)
            wallet_tiers[wallet] = tier
            
            # Calculate Bayesian weighted score
            base_score = self.wallet_scorer.get_wallet_score(wallet)
            recency_boost = self.wallet_scorer.get_recency_boost(wallet)
            size_factor = self.wallet_scorer.get_size_factor(wallet)
            
            # Combined weight for this wallet
            wallet_weight = base_score * recency_boost * size_factor
            wallet_weights[wallet] = wallet_weight
            total_weight += wallet_weight
            
            self.logger.debug(f"Wallet {wallet[:8]}... weight: {wallet_weight:.3f} "
                            f"(base={base_score:.3f}, recency={recency_boost:.2f}, size={size_factor:.2f})")
        
        # Check if we meet minimum weight threshold
        min_weight = self.config.get('alpha_weight_min', 2.5)
        require_high_confidence = self.config.get('require_one_wallet_pge_55', False)
        
        # Use smart confidence scoring that handles fresh wallets appropriately
        has_high_confidence = False
        high_confidence_wallets = []
        if require_high_confidence:
            for wallet in alpha_buyers:
                smart_confidence = self.wallet_scorer.get_smart_confidence_score(wallet)
                if smart_confidence >= 0.55:
                    has_high_confidence = True
                    high_confidence_wallets.append(wallet)
            
            # Log confidence scoring details for debugging
            confidence_details = []
            for wallet in alpha_buyers:
                smart_conf = self.wallet_scorer.get_smart_confidence_score(wallet)
                trade_count = self.wallet_scorer.get_wallet_trade_count(wallet)
                confidence_details.append(f"{wallet[:8]}({smart_conf:.2f}/tc:{trade_count})")
            
            self.logger.debug(f"Confidence check: {len(high_confidence_wallets)}/{len(alpha_buyers)} wallets >= 55%: "
                            f"[{', '.join(confidence_details)}]")
        
        # Calculate confidence score based on weighted voting
        if not alpha_buyers or total_weight < min_weight:
            return {'alpha_wallets': set(), 'wallet_tiers': {}, 'confidence_score': 0, 
                   'investment_multiplier': 0.6, 'total_weight': total_weight, 'meets_threshold': False}
        
        # Check smart confidence requirement
        if require_high_confidence and not has_high_confidence:
            # This should rarely happen now with smart scoring, but keep for safety
            avg_confidence = sum(self.wallet_scorer.get_smart_confidence_score(w) for w in alpha_buyers) / len(alpha_buyers)
            self.logger.info(f"Weight threshold met ({total_weight:.2f}) but no wallet with smart confidence >=55% "
                           f"(avg: {avg_confidence:.2f})")
            return {'alpha_wallets': alpha_buyers, 'wallet_tiers': wallet_tiers, 
                   'confidence_score': total_weight * 10, 'investment_multiplier': 0.6, 
                   'total_weight': total_weight, 'meets_threshold': False}
        
        # NEW: Apply independence penalties
        independence_penalties = self.calculate_independence_penalty(list(alpha_buyers))
        independent_wallet_count = self.get_independent_wallet_count(list(alpha_buyers))
        
        # Apply penalties to wallet weights
        adjusted_weights = {}
        total_adjusted_weight = 0.0
        
        for wallet in alpha_buyers:
            base_weight = wallet_weights.get(wallet, 1.0)
            penalty = independence_penalties.get(wallet, 1.0)
            adjusted_weight = base_weight * penalty
            
            adjusted_weights[wallet] = adjusted_weight
            total_adjusted_weight += adjusted_weight
            
            if penalty < 1.0:
                self.logger.debug(f"Applied correlation penalty to {wallet[:8]}...: "
                                f"{base_weight:.2f} -> {adjusted_weight:.2f}")
        
        # Check independence requirements
        min_independent_wallets = self.config.get('alpha_enhanced', {}).get('min_independent_wallets', 2)
        
        if independent_wallet_count < min_independent_wallets:
            self.logger.info(f"Insufficient independent wallets: {independent_wallet_count}/{min_independent_wallets}")
            return {'alpha_wallets': set(), 'wallet_tiers': {}, 'confidence_score': 0, 
                   'investment_multiplier': 0.6, 'total_weight': total_adjusted_weight, 
                   'meets_threshold': False, 'independence_failure': True}
        
        # Update weight calculations with adjusted weights
        total_weight = total_adjusted_weight
        wallet_weights = adjusted_weights  # Replace original weights with adjusted
        
        # Convert weight to confidence score (scale to 0-100)
        confidence_score = min(total_weight * 20, 100)  # Scale weight to confidence
        
        # Calculate individual wallet multipliers and get the highest one for the signal
        wallet_multipliers = {}
        for wallet in alpha_buyers:
            multiplier = self.get_wallet_investment_multiplier(wallet)
            wallet_multipliers[wallet] = multiplier
        
        # Use the highest multiplier for the overall signal strength
        best_tier_multiplier = max(wallet_multipliers.values()) if wallet_multipliers else 0.60
        
        # Calibrate wallet scores from stream data if we have swap data
        if 'swaps' in locals() and swaps:
            self.calibrate_from_stream(swaps, mint_address)
        
        # Clean up old deduplication entries
        self._cleanup_dedup_cache()
        
        dedup_saved = self.dedupe_stats['deduped_checks']
        self.logger.info(f"Alpha analysis for {mint_address[:8]}: {len(alpha_buyers)} wallets, "
                        f"weight: {total_weight:.2f}, confidence: {confidence_score:.1f}, "
                        f"multiplier: {best_tier_multiplier:.1f}x (saved {dedup_saved} duplicate calls)")
        
        return {
            'alpha_wallets': alpha_buyers,
            'wallet_tiers': wallet_tiers,
            'wallet_weights': wallet_weights,  # Bayesian weights for each wallet
            'wallet_multipliers': wallet_multipliers,  # Individual multipliers for each wallet
            'confidence_score': confidence_score,
            'investment_multiplier': best_tier_multiplier,
            'total_weight': total_weight,  # Total weighted vote
            'meets_threshold': True,  # Passed all checks
            'last_swaps_data': swaps if 'swaps' in locals() else []  # Include swap data for reuse
        }

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

    async def _wallet_exits_early(self, wallet: str, mint: str, signal_time: float, 
                                 moralis_client, window_seconds: int = 20) -> bool:
        """
        Check if wallet exits within window_seconds after signal_time
        
        Args:
            wallet: Wallet address
            mint: Token mint address  
            signal_time: Unix timestamp of our entry signal
            moralis_client: Moralis client instance
            window_seconds: Time window to check for exits (default 20s)
            
        Returns:
            True if wallet exits early (copy-trade bait), False otherwise
        """
        try:
            # Get recent swaps for the token
            swaps = await moralis_client.get_token_swaps(mint, limit=50, ttl_override='swaps_alpha')
            
            # Check for sells from this wallet within the window
            for swap in swaps:
                if swap.get('wallet') != wallet:
                    continue
                    
                swap_time = self._parse_timestamp(swap.get('timestamp'))
                
                # Check if this is a sell within our window
                if (swap.get('side') == 'sell' and 
                    signal_time <= swap_time <= signal_time + window_seconds):
                    
                    self.logger.warning(f"Copy-trade bait detected: {wallet[:8]}... sold {mint[:8]}... "
                                      f"{swap_time - signal_time:.1f}s after signal")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking early exit for {wallet[:8]}...: {e}")
            return False  # Don't penalize on API errors

    def get_wallet_tier(self, wallet_address: str) -> str:
        """Get the performance tier for a wallet (S, A, B, C)"""
        # Get tier from performance tracker (where real data lives)
        wallet_stats = self.performance_tracker.get_wallet_stats(wallet_address)
        if wallet_stats and 'current_tier' in wallet_stats:
            return wallet_stats['current_tier']
        
        # Fallback to old system if available
        if wallet_address in self.wallet_tiers:
            return self.wallet_tiers[wallet_address]
        
        # Default tier for new wallets - use 'Unknown' to get default multiplier
        return 'Unknown'
    
    def get_wallet_investment_multiplier(self, wallet_address: str) -> float:
        """Get the investment multiplier for a wallet based on its tier"""
        tier = self.get_wallet_tier(wallet_address)
        
        if tier in self.tier_investment_multipliers:
            return self.tier_investment_multipliers[tier]
        
        # Default for unknown/new wallets: 0.60 (3.0% of capital)
        return 0.60
    
    def update_wallet_tier(self, wallet_address: str, win_rate: float, avg_profit_pct: float):
        """Update wallet tier based on performance metrics"""
        if win_rate >= 0.70 and avg_profit_pct >= 100:
            tier = 'S'
        elif win_rate >= 0.60 and avg_profit_pct >= 50:
            tier = 'A'
        elif win_rate >= 0.50 and avg_profit_pct >= 20:
            tier = 'B'
        else:
            tier = 'C'
        
        old_tier = self.wallet_tiers.get(wallet_address, 'B')
        self.wallet_tiers[wallet_address] = tier
        
        if old_tier != tier:
            self.logger.info(f"Wallet {wallet_address[:8]} tier updated: {old_tier} -> {tier}")
    
    def record_wallet_trade(self, wallet_address: str, token: str, action: str, entry_price: float, 
                          exit_price: float = None, profit_pct: float = None):
        """Record a trade for performance tracking"""
        if wallet_address not in self.wallet_performance:
            self.wallet_performance[wallet_address] = {'trades': [], 'stats': {}}
        
        trade = {
            'token': token,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_pct': profit_pct,
            'timestamp': time.time()
        }
        
        self.wallet_performance[wallet_address]['trades'].append(trade)
        
        # Keep only last 30 trades per wallet to avoid memory bloat
        if len(self.wallet_performance[wallet_address]['trades']) > 30:
            self.wallet_performance[wallet_address]['trades'] = \
                self.wallet_performance[wallet_address]['trades'][-30:]
        
        # Update stats and tier if we have complete trades
        if exit_price and profit_pct is not None:
            self._update_wallet_stats(wallet_address)
    
    def _update_wallet_stats(self, wallet_address: str):
        """Update wallet statistics and tier"""
        trades = self.wallet_performance[wallet_address]['trades']
        completed_trades = [t for t in trades if t.get('profit_pct') is not None]
        
        if len(completed_trades) < 5:  # Need minimum trades for reliable stats
            return
        
        # Calculate stats
        wins = sum(1 for t in completed_trades if t['profit_pct'] > 0)
        win_rate = wins / len(completed_trades)
        avg_profit = sum(t['profit_pct'] for t in completed_trades) / len(completed_trades)
        
        self.wallet_performance[wallet_address]['stats'] = {
            'win_rate': win_rate,
            'avg_profit_pct': avg_profit,
            'total_trades': len(completed_trades),
            'last_updated': time.time()
        }
        
        # Update tier
        self.update_wallet_tier(wallet_address, win_rate, avg_profit)
    
    def get_wallet_stats(self, wallet_address: str) -> Dict:
        """Get performance stats for a wallet"""
        if wallet_address not in self.wallet_performance:
            return {}
        
        return self.wallet_performance[wallet_address].get('stats', {})

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
        # CRITICAL FIX: Initialize activity timestamp to prevent immediate inactive marking
        self.wallet_last_activity[wallet_address] = time.time()
        # Remove from inactive set if it was previously inactive
        self.inactive_wallets.discard(wallet_address)
        # Initialize performance tracking
        self.performance_tracker.initialize_wallet(wallet_address)
        self.logger.info(f"Added wallet {wallet_address[:8]}... to watch list (marked as active)")

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
    
    def update_wallet_activity(self, wallet_address: str):
        """Update the last activity timestamp for a wallet"""
        self.wallet_last_activity[wallet_address] = time.time()
        # Remove from inactive set if it was marked inactive
        self.inactive_wallets.discard(wallet_address)
    
    def update_wallet_outcome(self, wallet: str, win: bool, timestamp: float = None, trade_size: float = None):
        """
        Update Bayesian scorer with trade outcome
        
        Args:
            wallet: Wallet address
            win: True if trade was profitable
            timestamp: Unix timestamp of trade
            trade_size: Size of trade in USD
        """
        self.wallet_scorer.update_wallet_outcome(wallet, win, timestamp, trade_size)
        self.logger.debug(f"Updated Bayesian score for wallet {wallet[:8]}... with outcome: {'WIN' if win else 'LOSS'}")
    
    def calibrate_from_stream(self, swaps: List[Dict], mint_address: str, time_window: int = 900) -> None:
        """
        Calibrate wallet scores from stream data by observing price movements
        
        Args:
            swaps: List of swap transactions from stream
            mint_address: Token mint address
            time_window: Time window to observe outcomes (default 15 minutes)
        """
        if not swaps:
            return
            
        current_time = time.time()
        
        # Calculate VWAP from recent trades (last 20 trades)
        recent_trades = []
        for swap in swaps[-20:]:
            if swap.get('side') in ['buy', 'sell']:
                price = swap.get('price', 0)
                amount = swap.get('amount', 0)
                if price > 0 and amount > 0:
                    recent_trades.append({'price': price, 'amount': amount})
        
        if len(recent_trades) < 5:
            return  # Not enough data for VWAP
            
        # Calculate VWAP
        total_value = sum(t['price'] * t['amount'] for t in recent_trades)
        total_volume = sum(t['amount'] for t in recent_trades)
        vwap = total_value / total_volume if total_volume > 0 else 0
        
        if vwap == 0:
            return
            
        # Track wallet buys and their entry prices
        wallet_entries = {}
        
        for swap in swaps:
            swap_time = self._parse_timestamp(swap.get('timestamp'))
            
            # Only look at recent transactions
            if current_time - swap_time > time_window:
                continue
                
            wallet = swap.get('wallet', '')
            side = swap.get('side', '')
            price = swap.get('price', 0)
            
            if wallet in self.watched_wallets and side == 'buy' and price > 0:
                if wallet not in wallet_entries:
                    wallet_entries[wallet] = {
                        'entry_price': price,
                        'entry_time': swap_time
                    }
        
        # Check outcomes for wallets that bought
        for wallet, entry_data in wallet_entries.items():
            entry_price = entry_data['entry_price']
            entry_time = entry_data['entry_time']
            
            # Check if enough time has passed to evaluate
            time_elapsed = current_time - entry_time
            if time_elapsed < 60:  # Need at least 1 minute
                continue
                
            # Calculate performance relative to entry
            price_change = (vwap - entry_price) / entry_price if entry_price > 0 else 0
            
            # Define win as VWAP +20% before -8% within observation window
            win = price_change >= 0.20  # 20% gain
            loss = price_change <= -0.08  # 8% loss
            
            if win or loss:
                # Update the Bayesian scorer with observed outcome
                self.update_wallet_outcome(wallet, win, entry_time)
                
                self.logger.info(f"Stream calibration: Wallet {wallet[:8]}... "
                               f"{'WIN' if win else 'LOSS'} (change: {price_change:.1%})")
        
    def check_inactive_wallets(self):
        """Check for wallets that haven't been active and mark them as inactive"""
        current_time = time.time()
        inactive_threshold_seconds = self.inactive_threshold_hours * 3600
        
        newly_inactive = []
        for wallet in self.watched_wallets:
            last_activity = self.wallet_last_activity.get(wallet, 0)
            
            # SAFETY: If wallet has no activity record, initialize it as active (prevents immediate inactive marking)
            if last_activity == 0:
                self.wallet_last_activity[wallet] = current_time
                self.logger.debug(f"Initialized activity timestamp for wallet {wallet[:8]}... (newly discovered)")
                continue
            
            # If activity too old, mark as inactive
            if current_time - last_activity > inactive_threshold_seconds:
                if wallet not in self.inactive_wallets:
                    self.inactive_wallets.add(wallet)
                    newly_inactive.append(wallet)
                    self.logger.info(f"Wallet {wallet[:8]}... marked as inactive (no activity for {self.inactive_threshold_hours}h)")
        
        return newly_inactive
    
    def get_active_wallets(self) -> Set[str]:
        """Get set of currently active wallets (excludes inactive ones)"""
        self.check_inactive_wallets()  # Update inactive status first
        return self.watched_wallets - self.inactive_wallets
    
    def get_inactive_wallets(self) -> Set[str]:
        """Get set of currently inactive wallets"""
        self.check_inactive_wallets()  # Update inactive status first
        return self.inactive_wallets.copy()
    
    def _cleanup_dedup_cache(self):
        """Clean up old entries from deduplication cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.alpha_detection_cache.items()
            if current_time - timestamp > self.dedup_cache_ttl
        ]
        
        for key in expired_keys:
            del self.alpha_detection_cache[key]
            
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired deduplication entries")
    
    def record_realtime_alpha_buy(self, wallet: str, mint: str, timestamp: str):
        """Record a real-time alpha wallet buy from PumpPortal"""
        import time
        
        # Clean up old cache entries
        self._cleanup_realtime_cache()
        
        # Initialize list for this mint if needed
        if mint not in self.realtime_trades_cache:
            self.realtime_trades_cache[mint] = []
        
        # Add the trade
        trade_time = self._parse_timestamp(timestamp) if isinstance(timestamp, str) else timestamp
        self.realtime_trades_cache[mint].append((wallet, trade_time))
        
        # Update wallet activity
        self.update_wallet_activity(wallet)
        self.performance_tracker.record_wallet_activity(wallet)
        
        self.logger.debug(f"Recorded realtime alpha buy: {wallet[:8]}... -> {mint[:8]}...")
    
    def _cleanup_realtime_cache(self):
        """Remove old entries from real-time trades cache"""
        current_time = time.time()
        
        # Clean up trades older than TTL
        for mint in list(self.realtime_trades_cache.keys()):
            trades = self.realtime_trades_cache[mint]
            # Keep only recent trades
            recent_trades = [(w, t) for w, t in trades if current_time - t < self.realtime_cache_ttl]
            
            if recent_trades:
                self.realtime_trades_cache[mint] = recent_trades
            else:
                del self.realtime_trades_cache[mint]
    
    def get_realtime_alpha_buyers(self, mint: str, time_window_sec: int) -> set:
        """Get alpha wallet buyers from real-time cache"""
        import time
        current_time = time.time()
        alpha_buyers = set()
        
        if mint in self.realtime_trades_cache:
            for wallet, trade_time in self.realtime_trades_cache[mint]:
                if current_time - trade_time <= time_window_sec:
                    if wallet in self.watched_wallets:
                        alpha_buyers.add(wallet)
        
        return alpha_buyers
    
    def get_deduplication_stats(self) -> Dict:
        """Get deduplication statistics"""
        total = self.dedupe_stats['total_checks']
        deduped = self.dedupe_stats['deduped_checks']
        
        return {
            'total_checks': total,
            'deduped_checks': deduped,
            'savings_pct': (deduped / total * 100) if total > 0 else 0,
            'cache_size': len(self.alpha_detection_cache)
        }
    
    def reset_dedup_stats(self):
        """Reset deduplication statistics"""
        self.dedupe_stats = {'total_checks': 0, 'deduped_checks': 0}
    
    def record_trade_follow(self, alpha_wallets: Set[str], mint_address: str, confidence_score: float):
        """Record that we followed these wallets' trades"""
        for wallet in alpha_wallets:
            self.performance_tracker.record_trade_follow(wallet, mint_address, confidence_score)
    
    def record_trade_outcome(self, mint_address: str, pnl_pct: float):
        """Record the outcome of a trade for performance tracking"""
        self.performance_tracker.record_trade_outcome(mint_address, pnl_pct)
    
    def get_wallets_for_rotation(self):
        """Get wallet categorization for rotation decisions"""
        return self.performance_tracker.get_wallets_for_rotation()
    
    def get_performance_summary(self):
        """Get performance tracking summary"""
        return self.performance_tracker.get_summary()
    
    def recalculate_all_wallet_tiers(self):
        """Recalculate tiers for all wallets based on performance data"""
        return self.performance_tracker.recalculate_all_tiers()
    
    def get_tier_performance_stats(self):
        """Get performance statistics grouped by wallet tiers"""
        return self.performance_tracker.get_tier_performance_stats()

    def update_wallet_correlation(self, wallets_in_signal: list, mint: str, 
                                buy_timestamps: dict):
        """
        Update correlation tracking when wallets co-buy
        
        Args:
            wallets_in_signal: List of wallets that bought this signal
            mint: Token mint address  
            buy_timestamps: {wallet: timestamp} mapping
        """
        # Find wallets that bought within 250ms of each other
        time_threshold = self.config.get('alpha_enhanced', {}).get('correlation_time_threshold_ms', 250) / 1000.0
        
        for i, wallet1 in enumerate(wallets_in_signal):
            for wallet2 in wallets_in_signal[i+1:]:
                time1 = buy_timestamps.get(wallet1, 0)
                time2 = buy_timestamps.get(wallet2, 0)
                
                # Check if they bought close in time
                if abs(time1 - time2) <= time_threshold:
                    # Create sorted pair key for consistent caching
                    pair_key = tuple(sorted([wallet1, wallet2]))
                    
                    # Increment co-buy count
                    self.wallet_correlation_cache[pair_key] = (
                        self.wallet_correlation_cache.get(pair_key, 0) + 1
                    )
                    
                    self.logger.debug(f"Co-buy detected: {wallet1[:8]}... & {wallet2[:8]}... "
                                    f"({abs(time1 - time2)*1000:.0f}ms apart)")

    def calculate_independence_penalty(self, wallets: list) -> dict:
        """
        Calculate correlation penalty for each wallet in the signal
        
        Args:
            wallets: List of wallets in current signal
            
        Returns:
            Dict of {wallet: penalty_factor} where factor is 0.0-1.0
        """
        penalties = {}
        config = self.config.get('alpha_enhanced', {})
        
        for wallet in wallets:
            penalty_factors = []
            
            # Check co-buying correlation with other wallets in signal
            for other_wallet in wallets:
                if wallet != other_wallet:
                    pair_key = tuple(sorted([wallet, other_wallet]))
                    co_buy_count = self.wallet_correlation_cache.get(pair_key, 0)
                    
                    # Apply penalty based on co-buy frequency
                    if co_buy_count >= 5:  # Frequent co-buyers
                        penalty = config.get('correlation_penalty_frequent', 0.8)
                        penalty_factors.append(penalty)
                    elif co_buy_count >= 3:  # Some co-buying
                        penalty = config.get('correlation_penalty_some', 0.9)
                        penalty_factors.append(penalty)
            
            # Check shared funder penalty (if available)
            wallet_funder = self.shared_funder_cache.get(wallet)
            if wallet_funder:
                for other_wallet in wallets:
                    if (wallet != other_wallet and 
                        self.shared_funder_cache.get(other_wallet) == wallet_funder):
                        penalty = config.get('correlation_penalty_shared_funder', 0.8)
                        penalty_factors.append(penalty)
            
            # Calculate final penalty (multiply all factors)
            final_penalty = 1.0
            for factor in penalty_factors:
                final_penalty *= factor
            
            penalties[wallet] = final_penalty
        
        return penalties

    def get_independent_wallet_count(self, wallets: list) -> int:
        """
        Count wallets with minimal correlation (independence > 0.67)
        
        Args:
            wallets: List of wallets to check
            
        Returns:
            Count of independent wallets
        """
        penalties = self.calculate_independence_penalty(wallets)
        
        independent_count = sum(
            1 for penalty in penalties.values() 
            if penalty > 0.67  # More than 67% independence
        )
        
        return independent_count