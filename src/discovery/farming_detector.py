"""
Farming Wallet Detection Module
Detects and filters out farming/manipulation wallets using advanced metrics
Based on GPT5's comprehensive farming detection methodology
"""

import logging
import time
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math


class FarmingDetector:
    """Detects farming wallets that manipulate copy traders"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Load thresholds from config or use defaults
        farming_config = self.config.get('farming_detection', {})
        
        # 1. Holding Time Thresholds
        self.max_short_hold_rate = farming_config.get('max_short_hold_rate', 0.70)  # 70% trades < 5min
        self.median_hold_threshold = farming_config.get('median_hold_threshold', 180)  # 3 minutes
        self.short_hold_window = farming_config.get('short_hold_window', 300)  # 5 minutes
        
        # 2. Entry Crowding Thresholds
        self.entry_disadvantage_threshold = farming_config.get('entry_disadvantage_threshold', -0.03)  # -3%
        self.entry_25pct_threshold = farming_config.get('entry_25pct_threshold', -0.05)  # -5%
        
        # 3. Follower Disadvantage Thresholds
        self.max_follower_disadvantage = farming_config.get('max_follower_disadvantage', 0.05)  # 5%
        self.max_75pct_follower_disadvantage = farming_config.get('max_75pct_follower_disadvantage', 0.08)  # 8%
        
        # 4. Exit Impact Thresholds
        self.max_exit_impact = farming_config.get('max_exit_impact', 0.25)  # 25% of pool volume
        self.high_exit_impact_rate = farming_config.get('high_exit_impact_rate', 0.50)  # 50% of exits
        
        # 5. Trade Synchrony Thresholds
        self.max_synchrony_rate = farming_config.get('max_synchrony_rate', 0.60)  # 60% trades synchronized
        self.synchrony_window = farming_config.get('synchrony_window', 5)  # 5 seconds
        self.min_correlation = farming_config.get('min_correlation', 0.70)  # 0.7 correlation
        
        # 6. Funding Entropy Thresholds
        self.min_entropy = farming_config.get('min_entropy', 1.0)  # Shannon entropy threshold
        self.max_single_source_rate = farming_config.get('max_single_source_rate', 0.60)  # 60% from one source
        
        # 7. Trade Size Uniformity
        self.max_size_uniformity = farming_config.get('max_size_uniformity', 0.50)  # 50% within 2%
        self.size_tolerance = farming_config.get('size_tolerance', 0.02)  # 2% tolerance
        self.max_coefficient_variation = farming_config.get('max_coefficient_variation', 0.1)  # CV < 0.1
        
        # 8. Win Rate vs Hold Time
        self.suspicious_win_rate = farming_config.get('suspicious_win_rate', 0.80)  # 80% win rate
        self.ultra_suspicious_win_rate = farming_config.get('ultra_suspicious_win_rate', 0.90)  # 90%
        
        # 9. Liquidity Contribution
        self.min_lp_events = farming_config.get('min_lp_events', 1)  # At least 1 LP event per 50 trades
        
        # 10. Token Survival
        self.min_survival_rate = farming_config.get('min_survival_rate', 0.20)  # 20% survival
        self.critical_survival_rate = farming_config.get('critical_survival_rate', 0.10)  # 10% critical
        
        # Overall farmer score thresholds
        self.farmer_threshold = farming_config.get('farmer_threshold', 0.60)  # Score > 0.6 = farmer
        self.suspicious_threshold = farming_config.get('suspicious_threshold', 0.40)  # 0.4-0.6 = suspicious
        
        self.logger.info(f"Farming detector initialized with thresholds: farmer>{self.farmer_threshold}, suspicious>{self.suspicious_threshold}")
    
    async def analyze_wallet(self, wallet: str, trades_data: List[Dict], moralis_client=None, bitquery_client=None) -> Dict:
        """
        Comprehensive farming analysis for a single wallet
        Returns detailed metrics and farmer score
        """
        metrics = {
            'wallet': wallet,
            'total_trades': len(trades_data),
            'farmer_score': 0.0,
            'classification': 'safe',
            'red_flags': [],
            'detailed_metrics': {}
        }
        
        if not trades_data:
            return metrics
        
        try:
            # 1. Holding Time Analysis
            holding_metrics = self._analyze_holding_patterns(trades_data)
            metrics['detailed_metrics']['holding'] = holding_metrics
            
            # 2. Entry Crowding Analysis
            entry_metrics = await self._analyze_entry_crowding(wallet, trades_data, bitquery_client)
            metrics['detailed_metrics']['entry_crowding'] = entry_metrics
            
            # 3. Follower Disadvantage Analysis
            follower_metrics = self._calculate_follower_disadvantage(trades_data)
            metrics['detailed_metrics']['follower_disadvantage'] = follower_metrics
            
            # 4. Exit Impact Analysis
            exit_metrics = await self._analyze_exit_impact(wallet, trades_data, bitquery_client)
            metrics['detailed_metrics']['exit_impact'] = exit_metrics
            
            # 5. Trade Synchrony Analysis
            synchrony_metrics = self._analyze_trade_synchrony(wallet, trades_data)
            metrics['detailed_metrics']['synchrony'] = synchrony_metrics
            
            # 6. Funding Entropy Analysis
            entropy_metrics = await self._analyze_funding_entropy(wallet, moralis_client)
            metrics['detailed_metrics']['funding_entropy'] = entropy_metrics
            
            # 7. Trade Size Uniformity
            uniformity_metrics = self._analyze_trade_uniformity(trades_data)
            metrics['detailed_metrics']['uniformity'] = uniformity_metrics
            
            # 8. Win Rate vs Hold Time Cross-Check
            winrate_metrics = self._analyze_winrate_holding_correlation(trades_data, holding_metrics)
            metrics['detailed_metrics']['winrate_holding'] = winrate_metrics
            
            # 9. Liquidity Contribution Check
            liquidity_metrics = await self._check_liquidity_contribution(wallet, trades_data, moralis_client)
            metrics['detailed_metrics']['liquidity'] = liquidity_metrics
            
            # 10. Token Survival Analysis
            survival_metrics = await self._analyze_token_survival(trades_data, moralis_client)
            metrics['detailed_metrics']['survival'] = survival_metrics
            
            # Calculate composite farmer score
            farmer_score = self._calculate_farmer_score(metrics['detailed_metrics'])
            metrics['farmer_score'] = farmer_score
            
            # Classify wallet
            if farmer_score >= self.farmer_threshold:
                metrics['classification'] = 'farmer'
                metrics['red_flags'].append(f"High farmer score: {farmer_score:.2f}")
            elif farmer_score >= self.suspicious_threshold:
                metrics['classification'] = 'suspicious'
                metrics['red_flags'].append(f"Suspicious activity score: {farmer_score:.2f}")
            else:
                metrics['classification'] = 'safe'
            
            # Log only suspicious/farmed wallets, not every wallet
            if metrics['classification'] in ['farmed', 'suspicious']:
                self.logger.info(f"Wallet {wallet[:8]}... farming analysis: "
                               f"score={farmer_score:.2f}, class={metrics['classification']}, "
                               f"flags={len(metrics['red_flags'])}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing wallet {wallet[:8]}... for farming: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _analyze_holding_patterns(self, trades_data: List[Dict]) -> Dict:
        """Analyze holding time patterns to detect ultra-fast flipping"""
        holding_times = []
        profitable_trades = []
        
        # Group trades by token to find buy-sell pairs
        token_trades = defaultdict(list)
        for trade in trades_data:
            token = trade.get('token_address', trade.get('mint', ''))
            if token:
                token_trades[token].append(trade)
        
        for token, trades in token_trades.items():
            # Sort by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
            
            # Find buy-sell pairs
            buy_time = None
            buy_price = None
            
            for trade in sorted_trades:
                if trade.get('side') == 'buy' and buy_time is None:
                    buy_time = trade.get('timestamp', 0)
                    buy_price = trade.get('price', 0)
                elif trade.get('side') == 'sell' and buy_time is not None:
                    sell_time = trade.get('timestamp', 0)
                    sell_price = trade.get('price', 0)
                    
                    # Calculate holding time
                    hold_time = sell_time - buy_time
                    holding_times.append(hold_time)
                    
                    # Check if profitable
                    if buy_price > 0 and sell_price > buy_price:
                        profitable_trades.append({
                            'hold_time': hold_time,
                            'profit_mult': sell_price / buy_price
                        })
                    
                    # Reset for next pair
                    buy_time = None
                    buy_price = None
        
        # Calculate metrics
        metrics = {
            'total_roundtrips': len(holding_times),
            'median_hold_time': 0,
            'short_hold_rate': 0,
            'ultra_short_hold_rate': 0,
            'profitable_short_hold_rate': 0,
            'is_farming_pattern': False
        }
        
        if holding_times:
            metrics['median_hold_time'] = statistics.median(holding_times)
            
            # Calculate rates
            short_holds = sum(1 for t in holding_times if t < self.short_hold_window)
            ultra_short = sum(1 for t in holding_times if t < 60)  # < 1 minute
            
            metrics['short_hold_rate'] = short_holds / len(holding_times)
            metrics['ultra_short_hold_rate'] = ultra_short / len(holding_times)
            
            # Check profitable trades holding patterns
            if profitable_trades:
                prof_short = sum(1 for t in profitable_trades if t['hold_time'] < self.short_hold_window)
                metrics['profitable_short_hold_rate'] = prof_short / len(profitable_trades)
            
            # Detect farming pattern
            if (metrics['short_hold_rate'] > self.max_short_hold_rate or
                metrics['median_hold_time'] < self.median_hold_threshold or
                metrics['profitable_short_hold_rate'] > 0.70):
                metrics['is_farming_pattern'] = True
        
        return metrics
    
    async def _analyze_entry_crowding(self, wallet: str, trades_data: List[Dict], bitquery_client) -> Dict:
        """Analyze if wallet consistently enters before crowd at better prices"""
        metrics = {
            'avg_entry_advantage': 0,
            'median_entry_advantage': 0,
            'p25_entry_advantage': 0,
            'crowding_detected': False,
            'sample_size': 0
        }
        
        if not bitquery_client:
            return metrics
        
        entry_advantages = []
        
        for trade in trades_data[:20]:  # Sample first 20 trades
            if trade.get('side') != 'buy':
                continue
                
            token = trade.get('token_address', trade.get('mint', ''))
            entry_time = trade.get('timestamp', 0)
            entry_price = trade.get('price', 0)
            
            if not token or not entry_time or not entry_price:
                continue
            
            try:
                # Get trades in next 30-60 seconds
                followers = await bitquery_client.get_token_trades_in_window(
                    token, 
                    entry_time, 
                    entry_time + 60
                )
                
                if len(followers) > 5:  # Need sufficient followers
                    follower_prices = [f.get('price', 0) for f in followers if f.get('price', 0) > 0]
                    
                    if follower_prices:
                        median_follower_price = statistics.median(follower_prices)
                        
                        # Calculate entry advantage (negative = wallet got better price)
                        entry_advantage = (entry_price - median_follower_price) / median_follower_price
                        entry_advantages.append(entry_advantage)
                        
            except Exception as e:
                self.logger.debug(f"Error analyzing entry crowding: {e}")
                continue
        
        if entry_advantages:
            metrics['sample_size'] = len(entry_advantages)
            metrics['avg_entry_advantage'] = statistics.mean(entry_advantages)
            metrics['median_entry_advantage'] = statistics.median(entry_advantages)
            
            # Calculate 25th percentile
            sorted_advantages = sorted(entry_advantages)
            p25_index = len(sorted_advantages) // 4
            metrics['p25_entry_advantage'] = sorted_advantages[p25_index]
            
            # Detect crowding pattern
            if (metrics['median_entry_advantage'] < self.entry_disadvantage_threshold or
                metrics['p25_entry_advantage'] < self.entry_25pct_threshold):
                metrics['crowding_detected'] = True
        
        return metrics
    
    def _calculate_follower_disadvantage(self, trades_data: List[Dict]) -> Dict:
        """Calculate disadvantage for followers copying trades"""
        metrics = {
            'median_follower_disadvantage': 0,
            'p75_follower_disadvantage': 0,
            'high_disadvantage_rate': 0,
            'follower_trap_detected': False
        }
        
        # Simulate follower entries with realistic latency
        follower_disadvantages = []
        
        # Group by token
        token_trades = defaultdict(list)
        for trade in trades_data:
            token = trade.get('token_address', trade.get('mint', ''))
            if token and trade.get('side') == 'buy':
                token_trades[token].append(trade)
        
        for token, buys in token_trades.items():
            if not buys:
                continue
                
            # Sort by time
            sorted_buys = sorted(buys, key=lambda x: x.get('timestamp', 0))
            
            for i, buy in enumerate(sorted_buys):
                wallet_entry_price = buy.get('price', 0)
                
                # Find next few trades (followers)
                followers = sorted_buys[i+1:i+10]  # Next 10 trades
                
                if followers and wallet_entry_price > 0:
                    # Calculate average follower entry price
                    follower_prices = [f.get('price', 0) for f in followers if f.get('price', 0) > 0]
                    
                    if follower_prices:
                        avg_follower_price = statistics.mean(follower_prices)
                        
                        # Calculate disadvantage (how much worse followers did)
                        disadvantage = (avg_follower_price - wallet_entry_price) / wallet_entry_price
                        follower_disadvantages.append(disadvantage)
        
        if follower_disadvantages:
            metrics['median_follower_disadvantage'] = statistics.median(follower_disadvantages)
            
            # Calculate 75th percentile
            sorted_disadvantages = sorted(follower_disadvantages)
            p75_index = int(len(sorted_disadvantages) * 0.75)
            metrics['p75_follower_disadvantage'] = sorted_disadvantages[p75_index]
            
            # Calculate high disadvantage rate
            high_disadvantage = sum(1 for d in follower_disadvantages if d > self.max_follower_disadvantage)
            metrics['high_disadvantage_rate'] = high_disadvantage / len(follower_disadvantages)
            
            # Detect follower trap pattern
            if (metrics['median_follower_disadvantage'] > self.max_follower_disadvantage or
                metrics['p75_follower_disadvantage'] > self.max_75pct_follower_disadvantage or
                metrics['high_disadvantage_rate'] > 0.50):
                metrics['follower_trap_detected'] = True
        
        return metrics
    
    async def _analyze_exit_impact(self, wallet: str, trades_data: List[Dict], bitquery_client) -> Dict:
        """Analyze wallet's impact on pool during exits - FIXED to use price impact"""
        metrics = {
            'avg_exit_impact': 0,
            'high_impact_rate': 0,
            'max_exit_impact': 0,
            'dump_pattern_detected': False,
            'sample_size': 0
        }
        
        if not bitquery_client:
            return metrics
        
        exit_impacts = []
        
        # Analyze sell trades
        sell_trades = [t for t in trades_data if t.get('side') == 'sell'][:20]  # Sample
        
        for trade in sell_trades:
            token = trade.get('token_address', trade.get('mint', ''))
            exit_time = trade.get('timestamp', 0)
            exit_amount = trade.get('amount_usd', 0)
            
            if not token or not exit_time or not exit_amount:
                continue
            
            try:
                # FIXED: Get price before and after the sell to measure actual impact
                # Get price 5 seconds before the sell
                price_before_trades = await bitquery_client.get_token_trades_in_window(
                    token,
                    exit_time - 10,
                    exit_time - 5
                )
                
                # Get price 10 seconds after the sell
                price_after_trades = await bitquery_client.get_token_trades_in_window(
                    token,
                    exit_time + 5,
                    exit_time + 15
                )
                
                if price_before_trades and price_after_trades:
                    # Calculate average prices
                    prices_before = [t.get('price', 0) for t in price_before_trades if t.get('price', 0) > 0]
                    prices_after = [t.get('price', 0) for t in price_after_trades if t.get('price', 0) > 0]
                    
                    if prices_before and prices_after:
                        avg_price_before = statistics.mean(prices_before)
                        avg_price_after = statistics.mean(prices_after)
                        
                        # Calculate actual price impact
                        if avg_price_before > 0:
                            price_impact = (avg_price_before - avg_price_after) / avg_price_before
                            exit_impacts.append(abs(price_impact))  # Use absolute value for impact
                        
            except Exception as e:
                self.logger.debug(f"Error analyzing exit impact: {e}")
                continue
        
        if exit_impacts:
            metrics['sample_size'] = len(exit_impacts)
            metrics['avg_exit_impact'] = statistics.mean(exit_impacts)
            metrics['max_exit_impact'] = max(exit_impacts)
            
            # Calculate high impact rate (price drops > 25%)
            high_impact = sum(1 for i in exit_impacts if i > 0.25)
            metrics['high_impact_rate'] = high_impact / len(exit_impacts)
            
            # Detect dump pattern - adjusted thresholds for price impact
            if (metrics['avg_exit_impact'] > 0.15 or  # Average 15% price drop
                metrics['high_impact_rate'] > 0.30 or  # 30% of exits cause >25% drops
                metrics['max_exit_impact'] > 0.35):    # Any exit causing >35% drop
                metrics['dump_pattern_detected'] = True
        
        return metrics
    
    def _analyze_trade_synchrony(self, wallet: str, trades_data: List[Dict]) -> Dict:
        """Detect synchronized trading patterns (bot rings)"""
        metrics = {
            'synchrony_rate': 0,
            'max_correlation': 0,
            'synchronized_wallets': [],
            'bot_ring_detected': False
        }
        
        # Build timestamp clusters
        trade_timestamps = defaultdict(list)
        
        for trade in trades_data:
            timestamp = trade.get('timestamp', 0)
            block = trade.get('block_number', trade.get('block', 0))
            other_wallet = trade.get('other_wallet')  # If available from transaction data
            
            if timestamp:
                # Round to nearest second for clustering
                cluster_time = round(timestamp)
                trade_timestamps[cluster_time].append({
                    'wallet': other_wallet,
                    'block': block
                })
        
        # Count synchronized trades
        synchronized_count = 0
        total_trades = len(trades_data)
        
        for cluster_time, cluster_trades in trade_timestamps.items():
            if len(cluster_trades) > 1:
                # Multiple trades in same second
                synchronized_count += len(cluster_trades)
                
                # Check for same block trades (high synchrony)
                blocks = [t['block'] for t in cluster_trades if t['block']]
                if blocks and len(set(blocks)) == 1:
                    # All in same block - very suspicious
                    synchronized_count += len(cluster_trades)  # Double count
        
        if total_trades > 0:
            metrics['synchrony_rate'] = synchronized_count / total_trades
            
            # Detect bot ring pattern
            if metrics['synchrony_rate'] > self.max_synchrony_rate:
                metrics['bot_ring_detected'] = True
        
        return metrics
    
    async def _analyze_funding_entropy(self, wallet: str, moralis_client) -> Dict:
        """Analyze diversity of funding sources (entropy)"""
        metrics = {
            'shannon_entropy': 0,
            'dominant_source_rate': 0,
            'unique_sources': 0,
            'centralized_funding': False
        }
        
        if not moralis_client:
            return metrics
        
        try:
            # Get wallet transfers/funding history
            transfers = await moralis_client.get_wallet_transfers(wallet, limit=100)
            
            # Count funding sources
            source_counts = defaultdict(int)
            total_transfers = 0
            
            for transfer in transfers:
                if transfer.get('direction') == 'receive':
                    source = transfer.get('from_address', '')
                    if source:
                        source_counts[source] += 1
                        total_transfers += 1
            
            if source_counts and total_transfers > 0:
                # Calculate Shannon entropy
                entropy = 0
                max_source_rate = 0
                
                for source, count in source_counts.items():
                    p = count / total_transfers
                    if p > 0:
                        entropy -= p * math.log2(p)
                    max_source_rate = max(max_source_rate, p)
                
                metrics['shannon_entropy'] = entropy
                metrics['dominant_source_rate'] = max_source_rate
                metrics['unique_sources'] = len(source_counts)
                
                # Detect centralized funding
                if (entropy < self.min_entropy or
                    max_source_rate > self.max_single_source_rate):
                    metrics['centralized_funding'] = True
                    
        except Exception as e:
            self.logger.debug(f"Error analyzing funding entropy: {e}")
        
        return metrics
    
    def _analyze_trade_uniformity(self, trades_data: List[Dict]) -> Dict:
        """Analyze uniformity of trade sizes (spam detection)"""
        metrics = {
            'size_uniformity_rate': 0,
            'coefficient_of_variation': 0,
            'most_common_size': 0,
            'spam_pattern_detected': False
        }
        
        # Get trade sizes
        trade_sizes = []
        for trade in trades_data:
            size = trade.get('amount_usd', trade.get('amount', 0))
            if size > 0:
                trade_sizes.append(float(size))
        
        if len(trade_sizes) < 3:
            return metrics
        
        # Calculate statistics
        mean_size = statistics.mean(trade_sizes)
        stdev_size = statistics.stdev(trade_sizes)
        
        if mean_size > 0:
            metrics['coefficient_of_variation'] = stdev_size / mean_size
        
        # Check for uniform sizes
        size_buckets = defaultdict(int)
        for size in trade_sizes:
            # Round to nearest 1% bucket
            bucket = round(size / mean_size * 100)
            size_buckets[bucket] += 1
        
        # Find most common size
        if size_buckets:
            most_common_bucket = max(size_buckets, key=size_buckets.get)
            most_common_count = size_buckets[most_common_bucket]
            
            metrics['size_uniformity_rate'] = most_common_count / len(trade_sizes)
            metrics['most_common_size'] = most_common_bucket * mean_size / 100
            
            # Detect spam pattern
            if (metrics['size_uniformity_rate'] > self.max_size_uniformity or
                metrics['coefficient_of_variation'] < self.max_coefficient_variation):
                metrics['spam_pattern_detected'] = True
        
        return metrics
    
    def _analyze_winrate_holding_correlation(self, trades_data: List[Dict], holding_metrics: Dict) -> Dict:
        """Cross-check win rate against holding patterns"""
        metrics = {
            'win_rate': 0,
            'avg_profit_multiplier': 0,
            'unrealistic_performance': False
        }
        
        # Calculate win rate
        wins = 0
        total_roundtrips = 0
        profit_multipliers = []
        
        # Group by token
        token_trades = defaultdict(list)
        for trade in trades_data:
            token = trade.get('token_address', trade.get('mint', ''))
            if token:
                token_trades[token].append(trade)
        
        for token, trades in token_trades.items():
            sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
            
            buy_price = None
            for trade in sorted_trades:
                if trade.get('side') == 'buy' and buy_price is None:
                    buy_price = trade.get('price', 0)
                elif trade.get('side') == 'sell' and buy_price is not None:
                    sell_price = trade.get('price', 0)
                    
                    if buy_price > 0 and sell_price > 0:
                        total_roundtrips += 1
                        profit_mult = sell_price / buy_price
                        profit_multipliers.append(profit_mult)
                        
                        if profit_mult > 1.0:
                            wins += 1
                    
                    buy_price = None
        
        if total_roundtrips > 0:
            metrics['win_rate'] = wins / total_roundtrips
            
        if profit_multipliers:
            metrics['avg_profit_multiplier'] = statistics.mean(profit_multipliers)
        
        # Cross-check with holding patterns
        median_hold = holding_metrics.get('median_hold_time', 0)
        
        # Detect unrealistic performance
        if median_hold < self.short_hold_window:
            if metrics['win_rate'] > self.suspicious_win_rate:
                metrics['unrealistic_performance'] = True
        
        if (metrics['win_rate'] > self.ultra_suspicious_win_rate and
            metrics['avg_profit_multiplier'] > 2.0):
            metrics['unrealistic_performance'] = True
        
        return metrics
    
    async def _check_liquidity_contribution(self, wallet: str, trades_data: List[Dict], moralis_client) -> Dict:
        """Check if wallet ever adds liquidity or only extracts"""
        metrics = {
            'lp_events': 0,
            'liquidity_ratio': 0,
            'pure_extractor': False
        }
        
        if not moralis_client:
            return metrics
        
        try:
            # Get LP events for wallet
            # This would need to be implemented in moralis_client
            # For now, use trade pattern as proxy
            
            total_trades = len(trades_data)
            
            # Check for LP-like patterns (large symmetric trades)
            potential_lp = 0
            for trade in trades_data:
                amount = trade.get('amount_usd', 0)
                if amount > 10000:  # Large trade that could be LP
                    potential_lp += 1
            
            metrics['lp_events'] = potential_lp
            metrics['liquidity_ratio'] = potential_lp / max(total_trades, 1)
            
            # Detect pure extractor
            if total_trades > 50 and potential_lp < self.min_lp_events:
                metrics['pure_extractor'] = True
                
        except Exception as e:
            self.logger.debug(f"Error checking liquidity contribution: {e}")
        
        return metrics
    
    async def _analyze_token_survival(self, trades_data: List[Dict], moralis_client) -> Dict:
        """Analyze survival rate of tokens traded"""
        metrics = {
            'tokens_traded': 0,
            'tokens_survived_24h': 0,
            'survival_rate': 0,
            'pump_and_dump_pattern': False
        }
        
        if not moralis_client:
            return metrics
        
        # Get unique tokens
        tokens = set()
        for trade in trades_data:
            token = trade.get('token_address', trade.get('mint', ''))
            if token:
                tokens.add(token)
        
        metrics['tokens_traded'] = len(tokens)
        
        if not tokens:
            return metrics
        
        # Check survival (sample for performance)
        sample_size = min(20, len(tokens))
        sample_tokens = list(tokens)[:sample_size]
        
        survived = 0
        for token in sample_tokens:
            try:
                # Check if token still has price/liquidity
                current_price = await moralis_client.get_current_price(token)
                if current_price > 0:
                    survived += 1
            except:
                pass
        
        if sample_size > 0:
            metrics['tokens_survived_24h'] = survived
            metrics['survival_rate'] = survived / sample_size
            
            # Detect pump and dump pattern
            if metrics['survival_rate'] < self.critical_survival_rate:
                metrics['pump_and_dump_pattern'] = True
            elif metrics['survival_rate'] < self.min_survival_rate:
                # Borderline suspicious
                metrics['pump_and_dump_pattern'] = True
        
        return metrics
    
    def _calculate_farmer_score(self, detailed_metrics: Dict) -> float:
        """Calculate composite farmer score from all metrics"""
        score = 0.0
        weights = {
            'holding': 0.20,
            'entry_crowding': 0.20,
            'follower_disadvantage': 0.20,
            'exit_impact': 0.15,
            'funding_entropy': 0.15,
            'uniformity': 0.10
        }
        
        # 1. Holding time component
        holding = detailed_metrics.get('holding', {})
        if holding.get('is_farming_pattern'):
            score += weights['holding']
        elif holding.get('short_hold_rate', 0) > 0.5:
            score += weights['holding'] * 0.5
        
        # 2. Entry crowding component
        entry = detailed_metrics.get('entry_crowding', {})
        if entry.get('crowding_detected'):
            score += weights['entry_crowding']
        elif entry.get('median_entry_advantage', 0) < -0.02:
            score += weights['entry_crowding'] * 0.5
        
        # 3. Follower disadvantage component
        follower = detailed_metrics.get('follower_disadvantage', {})
        if follower.get('follower_trap_detected'):
            score += weights['follower_disadvantage']
        elif follower.get('median_follower_disadvantage', 0) > 0.03:
            score += weights['follower_disadvantage'] * 0.5
        
        # 4. Exit impact component
        exit_impact = detailed_metrics.get('exit_impact', {})
        if exit_impact.get('dump_pattern_detected'):
            score += weights['exit_impact']
        elif exit_impact.get('avg_exit_impact', 0) > 0.15:
            score += weights['exit_impact'] * 0.5
        
        # 5. Funding entropy component
        funding = detailed_metrics.get('funding_entropy', {})
        if funding.get('centralized_funding'):
            score += weights['funding_entropy']
        elif funding.get('shannon_entropy', 2) < 1.5:
            score += weights['funding_entropy'] * 0.5
        
        # 6. Trade uniformity component
        uniformity = detailed_metrics.get('uniformity', {})
        if uniformity.get('spam_pattern_detected'):
            score += weights['uniformity']
        
        # Additional factors
        winrate = detailed_metrics.get('winrate_holding', {})
        if winrate.get('unrealistic_performance'):
            score += 0.1  # Bonus penalty for unrealistic performance
        
        liquidity = detailed_metrics.get('liquidity', {})
        if liquidity.get('pure_extractor'):
            score += 0.05  # Small penalty for pure extraction
        
        survival = detailed_metrics.get('survival', {})
        if survival.get('pump_and_dump_pattern'):
            score += 0.1  # Penalty for pump and dump tokens
        
        # Cap at 1.0
        return min(1.0, score)
    
    def batch_classify_wallets(self, wallet_metrics_list: List[Dict]) -> Dict:
        """Classify multiple wallets into categories"""
        classifications = {
            'farmers': [],
            'suspicious': [],
            'safe': [],
            'stats': {
                'total_analyzed': len(wallet_metrics_list),
                'farmer_count': 0,
                'suspicious_count': 0,
                'safe_count': 0,
                'avg_farmer_score': 0
            }
        }
        
        scores = []
        
        for metrics in wallet_metrics_list:
            wallet = metrics.get('wallet', '')
            score = metrics.get('farmer_score', 0)
            classification = metrics.get('classification', 'unknown')
            
            scores.append(score)
            
            if classification == 'farmer':
                classifications['farmers'].append(wallet)
                classifications['stats']['farmer_count'] += 1
            elif classification == 'suspicious':
                classifications['suspicious'].append(wallet)
                classifications['stats']['suspicious_count'] += 1
            elif classification == 'safe':
                classifications['safe'].append(wallet)
                classifications['stats']['safe_count'] += 1
        
        if scores:
            classifications['stats']['avg_farmer_score'] = statistics.mean(scores)
        
        # Log summary
        self.logger.info(f"Batch classification complete: "
                        f"farmers={classifications['stats']['farmer_count']}, "
                        f"suspicious={classifications['stats']['suspicious_count']}, "
                        f"safe={classifications['stats']['safe_count']}, "
                        f"avg_score={classifications['stats']['avg_farmer_score']:.2f}")
        
        return classifications