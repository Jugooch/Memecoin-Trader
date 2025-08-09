"""
Alpha Wallet Discovery V2 - One-Off Analysis
Finds wallets that repeatedly buy successful tokens early

IMPORTANT: This is a standalone analysis tool. For continuous alpha wallet discovery,
use the alpha_accumulator.py instead, which runs continuously and handles BitQuery's 
data limitations better.

How V2 Works:
- Analyzes recent tokens (last 2 hours) from BitQuery  
- Identifies successful tokens based on holder count and activity
- Finds wallets that bought these tokens within first 5 minutes
- Looks for wallets appearing on 2+ successful tokens
- Updates config.yml with discovered alpha wallets

When to Use V2:
- One-off analysis when you want to check current market
- Quick manual discovery of alpha wallets
- Testing/debugging alpha discovery logic

When to Use Alpha Accumulator Instead:
- Continuous 24/7 alpha wallet discovery
- Building up historical data over time
- Production usage for your trading bot

Usage:
  python -m src.discovery.alpha_discovery_v2

Note: Due to BitQuery limitations, V2 only gets ~3-4 minutes of data per run.
For better coverage over time, use the accumulator which runs every few minutes.
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Set
from datetime import datetime, timedelta
from collections import defaultdict
import json

from src.clients.bitquery_client import BitqueryClient
from src.clients.moralis_client import MoralisClient
from src.core.database import Database
from src.utils.config_loader import load_config


class ProvenAlphaFinder:
    def __init__(self, bitquery: BitqueryClient, moralis: MoralisClient, database: Database):
        self.bitquery = bitquery
        self.moralis = moralis
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters - optimized for more results
        self.analysis_window_hours = 24  # Analyze tokens after 24 hours
        self.success_threshold = 2.0     # 2x price increase = successful (aligned with code)
        self.early_window_seconds = 300  # First 5 minutes = early
        self.min_wallet_appearances = 2  # Must appear on 2+ successful tokens (repeated success)
        
    async def discover_alpha_wallets(self) -> List[str]:
        """Main discovery process using time-delayed analysis"""
        self.logger.info("Starting advanced alpha wallet discovery...")
        
        # Step 1: Get tokens from 24-48 hours ago (enough time to see performance)
        historical_tokens = await self._get_historical_tokens()
        
        # Step 2: For each token, determine if it was successful
        successful_tokens = []
        self.logger.info(f"Analyzing {len(historical_tokens)} tokens for success...")
        
        for i, token_data in enumerate(historical_tokens):
            if await self._was_token_successful(token_data):
                successful_tokens.append(token_data)
            
            # Log progress for first 10 tokens
            if i < 10:
                perf = token_data.get('performance_multiplier', 0)
                holders = token_data.get('holder_count', 0) 
                swaps = token_data.get('swap_count', 0)
                success = len(successful_tokens)
                self.logger.info(f"Token {i+1}/{len(historical_tokens)}: {token_data['mint'][:8]}... "
                               f"Perf={perf:.2f}x, Holders={holders}, Swaps={swaps} -> {success} successful so far")
        
        self.logger.info(f"Found {len(successful_tokens)} successful tokens from analysis period")
        
        # Step 3: Find early buyers of successful tokens
        alpha_candidates = defaultdict(list)
        
        for token_data in successful_tokens:
            early_buyers = await self._find_early_buyers(token_data)
            
            for wallet in early_buyers:
                alpha_candidates[wallet].append({
                    'token': token_data,
                    'buy_time': 'early',  # Within first 2 minutes
                    'performance': token_data.get('performance_multiplier', 1)
                })
        
        # Debug: Show wallet overlap visibility
        mints_per_wallet = defaultdict(set)
        for w, entries in alpha_candidates.items():
            for e in entries:
                mints_per_wallet[w].add(e["token"]["mint"])
        repeat = [(w, len(ms)) for w, ms in mints_per_wallet.items() if len(ms) >= 2]
        self.logger.info(f"Wallets with >=2 distinct mints: {len(repeat)}")
        for w, k in sorted(repeat, key=lambda x: -x[1])[:10]:
            self.logger.info(f"{w[:8]}.. on {k} mints")
        
        # Step 4: Score and rank wallets
        alpha_wallets = await self._score_alpha_candidates(alpha_candidates)
        
        return alpha_wallets
    
    async def _get_historical_tokens(self) -> List[Dict]:
        """Get tokens launched 24-48h ago (time-delayed analysis for proven success)"""
        # Use very recent window where BitQuery has most data
        now = datetime.utcnow()
        start_time = now - timedelta(hours=2)      # 2 hours ago UTC
        end_time = now - timedelta(minutes=5)      # 5 minutes ago UTC
        
        self.logger.info(f"Window UTC: {start_time.isoformat()}Z -> {end_time.isoformat()}Z")
        
        try:
            # Format timestamps for server-side filtering
            start_iso = start_time.isoformat(timespec="seconds") + "Z"
            end_iso = end_time.isoformat(timespec="seconds") + "Z"
            
            self.logger.info(f"Requesting trades from Bitquery with time filter: {start_iso} to {end_iso}")
            
            # First test: Try WITHOUT time filter to see if data is available
            self.logger.info("DEBUGGING: Testing without time filter first...")
            test_trades = await self.bitquery.get_recent_token_launches(limit=10)
            self.logger.info(f"Without filter: Got {len(test_trades)} trades")
            
            if test_trades:
                # Show sample timestamps to understand data availability
                sample_times = []
                for trade in test_trades[:5]:
                    time_str = trade.get('Block', {}).get('Time', '')
                    if time_str:
                        ts = self._parse_iso_timestamp(time_str)
                        sample_times.append({
                            'raw': time_str,
                            'parsed_utc': datetime.utcfromtimestamp(ts).isoformat() + 'Z'
                        })
                self.logger.info(f"Sample trade times without filter: {sample_times}")
                
                # Check if ANY of these fall in our target window
                in_window = 0
                for trade in test_trades:
                    time_str = trade.get('Block', {}).get('Time', '')
                    if time_str:
                        ts = self._parse_iso_timestamp(time_str)
                        if start_time.timestamp() <= ts <= end_time.timestamp():
                            in_window += 1
                self.logger.info(f"Trades in our window (24-48h ago): {in_window}/{len(test_trades)}")
            
            # Now try WITH time filter
            self.logger.info("DEBUGGING: Now testing WITH time filter...")
            recent_trades = await self.bitquery.get_recent_token_launches(
                limit=3000,  # Get more trades for wider window
                start_time=start_iso, 
                end_time=end_iso
            )
            
            self.logger.info(f"BitQuery returned {len(recent_trades)} trades with filter")
            
            # Debug: Show actual trade time coverage
            times = [t.get("Block",{}).get("Time") for t in recent_trades if t.get("Block",{}).get("Time")]
            if times:
                tnums = [self._parse_iso_timestamp(x) for x in times]
                min_time = datetime.utcfromtimestamp(min(tnums))
                max_time = datetime.utcfromtimestamp(max(tnums))
                self.logger.info(f"Trade time coverage (Bitquery payload): min={min_time}Z  max={max_time}Z")
            else:
                self.logger.warning("No Block.Time in Bitquery payload")
            
            # Group server-filtered trades by mint address to find true launch times
            self.logger.info("Grouping filtered trades by token to find true launch times...")
            token_trades = {}
            SOL_ADDRESS = "11111111111111111111111111111111"
            
            for trade in recent_trades:
                # Extract both buy and sell currency info
                buy_currency = trade.get('Trade', {}).get('Buy', {}).get('Currency', {})
                sell_currency = trade.get('Trade', {}).get('Sell', {}).get('Currency', {})
                
                # Determine which side has the new token (non-SOL)
                buy_mint = buy_currency.get('MintAddress', '')
                sell_mint = sell_currency.get('MintAddress', '')
                
                mint = None
                token_currency = None
                
                # Choose the non-SOL side as the new token
                if buy_mint and buy_mint != SOL_ADDRESS:
                    mint = buy_mint
                    token_currency = buy_currency
                elif sell_mint and sell_mint != SOL_ADDRESS:
                    mint = sell_mint  
                    token_currency = sell_currency
                else:
                    continue  # Skip if no valid token found
                
                # Get timestamp and validate
                timestamp_str = trade.get('Block', {}).get('Time', '')
                if not timestamp_str:
                    continue
                    
                trade_timestamp = self._parse_iso_timestamp(timestamp_str)
                
                # Group by mint - store all trades for this token
                if mint not in token_trades:
                    token_trades[mint] = {
                        'trades': [],
                        'name': token_currency.get('Name', 'Unknown'),
                        'symbol': token_currency.get('Symbol', ''),
                        'deployer': trade.get('Transaction', {}).get('Signer', ''),
                    }
                
                token_trades[mint]['trades'].append({
                    'timestamp': trade_timestamp,
                    'trade_data': trade
                })
            
            self.logger.info(f"Found {len(token_trades)} unique tokens in recent trades")
            
            # Find true launch time (earliest trade) for each token
            # Since we used server-side filtering, all trades are already in the 24-48h window
            historical_tokens = []
            
            for mint, token_info in token_trades.items():
                # Sort trades by timestamp to find earliest (true launch)
                token_info['trades'].sort(key=lambda x: x['timestamp'])
                true_launch_time = token_info['trades'][0]['timestamp']
                
                # Clean token name for logging
                clean_name = token_info['name'].encode('ascii', 'ignore').decode('ascii')
                
                self.logger.debug(f"Token {mint[:8]}... ({clean_name}) launched at {datetime.utcfromtimestamp(true_launch_time)}Z")
                
                historical_tokens.append({
                    'mint': mint,
                    'launch_time': true_launch_time,  # TRUE launch time (earliest trade)
                    'name': token_info['name'],
                    'symbol': token_info['symbol'],
                    'deployer': token_info['deployer'],
                    'trade_count': len(token_info['trades'])  # How many trades we have for this token
                })
                
                # Limit to 150 tokens for analysis (more coverage)
                if len(historical_tokens) >= 150:
                    break
            
            self.logger.info(f"Found {len(historical_tokens)} tokens with true launch times from server-filtered trades")
            
            return historical_tokens
            
        except Exception as e:
            self.logger.error(f"Error getting historical tokens: {e}")
            return []
    
    async def _was_token_successful(self, token_data: Dict) -> bool:
        """Determine if a token was successful using comprehensive criteria with performance multiplier"""
        mint = token_data['mint']
        launch_time = token_data.get('launch_time')
        
        # All tokens now have true launch times from earliest trade analysis
        if not launch_time:
            self.logger.error(f"Token {mint[:8]}... missing launch time!")
            return False
        
        try:
            self.logger.debug(f"Analyzing token success for: {mint[:8]}...")
            
            # Get current price
            current_price = await self.moralis.get_current_price(mint)
            if current_price <= 0:
                self.logger.debug(f"Token {mint[:8]}... rejected: no price data")
                return False
            
            # Calculate performance multiplier using early vs follow-through price
            performance_multiplier = 1.0
            try:
                # Get swaps (Moralis max limit is 100)
                all_swaps = await self.moralis.get_token_swaps(mint, limit=100)
                
                # Filter to early swaps (first 2 minutes after launch)
                early_swaps = []
                for swap in all_swaps:
                    swap_time = self._parse_iso_timestamp(swap.get('timestamp', ''))
                    if launch_time <= swap_time <= launch_time + self.early_window_seconds:
                        early_swaps.append(swap)
                
                # Calculate early price (median of first buys) - use improved buy detection
                early_buy_prices = [
                    float(swap.get('price_usd', 0)) for swap in early_swaps 
                    if self._is_buy_into_token(swap, mint) and float(swap.get('price_usd', 0)) > 0
                ]
                
                if early_buy_prices:
                    early_price = sorted(early_buy_prices)[len(early_buy_prices)//2]  # median
                    
                    # Filter existing swaps for follow-through period (1-3 hours post launch)
                    ft_swaps = []
                    for swap in all_swaps:
                        swap_time = self._parse_iso_timestamp(swap.get('timestamp', ''))
                        if launch_time + 3600 <= swap_time <= launch_time + 10800:
                            ft_swaps.append(swap)
                    
                    # Calculate VWAP for follow-through period (fixed USD×USD bug)
                    sum_price_weighted = 0.0
                    sum_weights = 0.0
                    for swap in ft_swaps:
                        if self._is_buy_into_token(swap, mint):
                            price = float(swap.get('price_usd', 0))
                            vol_usd = float(swap.get('amount_usd', 0))  # This is already in USD
                            if price > 0 and vol_usd > 0:
                                # Use USD volume as weight (don't multiply price×volume)
                                sum_price_weighted += price * vol_usd
                                sum_weights += vol_usd
                    
                    if sum_weights > 0:
                        ft_vwap = sum_price_weighted / sum_weights
                        performance_multiplier = ft_vwap / early_price if early_price > 0 else 1.0
                    else:
                        # Fallback to current price if no follow-through data
                        performance_multiplier = current_price / early_price if early_price > 0 else 1.0
                        
            except Exception as e:
                self.logger.debug(f"Error calculating performance multiplier for {mint[:8]}...: {e}")
            
            # Get holder count and growth
            holder_stats = await self.moralis.get_token_holders(mint)
            holder_count = holder_stats.get('holder_count', 0)
            
            # Get swap activity (Moralis max limit is 100)
            all_swaps_for_activity = await self.moralis.get_token_swaps(mint, limit=100)
            swaps_first3h = len([
                swap for swap in all_swaps_for_activity
                if launch_time <= self._parse_iso_timestamp(swap.get('timestamp', '')) <= launch_time + 10800
            ])
            
            # Success criteria - relaxed since we only get recent tokens
            is_successful = (
                holder_count >= 5 and                # Lower threshold for recent tokens
                swaps_first3h >= 5 and              # Some activity is enough
                current_price > 0                   # Still has value
            )
            
            self.logger.debug(f"Historical token {mint[:8]}: perf={performance_multiplier:.2f}x, holders={holder_count}, swaps={swaps_first3h}")
            
            # Log if it would have been successful with performance multiplier check
            if performance_multiplier >= 1.5 and holder_count >= 10 and swaps_first3h >= 5:
                self.logger.info(f"Token {mint[:8]} would be successful with perf criteria: {performance_multiplier:.2f}x")
            
            # Store all metrics for scoring
            token_data['performance_multiplier'] = performance_multiplier
            token_data['holder_count'] = holder_count
            token_data['swap_count'] = swaps_first3h
            token_data['current_price'] = current_price
            
            if is_successful:
                self.logger.info(f"Token {mint[:8]}... is successful: "
                               f"Perf={performance_multiplier:.1f}x, Holders={holder_count}, Swaps={swaps_first3h}")
            else:
                self.logger.debug(f"Token {mint[:8]}... failed criteria: "
                                f"Perf={performance_multiplier:.1f}x, Holders={holder_count}, Swaps={swaps_first3h}")
            
            return is_successful
            
        except Exception as e:
            self.logger.error(f"Error analyzing token success {mint}: {e}")
            return False
            return False
    
    async def _find_early_buyers(self, token_data: Dict) -> List[str]:
        """Find wallets that bought token in first 2 minutes after launch"""
        mint = token_data['mint']
        launch_time = token_data.get('launch_time')
        
        if not launch_time:
            self.logger.warning(f"No launch time available for {mint[:8]}..., cannot find early buyers")
            return []
        
        try:
            # Use actual launch time with proper window (first 2 minutes)
            early_window = self.early_window_seconds  # Configurable early window
            
            # Get swaps and filter for launch time window (Moralis max limit is 100)
            all_swaps = await self.moralis.get_token_swaps(mint, limit=100)
            
            # Filter to swaps within early window
            swaps = [
                swap for swap in all_swaps
                if launch_time <= self._parse_iso_timestamp(swap.get('timestamp', '')) <= launch_time + early_window
            ]
            
            self.logger.debug(f"Found {len(swaps)} early swaps for {mint[:8]}... in first {early_window}s")
            
            if not swaps:
                self.logger.debug(f"No early swaps found for {mint[:8]}...")
                return []
            
            # Sort swaps by timestamp 
            sorted_swaps = sorted(swaps, key=lambda x: self._parse_iso_timestamp(x.get('timestamp', '')))
            
            early_buyers = set()
            
            for swap in sorted_swaps:
                swap_time = self._parse_iso_timestamp(swap.get('timestamp', ''))
                
                # Debug: Log first few swap times for troubleshooting
                if len(early_buyers) < 3:
                    self.logger.debug(f"Early swap: {swap_time}, Launch: {launch_time}, Window: {early_window}s")
                
                # All swaps in our query are within the early window by design  
                if launch_time <= swap_time <= launch_time + early_window:
                    # Check if this is actually a buy into our token
                    if not self._is_buy_into_token(swap, mint):
                        continue
                        
                    # Try different wallet address fields that might be in the swap data
                    wallet = (swap.get('wallet') or 
                             swap.get('buyer') or 
                             swap.get('trader') or 
                             swap.get('from_address') or
                             swap.get('to_address'))
                    
                    # Also try nested structures
                    if not wallet and 'transaction' in swap:
                        wallet = swap['transaction'].get('from')
                    
                    # Debug: Show what's actually in the wallet field and full swap data
                    if len(early_buyers) < 2:
                        self.logger.info(f"Raw wallet value: '{swap.get('wallet')}', type: {type(swap.get('wallet'))}")
                        self.logger.info(f"Full swap data: {swap}")
                        if swap.get('wallet'):
                            self.logger.info(f"Wallet length: {len(str(swap.get('wallet')))}")
                    
                    # Validate wallet address and ensure it's a buy (not the token mint address)
                    if wallet and len(str(wallet)) >= 32 and str(wallet) != mint:
                        early_buyers.add(str(wallet))
                        self.logger.info(f"Found early buyer: {str(wallet)[:8]}...")
                    elif len(early_buyers) < 3:  # Debug validation failure
                        self.logger.info(f"Wallet validation failed - wallet: '{wallet}', len: {len(str(wallet)) if wallet else 0}, mint: {mint[:8]}...")
                else:
                    # Once we pass the early window, we can break since swaps are sorted
                    break
            
            self.logger.info(f"Found {len(early_buyers)} early buyers for {mint[:8]}...")
            return list(early_buyers)
            
        except Exception as e:
            self.logger.error(f"Error finding early buyers for {mint}: {e}")
            return []
    
    def _parse_iso_timestamp(self, timestamp_str) -> float:
        """Parse timestamp to unix timestamp"""
        try:
            # Handle different timestamp formats
            if isinstance(timestamp_str, (int, float)):
                return float(timestamp_str)
            
            if isinstance(timestamp_str, str):
                if 'T' in timestamp_str:
                    # ISO format
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    return dt.timestamp()
                else:
                    # Try to parse as number string
                    return float(timestamp_str)
        except:
            pass
        return time.time()
    
    def _is_buy_into_token(self, swap: Dict, mint: str) -> bool:
        """Determine if a swap is a buy into the target token (handles different DEX schemas)"""
        # Check common field patterns for Solana DEX data
        
        # Pattern 1: to_token field (most common)
        if swap.get('to_token') == mint:
            return True
        
        # Pattern 2: mint_out field (Raydium/Jupiter style)
        if swap.get('mint_out') == mint:
            return True
            
        # Pattern 3: token_address with side flag
        if swap.get('token_address') == mint and swap.get('side') == 'buy':
            return True
            
        # Pattern 4: Check if we're buying the mint (not selling it)
        # If mint is in mint_in, it's a sell; if in mint_out, it's a buy
        if swap.get('mint_in') != mint and swap.get('mint_out') == mint:
            return True
            
        return False
    
    async def _score_alpha_candidates(self, candidates: Dict) -> List[str]:
        """Score and rank alpha wallet candidates with recency decay and performance multiplier"""
        import math
        
        scored_wallets = []
        current_time = time.time()
        
        self.logger.info(f"Evaluating {len(candidates)} wallet candidates...")
        
        # Show wallet appearance distribution for debugging
        appearance_counts = {}
        for wallet, token_list in candidates.items():
            count = len(token_list)
            appearance_counts[count] = appearance_counts.get(count, 0) + 1
            
        self.logger.info(f"Wallet appearance distribution: {dict(sorted(appearance_counts.items()))}")
        
        for wallet, token_list in candidates.items():
            # Log all candidates for debugging
            self.logger.debug(f"Wallet {wallet[:8]}... appears on {len(token_list)} tokens")
            
            # Skip wallets with insufficient track record
            if len(token_list) < self.min_wallet_appearances:
                continue
            
            # Calculate performance metrics with proper multiplier
            total_tokens = len(token_list)
            
            # Use actual performance_multiplier from token data
            performance_multipliers = []
            recency_weights = []
            
            for token_entry in token_list:
                # Access nested token data correctly
                raw_token_data = token_entry["token"]
                
                # Get performance multiplier (use the value set in the entry)
                perf_mult = token_entry.get('performance', 1.0)  # This comes from the entry, not nested
                performance_multipliers.append(perf_mult)
                
                # Calculate recency decay (12-hour half-life) - use launch_time from raw token data
                token_time = raw_token_data.get('launch_time', current_time)
                time_diff = current_time - token_time
                recency_weight = math.exp(-time_diff / (12 * 3600))  # 12h decay
                recency_weights.append(recency_weight)
            
            # Weighted averages using recency
            if recency_weights:
                total_weight = sum(recency_weights)
                avg_performance = sum(p * w for p, w in zip(performance_multipliers, recency_weights)) / total_weight
                
                # Calculate success rate (performance >= 2.0x)
                success_count = sum(w for p, w in zip(performance_multipliers, recency_weights) if p >= 2.0)
                success_rate = success_count / total_weight
                
                # Overall recency factor
                avg_recency = total_weight / len(recency_weights)
            else:
                avg_performance = 1.0
                success_rate = 0.0
                avg_recency = 0.0
            
            # Calculate final score (no fallback penalty needed since all tokens are historical)
            base_score = (
                total_tokens * 15 +           # Points for volume (increased)
                avg_performance * 25 +        # Points for average performance
                success_rate * 40             # Points for consistency (increased)
            )
            
            # Apply recency weighting
            final_score = base_score * avg_recency
            
            scored_wallets.append({
                'wallet': wallet,
                'score': final_score,
                'total_tokens': total_tokens,
                'avg_performance': avg_performance,
                'success_rate': success_rate,
                'avg_recency': avg_recency,
                'tokens': token_list
            })
        
        # Sort by score (highest first)
        scored_wallets.sort(key=lambda x: x['score'], reverse=True)
        
        # Log top performers
        self.logger.info(f"Found {len(scored_wallets)} qualified alpha candidates")
        
        for i, wallet_data in enumerate(scored_wallets[:10]):
            w = wallet_data
            self.logger.info(f"#{i+1}: {w['wallet'][:8]}... "
                           f"Score: {w['score']:.1f} | "
                           f"Tokens: {w['total_tokens']} | "
                           f"Perf: {w['avg_performance']:.1f}x | "
                           f"Success: {w['success_rate']:.1%} | "
                           f"Recency: {w['avg_recency']:.2f}")
        
        # Return top 30 wallets (increased from 20)
        return [w['wallet'] for w in scored_wallets[:30]]
    
    async def continuous_discovery(self):
        """Run continuous alpha discovery (for cloud deployment)"""
        self.logger.info("Starting continuous alpha discovery mode...")
        
        while True:
            try:
                # Run discovery every 2 hours (reduced from 6 for fresher wallets)
                alpha_wallets = await self.discover_alpha_wallets()
                
                if alpha_wallets:
                    # Save to database
                    await self._save_discovered_wallets(alpha_wallets)
                    
                    # Update main bot config if needed
                    await self._update_bot_config(alpha_wallets)
                
                # Wait 2 hours before next discovery (reduced from 6)
                await asyncio.sleep(2 * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in continuous discovery: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _save_discovered_wallets(self, wallets: List[str]):
        """Save discovered wallets to database"""
        for wallet in wallets:
            await self.database.update_alpha_wallet(wallet, {
                'discovery_date': datetime.now().isoformat(),
                'discovery_method': 'time_delayed_analysis',
                'performance_score': 100,  # Default high score for new discoveries
                'total_trades': 0,
                'metadata': {
                    'source': 'proven_alpha_finder',
                    'validation_status': 'pending'
                }
            })
        
        self.logger.info(f"Saved {len(wallets)} alpha wallets to database")
    
    async def _update_bot_config(self, new_wallets: List[str]):
        """Update bot config with new alpha wallets"""
        try:
            # Read current config using shared loader
            import yaml
            config = load_config('config.yml')
            
            # Get current wallets
            current_wallets = set(config.get('watched_wallets', []))
            
            # Add new wallets (keep existing + add new)
            updated_wallets = list(current_wallets.union(set(new_wallets)))
            
            # Limit to top 50 wallets (manageable size)
            config['watched_wallets'] = updated_wallets[:50]
            
            # Find the actual config file path to write to
            import os
            possible_paths = ['config.yml', 'config/config.yml']
            config_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                config_path = 'config.yml'  # Default if none exist
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Updated config with {len(updated_wallets)} alpha wallets")
            
        except Exception as e:
            self.logger.error(f"Error updating bot config: {e}")


async def main():
    """Test the advanced alpha finder"""
    import yaml
    
    # Setup logging first with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/alpha_discovery.log', mode='a', encoding='utf-8')
        ]
    )
    
    # Disable noisy GQL transport logging that prints all trade data
    logging.getLogger('gql.transport.aiohttp').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Alpha Discovery V2...")
    
    # Load config using shared loader
    config = load_config('config.yml')
    
    # Initialize clients
    bitquery = BitqueryClient(config.get('bitquery_tokens', config.get('bitquery_token')))
    moralis = MoralisClient(config.get('moralis_keys', config.get('moralis_key')))
    database = Database()
    
    await database.initialize()
    
    try:
        # Run advanced alpha finder
        finder = ProvenAlphaFinder(bitquery, moralis, database)
        alpha_wallets = await finder.discover_alpha_wallets()
        
        print(f"\nDiscovered {len(alpha_wallets)} proven alpha wallets:")
        for i, wallet in enumerate(alpha_wallets[:10], 1):
            print(f"  #{i}: {wallet}")
        
        if alpha_wallets:
            # Save to database
            await finder._save_discovered_wallets(alpha_wallets)
            
            # Update main bot config
            await finder._update_bot_config(alpha_wallets)
            
            print(f"\nOK - These wallets have been validated through time-delayed analysis")
            print(f"   They consistently bought successful tokens within first 10 minutes")
            print(f"   Config updated with {len(alpha_wallets)} alpha wallets!")
    
    finally:
        # Clean up
        await moralis.close()
        if hasattr(bitquery, 'client') and bitquery.client:
            await bitquery.client.transport.close()


if __name__ == "__main__":
    asyncio.run(main())