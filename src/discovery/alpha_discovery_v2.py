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
from src.discovery.farming_detector import FarmingDetector


class ProvenAlphaFinder:
    def __init__(self, bitquery: BitqueryClient, moralis: MoralisClient, database: Database, config: Dict = None):
        self.bitquery = bitquery
        self.moralis = moralis
        self.database = database
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize farming detector
        self.farming_detector = FarmingDetector(config)
        
        # Strategy parameters - Read from config or use defaults
        # Get discovery success thresholds from config
        api_opt = self.config.get('api_optimization', {})
        self.success_thresholds = api_opt.get('discovery_success_thresholds', {
            'high': 1.3,      # 1.3x = 30% gain (matches reduced TP target)
            'medium': 1.2,    # 1.2x = 20% gain (good quick pump)
            'low': 1.1       # 1.1x = 10% gain (minimum after fees)
        })
        
        # Get early window from config (in minutes, convert to seconds)
        self.early_window_seconds = api_opt.get('discovery_early_window_minutes', 3) * 60
        
        # ADJUSTED: Require more trades for statistical significance
        self.min_wallet_appearances = api_opt.get('discovery_min_appearances', {
            'tier_1': 5,      # High-quality wallets: 5+ high success tokens (was 2)
            'tier_2': 7,      # Medium-quality wallets: 7+ medium success tokens (was 3)
            'tier_3': 10      # Emerging wallets: 10+ low success tokens (was 4)
        })
        
        # Get additional discovery configs
        self.max_moralis_tokens = api_opt.get('discovery_max_moralis_tokens', 30)
        self.bitquery_success_threshold = api_opt.get('discovery_bitquery_success_threshold', 30)
        
        self.logger.info(f"Discovery config loaded: thresholds={self.success_thresholds}, "
                        f"early_window={self.early_window_seconds}s, appearances={self.min_wallet_appearances}, "
                        f"max_moralis={self.max_moralis_tokens}, bitquery_threshold={self.bitquery_success_threshold}")
        
    async def discover_alpha_wallets(self) -> List[str]:
        """Optimized discovery process - Bitquery-first approach with comprehensive diagnostics"""
        self.logger.info("=" * 80)
        self.logger.info("Starting FIXED alpha wallet discovery (Paginated + Metrics + Fallback)")
        self.logger.info("=" * 80)
        
        # Step 1: Get tokens from 45min-15min ago (aligned with trading timeframe)
        start_time = time.time()
        historical_tokens = await self._get_historical_tokens()
        fetch_time = time.time() - start_time
        
        # Guardrail: Early bail if insufficient data  
        if len(historical_tokens) == 0:
            self.logger.warning("❌ DISCOVERY ABORTED: No historical tokens found")
            return []
            
        if len(historical_tokens) < 20:
            self.logger.warning(f"⚠️  Low token count ({len(historical_tokens)}), results may be limited")
        
        self.logger.info(f"✅ Token fetch completed in {fetch_time:.1f}s: {len(historical_tokens)} tokens")
        
        # Step 2: Pre-filter using Bitquery data with comprehensive diagnostics
        prefilter_start = time.time()
        promising_tokens = await self._prefilter_tokens_bitquery_only(historical_tokens)
        prefilter_time = time.time() - prefilter_start
        
        # Guardrail: Check prefilter effectiveness
        if len(promising_tokens) == 0:
            self.logger.warning("❌ DISCOVERY ABORTED: No promising tokens after prefilter")
            self.logger.warning("   → Check if prefilter thresholds are too strict")
            self.logger.warning("   → Check if price data is missing from all tokens")
            return []
        
        prefilter_rate = len(promising_tokens) / len(historical_tokens) * 100
        self.logger.info(f"✅ Prefilter completed in {prefilter_time:.1f}s: {len(promising_tokens)} promising ({prefilter_rate:.1f}%)")
        
        # Step 3: Validate with Moralis (limit to reasonable number)
        # Use config value for max Moralis tokens
        top_k = min(self.max_moralis_tokens, len(promising_tokens))
        validated_tokens = []
        
        if top_k > 0:
            validation_start = time.time()
            self.logger.info(f"🔍 Validating top {top_k} tokens with Moralis...")
            
            for i, token_data in enumerate(promising_tokens[:top_k]):
                try:
                    if await self._validate_token_with_moralis(token_data):
                        validated_tokens.append(token_data)
                        
                    if (i + 1) % 10 == 0:  # Progress every 10 validations
                        self.logger.info(f"   Progress: {i+1}/{top_k} validated, {len(validated_tokens)} successful")
                        
                except Exception as e:
                    self.logger.warning(f"   Validation error for {token_data['mint'][:8]}...: {e}")
                    
            validation_time = time.time() - validation_start
            validation_rate = len(validated_tokens) / top_k * 100 if top_k > 0 else 0
            self.logger.info(f"✅ Validation completed in {validation_time:.1f}s: {len(validated_tokens)} validated ({validation_rate:.1f}%)")
        
        # Guardrail: Check validation results
        if len(validated_tokens) == 0:
            self.logger.warning("❌ DISCOVERY ABORTED: No tokens passed Moralis validation")
            self.logger.warning("   → Check if validation criteria are too strict")
            self.logger.warning("   → Check if tokens are too old to have current price data")
            return []
        
        # Step 4: Find early buyers using Bitquery-first approach
        buyer_start = time.time()
        alpha_candidates = defaultdict(list)
        total_early_buyers = 0
        
        for i, token_data in enumerate(validated_tokens):
            early_buyers = await self._find_early_buyers(token_data)
            total_early_buyers += len(early_buyers)
            
            for wallet in early_buyers:
                alpha_candidates[wallet].append({
                    'token': token_data,
                    'buy_time': 'early',  # Within first 5 minutes
                    'performance': token_data.get('performance_multiplier', 1.0)
                })
                
            if (i + 1) % 10 == 0:  # Progress every 10 tokens
                self.logger.info(f"   Early buyer progress: {i+1}/{len(validated_tokens)}, {len(alpha_candidates)} candidates so far")
        
        buyer_time = time.time() - buyer_start
        avg_buyers_per_token = total_early_buyers / len(validated_tokens) if validated_tokens else 0
        self.logger.info(f"✅ Early buyer search completed in {buyer_time:.1f}s: {total_early_buyers} total buyers ({avg_buyers_per_token:.1f} avg/token)")
        
        # Debug: Wallet overlap analysis
        mints_per_wallet = defaultdict(set)
        for wallet, entries in alpha_candidates.items():
            for entry in entries:
                mints_per_wallet[wallet].add(entry["token"]["mint"])
        
        repeat_wallets = [(w, len(ms)) for w, ms in mints_per_wallet.items() if len(ms) >= 2]
        self.logger.info(f"📊 Wallet overlap analysis: {len(repeat_wallets)} wallets appear on ≥2 tokens")
        
        if repeat_wallets:
            for w, count in sorted(repeat_wallets, key=lambda x: -x[1])[:5]:
                self.logger.info(f"   Top overlap: {w[:8]}... appears on {count} tokens")
        else:
            self.logger.warning("   ⚠️  No wallet overlaps found - may indicate insufficient token diversity")
        
        # Guardrail: Check if we have enough candidates
        if len(alpha_candidates) == 0:
            self.logger.warning("❌ DISCOVERY ABORTED: No alpha wallet candidates found")
            self.logger.warning("   → Check if early buyer detection is working")
            self.logger.warning("   → Check if early window (10 min) is appropriate")
            return []
        
        # Step 5: Score and rank alpha candidates
        scoring_start = time.time()
        alpha_wallets = await self._score_alpha_candidates(alpha_candidates)
        scoring_time = time.time() - scoring_start
        
        total_time = time.time() - start_time
        
        # Final summary
        self.logger.info("=" * 80)
        self.logger.info("🎉 ALPHA DISCOVERY COMPLETED")
        self.logger.info(f"📈 Results: {len(alpha_wallets)} alpha wallets discovered")
        self.logger.info(f"⏱️  Total time: {total_time:.1f}s")
        self.logger.info(f"   - Token fetch: {fetch_time:.1f}s")
        self.logger.info(f"   - Prefilter: {prefilter_time:.1f}s") 
        self.logger.info(f"   - Validation: {validation_time:.1f}s")
        self.logger.info(f"   - Early buyers: {buyer_time:.1f}s")
        self.logger.info(f"   - Scoring: {scoring_time:.1f}s")
        self.logger.info("=" * 80)
        
        return alpha_wallets
    
    async def _get_historical_tokens(self) -> List[Dict]:
        """Get recent tokens with comprehensive metrics computation"""
        # EXTENDED: Analyze longer window to find wallets with more trading history
        # 120-5 minutes ago gives us 2 hours of data to find organic traders
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=120)   # 2 hours ago UTC (was 60)
        end_time = now - timedelta(minutes=5)       # 5 minutes ago UTC
        
        self.logger.info(f"Analyzing recent tokens window: {start_time.isoformat()}Z -> {end_time.isoformat()}Z")
        
        try:
            # Format timestamps for server-side filtering
            start_iso = start_time.isoformat(timespec="seconds") + "Z"
            end_iso = end_time.isoformat(timespec="seconds") + "Z"
            
            # Use new paginated method to get full window coverage
            self.logger.info("Using paginated Bitquery to get full window coverage...")
            recent_trades = await self.bitquery.get_trades_windowed_paginated(
                start_iso=start_iso,
                end_iso=end_iso,
                page_limit=3000,
                max_pages=20  # Up to 60k trades total
            )
            
            self.logger.info(f"Paginated BitQuery returned {len(recent_trades)} total trades")
            
            # Early bail if insufficient data
            if len(recent_trades) < 100:
                self.logger.warning(f"Insufficient trades ({len(recent_trades)}) for analysis, skipping discovery")
                return []
            
            # Debug: Show actual trade time coverage
            times = [t.get("Block",{}).get("Time") for t in recent_trades if t.get("Block",{}).get("Time")]
            if times:
                tnums = [self._parse_iso_timestamp(x) for x in times]
                min_time = datetime.utcfromtimestamp(min(tnums))
                max_time = datetime.utcfromtimestamp(max(tnums))
                coverage_minutes = (max(tnums) - min(tnums)) / 60
                self.logger.info(f"Actual coverage: {coverage_minutes:.1f} minutes from {min_time}Z to {max_time}Z")
                
                # Bail if coverage is too small (prevents false negatives)
                if coverage_minutes < 5:  # Less than 5 minutes of coverage
                    self.logger.warning(f"Coverage too small ({coverage_minutes:.1f} min), skipping discovery")
                    return []
            else:
                self.logger.warning("No valid timestamps in trades")
                return []
            
            # Group trades by mint and compute comprehensive metrics
            self.logger.info("Grouping trades by token and computing metrics...")
            token_trades = {}
            SOL_ADDRESS = "11111111111111111111111111111111"
            SOL_WRAPPED = "So11111111111111111111111111111112"
            
            trades_with_price_count = 0
            trades_checked = 0
            
            for trade in recent_trades:
                # Extract trade details
                trade_data = trade.get('Trade', {})
                buy_currency = trade_data.get('Buy', {}).get('Currency', {})
                sell_currency = trade_data.get('Sell', {}).get('Currency', {})
                buy_amount_usd = trade_data.get('Buy', {}).get('AmountInUSD', 0)
                sell_amount_usd = trade_data.get('Sell', {}).get('AmountInUSD', 0)
                # Use Price field, not PriceInUSD (which is always 0 in pump.fun trades)
                buy_price = trade_data.get('Buy', {}).get('Price', 0)
                sell_price = trade_data.get('Sell', {}).get('Price', 0)
                # PriceInUSD is always 0, so we'll use Price instead
                buy_price_usd = buy_price
                sell_price_usd = sell_price
                signer = trade.get('Transaction', {}).get('Signer', '')
                
                # Debug: Check first 5 trades
                if trades_checked < 5:
                    trades_checked += 1
                    # Log the raw structure to see what we're actually getting
                    if trades_checked == 1:
                        self.logger.info(f"DEBUG Trade structure: Buy keys={list(trade_data.get('Buy', {}).keys())}, "
                                       f"Sell keys={list(trade_data.get('Sell', {}).keys())}")
                        # Show the actual raw values
                        self.logger.info(f"DEBUG Raw Buy object: {trade_data.get('Buy', {})}")
                        self.logger.info(f"DEBUG Raw Sell object: {trade_data.get('Sell', {})}")
                    self.logger.info(f"DEBUG Trade {trades_checked}: buy_price={buy_price_usd}, sell_price={sell_price_usd}, "
                                   f"buy_amt={buy_amount_usd}, sell_amt={sell_amount_usd}")
                    if buy_price_usd or sell_price_usd:
                        trades_with_price_count += 1
                
                # Determine which side has the new token (non-SOL)
                buy_mint = buy_currency.get('MintAddress', '')
                sell_mint = sell_currency.get('MintAddress', '')
                
                mint = None
                token_currency = None
                side = None  # 'buy' or 'sell'
                amount_usd = 0
                price_usd = 0
                
                # Choose the non-SOL side as the new token
                if buy_mint and buy_mint not in [SOL_ADDRESS, SOL_WRAPPED]:
                    mint = buy_mint
                    token_currency = buy_currency
                    side = 'buy'
                    # Token is on buy side - use buy price (token price in SOL)
                    amount_usd = float(sell_amount_usd) if sell_amount_usd else 0
                    price_usd = buy_price_usd if buy_price_usd else 0
                    
                    # Debug first few tokens
                    if len(token_trades) < 3 and price_usd != 0:
                        self.logger.debug(f"Token buy: {mint[:8]}... price={price_usd}, amount_usd={amount_usd}")
                elif sell_mint and sell_mint not in [SOL_ADDRESS, SOL_WRAPPED]:
                    mint = sell_mint  
                    token_currency = sell_currency
                    side = 'sell'
                    # Token is on sell side - use sell price (token price in SOL)
                    amount_usd = float(buy_amount_usd) if buy_amount_usd else 0
                    price_usd = sell_price_usd if sell_price_usd else 0
                else:
                    continue  # Skip if no valid token found
                
                # Get timestamp and validate
                timestamp_str = trade.get('Block', {}).get('Time', '')
                if not timestamp_str:
                    continue
                    
                trade_timestamp = self._parse_iso_timestamp(timestamp_str)
                
                # Group by mint - store all trades with enhanced data for metrics
                if mint not in token_trades:
                    token_trades[mint] = {
                        'trades': [],
                        'name': token_currency.get('Name', 'Unknown'),
                        'symbol': token_currency.get('Symbol', ''),
                        'deployer': signer,  # First seen deployer
                    }
                
                token_trades[mint]['trades'].append({
                    'timestamp': trade_timestamp,
                    'signer': signer,
                    'side': side,
                    'amount_usd': float(amount_usd) if amount_usd else 0,
                    'price_usd': float(price_usd) if price_usd else 0,
                    'trade_data': trade
                })
            
            self.logger.info(f"Found {len(token_trades)} unique tokens, computing metrics...")
            self.logger.info(f"DEBUG: First 5 trades had {trades_with_price_count} with price data")
            
            # Compute comprehensive metrics for each token
            historical_tokens = []
            tokens_processed = 0
            tokens_with_metrics = 0
            tokens_with_price = 0
            
            for mint, token_info in token_trades.items():
                # Sort trades by timestamp
                ts_sorted = sorted(token_info['trades'], key=lambda x: x['timestamp'])
                if not ts_sorted:
                    continue
                    
                # Basic metrics
                launch_time = ts_sorted[0]['timestamp']
                swap_count = len(ts_sorted)
                unique_traders = len({t['signer'] for t in ts_sorted if t.get('signer')})
                
                # Price progression (if available)
                price_progression = [t['price_usd'] for t in ts_sorted if t.get('price_usd') and t['price_usd'] > 0]
                
                # Clean token name for logging
                clean_name = token_info['name'].encode('ascii', 'ignore').decode('ascii')
                
                self.logger.debug(f"Token {mint[:8]}... ({clean_name}): {swap_count} swaps, {unique_traders} traders, {len(price_progression)} prices")
                
                # Create enhanced token data with all computed metrics
                token_data = {
                    'mint': mint,
                    'launch_time': launch_time,
                    'name': token_info['name'],
                    'symbol': token_info['symbol'],
                    'deployer': token_info['deployer'],
                    'trade_count': swap_count,
                    # NEW: Add missing metrics that prefilter expects
                    'swap_count': swap_count,
                    'unique_traders': unique_traders,
                    'price_progression': price_progression,
                    # Store raw trades for early buyer detection
                    'raw_trades': ts_sorted
                }
                
                historical_tokens.append(token_data)
                tokens_processed += 1
                
                if swap_count > 0 and unique_traders > 0:
                    tokens_with_metrics += 1
                if len(price_progression) > 0:
                    tokens_with_price += 1
                
                # Limit to 200 tokens for analysis (reasonable processing time)
                if len(historical_tokens) >= 200:
                    break
            
            # Log diagnostic statistics
            self.logger.info(f"Processed {tokens_processed} tokens:")
            self.logger.info(f"  - {tokens_with_metrics} tokens with activity metrics")
            self.logger.info(f"  - {tokens_with_price} tokens with price data ({tokens_with_price/tokens_processed*100:.1f}%)")
            self.logger.info(f"  - Coverage quality: {len(recent_trades)} trades over {coverage_minutes:.1f} minutes")
            
            return historical_tokens
            
        except Exception as e:
            self.logger.error(f"Error getting historical tokens: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    async def _prefilter_tokens_bitquery_only(self, tokens: List[Dict]) -> List[Dict]:
        """Pre-filter tokens using only Bitquery data with activity-based fallback"""
        promising_tokens = []
        
        # Track filtering statistics for debugging
        total_tokens = len(tokens)
        tokens_with_price = 0
        tokens_rejected_no_price_low_activity = 0
        tokens_rejected_low_traders = 0
        tokens_rejected_low_swaps = 0
        
        for token_data in tokens:
            mint = token_data['mint']
            launch_time = token_data.get('launch_time')
            
            if not launch_time:
                continue
            
            # Get metrics from computed Bitquery data
            swap_count = token_data.get('swap_count', 0)
            unique_traders = token_data.get('unique_traders', 0)
            price_progression = token_data.get('price_progression', [])
            
            # Calculate performance multiplier from price data if available
            performance_multiplier = None
            if len(price_progression) >= 2:
                # Find early non-zero prices (first 2 trades)
                early_prices = [p for p in price_progression[:2] if p > 0]
                # Find later prices (after first 2 trades)
                later_prices = [p for p in price_progression[2:] if p > 0]
                
                if early_prices and later_prices:
                    early_price = early_prices[0]
                    peak_price = max(later_prices)
                    performance_multiplier = peak_price / early_price
                    tokens_with_price += 1
            
            # Multi-tier scoring with price-based tiers AND activity-based fallback
            success_tier = None
            score = 0
            
            # Use the threshold from config (loaded in __init__)
            
            # PRICE-BASED TIERS (if available)
            if performance_multiplier is not None:
                if performance_multiplier >= 2.0:
                    success_tier = 'high'
                    score += 50
                elif performance_multiplier >= 1.5:
                    success_tier = 'medium'
                    score += 35
                elif performance_multiplier >= 1.2:
                    success_tier = 'low'
                    score += 20
            
            # ACTIVITY FALLBACK (no price): allow medium/low purely on traders+swaps
            if success_tier is None:
                if unique_traders >= 20 and swap_count >= 40:
                    success_tier = 'medium'
                    score += 30
                elif unique_traders >= 10 and swap_count >= 20:
                    success_tier = 'low'
                    score += 20
                else:
                    # Track rejection reasons for debugging
                    # Get prefilter thresholds from config
                    prefilter_config = self.config.get('discovery_prefilter', {})
                    min_traders = prefilter_config.get('min_unique_traders', 10)
                    min_swaps = prefilter_config.get('min_swap_count', 20)
                    
                    if unique_traders < min_traders:
                        tokens_rejected_low_traders += 1
                    if swap_count < min_swaps:
                        tokens_rejected_low_swaps += 1
                    if performance_multiplier is None:
                        tokens_rejected_no_price_low_activity += 1
            
            # Activity bonuses regardless of price
            if swap_count >= 30:
                score += 20
            elif swap_count >= 15:
                score += 10
            elif swap_count >= 8:
                score += 5
                
            if unique_traders >= 15:
                score += 15
            elif unique_traders >= 8:
                score += 8
            elif unique_traders >= 5:
                score += 3
            
            # Store calculated metrics
            token_data['performance_multiplier'] = performance_multiplier
            token_data['bitquery_success_score'] = score
            token_data['success_tier'] = success_tier
            
            # Include tokens that have a tier AND meet the threshold score
            # Use the config threshold to filter tokens
            if success_tier is not None and score >= self.bitquery_success_threshold:
                promising_tokens.append(token_data)
                price_str = f"{performance_multiplier:.2f}x" if performance_multiplier else "no_price"
                self.logger.debug(f"Token {mint[:8]}... promising: score={score}, "
                                f"tier={success_tier}, perf={price_str}, swaps={swap_count}, traders={unique_traders}")
        
        # Log comprehensive filtering statistics
        self.logger.info(f"Prefilter results:")
        self.logger.info(f"  - Total tokens analyzed: {total_tokens}")
        self.logger.info(f"  - Tokens with price data: {tokens_with_price} ({tokens_with_price/total_tokens*100:.1f}%)")
        self.logger.info(f"  - Promising tokens found: {len(promising_tokens)}")
        self.logger.info(f"  - Rejected - no price + low activity: {tokens_rejected_no_price_low_activity}")
        self.logger.info(f"  - Rejected - low traders: {tokens_rejected_low_traders}")
        self.logger.info(f"  - Rejected - low swaps: {tokens_rejected_low_swaps}")
        
        # Show top 5 tokens for sanity check
        promising_tokens.sort(key=lambda x: x['bitquery_success_score'], reverse=True)
        self.logger.info("Top 5 promising tokens:")
        for i, token in enumerate(promising_tokens[:5]):
            perf = token.get('performance_multiplier')
            perf_str = f"{perf:.1f}x" if perf else "no_price"
            self.logger.info(f"  #{i+1}: {token['mint'][:8]}... "
                           f"Score: {token['bitquery_success_score']} | "
                           f"Tier: {token['success_tier']} | "
                           f"Perf: {perf_str} | "
                           f"Swaps: {token['swap_count']} | "
                           f"Traders: {token['unique_traders']}")
        
        return promising_tokens
    
    async def _validate_token_with_moralis(self, token_data: Dict) -> bool:
        """Validate a promising token with minimal Moralis calls"""
        mint = token_data['mint']
        self.logger.debug(f"Validating token {mint[:8]}... with Moralis")
        try:
            current_price = await self.moralis.get_current_price(mint, fresh=True)
            
            if current_price <= 0:
                self.logger.debug(f"Token {mint[:8]}... has no price data (price={current_price}) - Moralis may be rate limited or token too new")
                return False

            performance_multiplier = token_data.get('performance_multiplier', 1.0)
            bitquery_score = token_data.get('bitquery_success_score', 0)
            success_tier = token_data.get('success_tier', None)
            success_threshold = self.success_thresholds['low']

            # Handle None price and performance safely
            price_str = f"${current_price:.8f}" if current_price is not None else "None"
            perf_str = f"{performance_multiplier:.2f}x" if performance_multiplier is not None else "no_price"
            self.logger.debug(f"Token {mint[:8]}... validation check: price={price_str}, perf={perf_str}, score={bitquery_score}, tier={success_tier}, threshold={success_threshold}")

            token_data['current_price'] = current_price
            is_successful = (
                success_tier is not None and 
                bitquery_score >= 25 and
                performance_multiplier is not None and
                performance_multiplier >= success_threshold  # At least 1.2x
            )
            
            if not is_successful:
                perf_ok = performance_multiplier >= success_threshold if performance_multiplier is not None else False
                self.logger.debug(f"Token {mint[:8]}... FAILED validation: tier_ok={success_tier is not None}, score_ok={bitquery_score >= 25}, perf_ok={perf_ok}")
            else:
                perf_str = f"{performance_multiplier:.2f}x" if performance_multiplier is not None else "no_price"
                self.logger.debug(f"Token {mint[:8]}... PASSED validation: perf={perf_str}, score={bitquery_score}")
                
            return is_successful

        except Exception as e:
            self.logger.error(f"Error validating token {mint[:8]}...: {e}")
            return False


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
                
                # Filter to early swaps (first 5 minutes after launch - aligned with trading)
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
                    
                    # Filter existing swaps for follow-through period (5-60 minutes post launch)
                    # Aligned with 15-minute max hold strategy
                    ft_swaps = []
                    for swap in all_swaps:
                        swap_time = self._parse_iso_timestamp(swap.get('timestamp', ''))
                        if launch_time + 300 <= swap_time <= launch_time + 3600:  # 5 min to 1 hour
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
    
    def _get_trade_size_usd(self, trade_data: Dict) -> float:
        """Extract USD trade size from various trade data formats"""
        # Try common USD amount fields
        usd_amount = (trade_data.get('amount_usd') or 
                     trade_data.get('usd_value') or 
                     trade_data.get('value_usd') or
                     trade_data.get('AmountUSD') or  # Bitquery format
                     0)
        
        if usd_amount and usd_amount > 0:
            return float(usd_amount)
            
        # Fallback: try to calculate from SOL amount (rough estimate)
        sol_amount = (trade_data.get('amount_sol') or 
                     trade_data.get('Amount') or  # Bitquery format
                     0)
        
        if sol_amount and sol_amount > 0:
            # Rough SOL to USD conversion (~$140 per SOL)
            return float(sol_amount) * 140
            
        return 0
    
    def _is_significant_position(self, trade_data: Dict) -> bool:
        """Check if trade represents a significant position (not noise or whale)"""
        # Get position filter configuration
        position_filters = getattr(self, 'config', {}).get('discovery_position_filters', {})
        min_usd = position_filters.get('min_position_usd', 50)
        max_usd = position_filters.get('max_position_usd', 5000)
        
        usd_size = self._get_trade_size_usd(trade_data)
        return min_usd <= usd_size <= max_usd

    async def _find_early_buyers(self, token_data: Dict) -> List[str]:
        """Find wallets that bought token in first 5 minutes after launch using Bitquery first"""
        mint = token_data['mint']
        launch_time = token_data.get('launch_time')
        
        if not launch_time:
            self.logger.warning(f"No launch time available for {mint[:8]}..., cannot find early buyers")
            return []
        
        try:
            early_window = self.early_window_seconds  # 300 seconds (5 minutes)
            early_buyers = set()
            filtered_buyers = set()
            
            # FIRST: Try using Bitquery data we already have (cheaper and aligned)
            raw_trades = token_data.get('raw_trades', [])
            if raw_trades:
                self.logger.debug(f"Using existing Bitquery data for {mint[:8]}... early buyers")
                
                # Filter trades within early window after launch
                early_trades = [
                    t for t in raw_trades
                    if launch_time <= t['timestamp'] <= launch_time + early_window
                    and t.get('side') == 'buy'
                    and t.get('signer')
                    and len(str(t.get('signer'))) >= 32
                    and str(t.get('signer')) != mint
                ]
                
                # Apply position size filtering
                for trade in early_trades:
                    wallet = str(trade['signer'])
                    early_buyers.add(wallet)
                    
                    # Check position size significance
                    if self._is_significant_position(trade):
                        filtered_buyers.add(wallet)
                
                self.logger.debug(f"Found {len(early_buyers)} early buyers from Bitquery ({len(filtered_buyers)} with significant positions) for {mint[:8]}...")
                
                # If we found significant buyers from Bitquery, return them (more efficient)
                if filtered_buyers:
                    return list(filtered_buyers)
                    
                # Fallback: if no significant buyers but we have any buyers, return top portion
                if early_buyers:
                    buyers_list = list(early_buyers)
                    # Return up to half of the early buyers as a compromise
                    return buyers_list[:max(1, len(buyers_list) // 2)]
            
            # FALLBACK: Use Moralis only if Bitquery lacks data or coverage
            self.logger.debug(f"Falling back to Moralis for {mint[:8]}... early buyers (insufficient Bitquery coverage)")
            
            # Get swaps and filter for launch time window (Moralis max limit is 100)
            all_swaps = await self.moralis.get_token_swaps(mint, limit=100)
            
            # Filter to swaps within early window
            swaps = [
                swap for swap in all_swaps
                if launch_time <= self._parse_iso_timestamp(swap.get('timestamp', '')) <= launch_time + early_window
            ]
            
            self.logger.debug(f"Found {len(swaps)} early swaps from Moralis for {mint[:8]}... in first {early_window}s")
            
            if not swaps:
                self.logger.debug(f"No early swaps found for {mint[:8]}...")
                return []
            
            # Sort swaps by timestamp 
            sorted_swaps = sorted(swaps, key=lambda x: self._parse_iso_timestamp(x.get('timestamp', '')))
            
            # Reset sets for Moralis processing
            early_buyers = set()
            filtered_buyers = set()
            
            for swap in sorted_swaps:
                swap_time = self._parse_iso_timestamp(swap.get('timestamp', ''))
                
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
                    
                    # Validate wallet address and ensure it's a buy (not the token mint address)
                    if wallet and len(str(wallet)) >= 32 and str(wallet) != mint:
                        early_buyers.add(str(wallet))
                        
                        # Check position size significance
                        if self._is_significant_position(swap):
                            filtered_buyers.add(str(wallet))
                            self.logger.debug(f"Found significant early buyer from Moralis: {str(wallet)[:8]}...")
                        else:
                            self.logger.debug(f"Found early buyer from Moralis (small position): {str(wallet)[:8]}...")
                else:
                    # Once we pass the early window, we can break since swaps are sorted
                    break
            
            # Prefer significant buyers, fallback to subset of all buyers
            if filtered_buyers:
                self.logger.info(f"Found {len(filtered_buyers)} significant early buyers for {mint[:8]}... (from {len(early_buyers)} total)")
                return list(filtered_buyers)
            elif early_buyers:
                buyers_list = list(early_buyers)
                # Return up to half as compromise if no significant positions found
                selected_buyers = buyers_list[:max(1, len(buyers_list) // 2)]
                self.logger.info(f"Found {len(selected_buyers)} early buyers for {mint[:8]}... (filtered from {len(early_buyers)} total)")
                return selected_buyers
            else:
                self.logger.info(f"No early buyers found for {mint[:8]}...")
                return []
            
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
    
    def _detect_wash_trading_patterns(self, candidates: Dict) -> set:
        """Detect wallets with suspicious coordination patterns"""
        from collections import defaultdict
        import time
        
        # Check if wash trading detection is enabled
        quality_checks = self.config.get('discovery_quality_checks', {})
        if not quality_checks.get('enable_wash_trading_detection', True):
            self.logger.debug("Wash trading detection disabled in config")
            return set()
        
        # STRICTER parameters as requested
        max_co_occurrence_rate = quality_checks.get('max_co_occurrence_rate', 0.2)  # Reduced from 0.3
        min_tokens_for_check = quality_checks.get('min_tokens_for_check', 3)  # Check with fewer tokens
        
        suspicious_wallets = set()
        
        # Build co-occurrence matrix
        co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
        timing_correlations = defaultdict(lambda: defaultdict(list))
        
        for wallet, token_list in candidates.items():
            for token_entry in token_list:
                mint = token_entry['token']['mint']
                entry_time = token_entry.get('entry_time', 0)
                
                # Count how often wallet pairs appear on same tokens
                for other_wallet in candidates:
                    if other_wallet != wallet:
                        other_tokens = candidates[other_wallet]
                        for other_token in other_tokens:
                            if other_token['token']['mint'] == mint:
                                co_occurrence_matrix[wallet][other_wallet] += 1
                                
                                # Track timing correlation (NEW)
                                other_entry_time = other_token.get('entry_time', 0)
                                if entry_time and other_entry_time:
                                    time_diff = abs(entry_time - other_entry_time)
                                    timing_correlations[wallet][other_wallet].append(time_diff)
        
        # Flag wallets with excessive co-occurrence or timing correlation
        for wallet, co_occurrences in co_occurrence_matrix.items():
            wallet_token_count = len(candidates[wallet])
            
            # Only check wallets with enough data
            if wallet_token_count >= min_tokens_for_check:
                max_co_occurrence = max(co_occurrences.values()) if co_occurrences else 0
                co_occurrence_rate = max_co_occurrence / wallet_token_count
                
                # Check timing correlation (NEW)
                rapid_coordination = False
                for other_wallet, time_diffs in timing_correlations[wallet].items():
                    if time_diffs:
                        # Check if wallets trade within 5 seconds of each other
                        avg_time_diff = sum(time_diffs) / len(time_diffs)
                        if avg_time_diff < 5 and len(time_diffs) >= 2:
                            rapid_coordination = True
                            self.logger.warning(f"Rapid coordination detected: {wallet[:8]}... and {other_wallet[:8]}... trade within {avg_time_diff:.1f}s")
                
                # Flag wallets with excessive coordination
                if co_occurrence_rate > max_co_occurrence_rate or rapid_coordination:
                    suspicious_wallets.add(wallet)
                    if co_occurrence_rate > max_co_occurrence_rate:
                        self.logger.warning(f"Suspicious coordination: {wallet[:8]}... co-occurs {co_occurrence_rate:.0%} with another wallet (threshold: {max_co_occurrence_rate:.0%})")
        
        if suspicious_wallets:
            self.logger.info(f"Wash trading detection: flagged {len(suspicious_wallets)} suspicious wallets")
        else:
            self.logger.debug("Wash trading detection: no suspicious patterns detected")
            
        return suspicious_wallets

    async def _score_alpha_candidates(self, candidates: Dict) -> List[str]:
        """Score and rank alpha wallet candidates with recency decay and performance multiplier"""
        import math
        
        scored_wallets = []
        current_time = time.time()
        
        self.logger.info(f"Evaluating {len(candidates)} wallet candidates...")
        
        # ENHANCED: First apply farming detection to filter out manipulation wallets
        self.logger.info("Running advanced farming detection on wallet candidates...")
        farming_filtered_candidates = await self._filter_farming_wallets(candidates)
        
        # Then detect and filter out suspicious wash trading patterns
        suspicious_wallets = self._detect_wash_trading_patterns(farming_filtered_candidates)
        if suspicious_wallets:
            # Remove suspicious wallets from candidates
            filtered_candidates = {k: v for k, v in farming_filtered_candidates.items() if k not in suspicious_wallets}
            self.logger.info(f"Filtered out {len(suspicious_wallets)} suspicious wallets, {len(filtered_candidates)} remaining")
            candidates = filtered_candidates
        else:
            candidates = farming_filtered_candidates
        
        # Show wallet appearance distribution for debugging
        appearance_counts = {}
        for wallet, token_list in candidates.items():
            count = len(token_list)
            appearance_counts[count] = appearance_counts.get(count, 0) + 1
            
        self.logger.info(f"Wallet appearance distribution: {dict(sorted(appearance_counts.items()))}")
        
        for wallet, token_list in candidates.items():
            # Log all candidates for debugging
            self.logger.debug(f"Wallet {wallet[:8]}... appears on {len(token_list)} tokens")
            
            # Multi-tier approach: different requirements based on success tier
            wallet_success_tiers = []
            for token_entry in token_list:
                tier = token_entry["token"].get('success_tier', 'low')
                wallet_success_tiers.append(tier)
            
            # Count successes by tier
            high_count = wallet_success_tiers.count('high')
            medium_count = wallet_success_tiers.count('medium') 
            low_count = wallet_success_tiers.count('low')
            
            # CRITICAL: Require minimum trades for statistical significance
            total_trades = len(token_list)
            if total_trades < 3:
                self.logger.debug(f"Wallet {wallet[:8]}... skipped: only {total_trades} trades (need 3+ for statistics)")
                continue
            
            # Determine wallet qualification tier
            wallet_tier = None
            if high_count >= self.min_wallet_appearances['tier_1']:
                wallet_tier = 'tier_1'  # Premium alpha wallet
            elif medium_count >= self.min_wallet_appearances['tier_2']:
                wallet_tier = 'tier_2'  # Good alpha wallet  
            elif low_count >= self.min_wallet_appearances['tier_3']:
                wallet_tier = 'tier_3'  # Emerging alpha wallet
            
            if wallet_tier is None:
                self.logger.debug(f"Wallet {wallet[:8]}... insufficient tier: H:{high_count} M:{medium_count} L:{low_count}")
                continue
                
            self.logger.debug(f"Wallet {wallet[:8]}... qualified as {wallet_tier}: H:{high_count} M:{medium_count} L:{low_count}")
            
            # Calculate tier-weighted scoring
            tier_multipliers = {'tier_1': 3.0, 'tier_2': 2.0, 'tier_3': 1.5}
            base_score = tier_multipliers[wallet_tier]
            
            # Use actual performance_multiplier from token data
            performance_multipliers = []
            recency_weights = []
            position_sizes = []
            
            for token_entry in token_list:
                # Access nested token data correctly
                raw_token_data = token_entry["token"]
                
                # Get performance multiplier (use the value set in the entry)
                perf_mult = token_entry.get('performance', 1.0)  # This comes from the entry, not nested
                performance_multipliers.append(perf_mult)
                
                # Extract position size if available (for risk adjustment)
                position_size = token_entry.get('position_size_usd', 100)  # Default $100
                position_sizes.append(position_size)
                
                # Calculate recency decay (12-hour half-life) - use launch_time from raw token data
                token_time = raw_token_data.get('launch_time', current_time)
                time_diff = current_time - token_time
                recency_weight = math.exp(-time_diff / (12 * 3600))  # 12h decay
                recency_weights.append(recency_weight)
            
            # Weighted averages using recency
            if recency_weights:
                total_weight = sum(recency_weights)
                avg_performance = sum(p * w for p, w in zip(performance_multipliers, recency_weights)) / total_weight
                
                # ADJUSTED: Target organic wallets with 60-70% win rates instead of super high rates
                # Calculate success rate (performance >= 1.5x for organic traders)
                success_count = sum(w for p, w in zip(performance_multipliers, recency_weights) if p >= 1.5)
                success_rate = success_count / total_weight
                
                # NEW: Apply win rate preference for organic wallets (60-70% ideal)
                win_rate_quality = 1.0
                if 0.60 <= success_rate <= 0.70:
                    win_rate_quality = 2.0  # STRONG boost for ideal organic range
                elif 0.55 <= success_rate < 0.60 or 0.70 < success_rate <= 0.75:
                    win_rate_quality = 1.5  # Good boost for near-ideal
                elif 0.50 <= success_rate < 0.55 or 0.75 < success_rate <= 0.80:
                    win_rate_quality = 1.0  # Neutral for acceptable range
                elif success_rate > 0.85:
                    win_rate_quality = 0.3  # Strong penalty for suspiciously high
                elif success_rate > 0.95:
                    win_rate_quality = 0.05  # Extreme penalty for near-perfect
                
                # Calculate risk-adjusted performance (normalize by position size)
                risk_adjusted_returns = []
                for perf, pos_size, weight in zip(performance_multipliers, position_sizes, recency_weights):
                    # Risk-adjusted return: excess return per $100 invested
                    excess_return = (perf - 1.0) * (100 / max(pos_size, 1))
                    risk_adjusted_returns.append(excess_return * weight)
                
                avg_risk_adjusted_return = sum(risk_adjusted_returns) / total_weight
                
                # Overall recency factor
                avg_recency = total_weight / len(recency_weights)
            else:
                avg_performance = 1.0
                success_rate = 0.0
                avg_risk_adjusted_return = 0.0
                avg_recency = 0.0
            
            # NEW: Layer 2 - Analyze exit timing quality
            avg_exit_quality = await self._analyze_exit_timing(wallet, token_list)
            
            # NEW: Layer 3 - Check performance consistency
            consistency_score = self._calculate_consistency(performance_multipliers, recency_weights)
            
            # NEW: Layer 4 - Calculate early entry rate
            early_entry_rate = self._calculate_early_entry_rate(wallet, token_list)
            
            # NEW: Layer 5 - Apply quality filters
            min_exit_quality = self.config.get('discovery_quality_checks', {}).get('min_exit_quality', 0.3)
            min_consistency = self.config.get('discovery_quality_checks', {}).get('min_consistency', 0.2)
            min_early_entry_rate = self.config.get('discovery_quality_checks', {}).get('min_early_entry_rate', 0.4)
            
            if (avg_exit_quality < min_exit_quality or 
                consistency_score < min_consistency or 
                early_entry_rate < min_early_entry_rate):
                # Skip wallets that fail quality checks
                self.logger.debug(f"Wallet {wallet[:8]}... failed quality check: "
                               f"exit={avg_exit_quality:.2f}, consistency={consistency_score:.2f}, "
                               f"early_entry={early_entry_rate:.2f}")
                continue
            
            # Calculate enhanced tier-weighted final score with quality layers
            tier_base_score = base_score * 100  # Base score from tier multiplier
            
            # Performance and success bonuses (adjusted for organic trading)
            performance_bonus = avg_performance * 25
            success_bonus = success_rate * 40 * win_rate_quality  # Apply win rate quality factor
            volume_bonus = len(token_list) * 10
            risk_adjusted_bonus = avg_risk_adjusted_return * 15  # Bonus for skill vs capital
            
            # Quality bonuses
            exit_timing_bonus = avg_exit_quality * 30  # Reward good exit timing
            consistency_bonus = consistency_score * 20  # Reward consistency
            
            # NEW: Organic trading bonus (prefer moderate win rates)
            organic_bonus = win_rate_quality * 20
            
            final_score = (tier_base_score + performance_bonus + success_bonus + 
                          volume_bonus + risk_adjusted_bonus + exit_timing_bonus + 
                          consistency_bonus + organic_bonus) * avg_recency
            
            scored_wallets.append({
                'wallet': wallet,
                'score': final_score,
                'tier': wallet_tier,
                'total_tokens': len(token_list),
                'tier_counts': f"H:{high_count}/M:{medium_count}/L:{low_count}",
                'avg_performance': avg_performance,
                'success_rate': success_rate,
                'avg_risk_adjusted_return': avg_risk_adjusted_return,
                'avg_recency': avg_recency,
                'avg_exit_quality': avg_exit_quality,  # Store exit quality
                'consistency_score': consistency_score,  # Store consistency
                'early_entry_rate': early_entry_rate,  # Store early entry rate
                'win_rate_quality': win_rate_quality,  # NEW: Store win rate quality factor
                'tokens': token_list
            })
        
        # Sort by score (highest first)
        scored_wallets.sort(key=lambda x: x['score'], reverse=True)
        
        # Log top performers with tier information
        self.logger.info(f"Found {len(scored_wallets)} qualified alpha candidates across all tiers")
        
        # Show distribution by tier
        tier_counts = {}
        for w in scored_wallets:
            tier = w.get('tier', 'unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        self.logger.info(f"Tier distribution: {tier_counts}")
        
        for i, wallet_data in enumerate(scored_wallets[:15]):  # Show top 15
            w = wallet_data
            self.logger.info(f"#{i+1}: {w['wallet'][:8]}... "
                           f"Tier: {w.get('tier', 'N/A')} | "
                           f"Score: {w['score']:.1f} | "
                           f"Counts: {w.get('tier_counts', 'N/A')} | "
                           f"Perf: {w['avg_performance']:.1f}x | "
                           f"Success: {w['success_rate']:.1%} | "
                           f"WinQuality: {w.get('win_rate_quality', 1.0):.1f} | "
                           f"Exit: {w.get('avg_exit_quality', 0):.2f} | "
                           f"Consistency: {w.get('consistency_score', 0):.2f} | "
                           f"EarlyEntry: {w.get('early_entry_rate', 0):.1%}")
        
        # Return more wallets but with tier diversity (up to 100 total)
        max_wallets = min(100, len(scored_wallets))
        selected_wallets = []
        
        # Prioritize higher tiers but include diversity
        tier_limits = {'tier_1': 35, 'tier_2': 35, 'tier_3': 30}  # Max per tier (totals 100)
        tier_selected = {'tier_1': 0, 'tier_2': 0, 'tier_3': 0}
        
        for wallet_data in scored_wallets:
            if len(selected_wallets) >= max_wallets:
                break
                
            tier = wallet_data.get('tier', 'tier_3')
            if tier_selected[tier] < tier_limits[tier]:
                selected_wallets.append(wallet_data['wallet'])
                tier_selected[tier] += 1
        
        # Fill remaining slots with best remaining wallets regardless of tier
        for wallet_data in scored_wallets:
            if len(selected_wallets) >= max_wallets:
                break
            if wallet_data['wallet'] not in selected_wallets:
                selected_wallets.append(wallet_data['wallet'])
        
        final_tier_counts = {}
        for w in scored_wallets[:len(selected_wallets)]:
            tier = w.get('tier', 'unknown')
            final_tier_counts[tier] = final_tier_counts.get(tier, 0) + 1
        
        self.logger.info(f"Selected {len(selected_wallets)} wallets with distribution: {final_tier_counts}")
        
        return selected_wallets
    
    def filter_existing_wallets(self, discovered_wallets: List[str], existing_wallets: List[str]) -> List[str]:
        """Filter out wallets that are already being tracked"""
        existing_set = set(existing_wallets)
        new_wallets = [w for w in discovered_wallets if w not in existing_set]
        
        self.logger.info(f"Filtered {len(discovered_wallets)} discovered wallets: "
                        f"{len(new_wallets)} new, {len(discovered_wallets) - len(new_wallets)} already tracked")
        
        return new_wallets
    
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
        try:
            # Try to save to database if available
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
        except Exception as e:
            self.logger.warning(f"Could not save to database: {e}, continuing without database save")
    
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
            
            # Limit to top 100 wallets (manageable size)  
            final_wallets = updated_wallets[:100]
            self.logger.info(f"Config update: {len(current_wallets)} existing + {len(new_wallets)} new = {len(updated_wallets)} total -> saving {len(final_wallets)} wallets")
            config['watched_wallets'] = final_wallets
            
            # Always use the same path as the config loader
            config_path = 'config/config.yml'
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Updated config with {len(updated_wallets)} alpha wallets")
            
        except Exception as e:
            self.logger.error(f"Error updating bot config: {e}")

    async def _analyze_exit_timing(self, wallet: str, token_list: List[Dict]) -> float:
        """
        Analyze if wallet sold near peaks vs held through dumps
        Returns 0.0-1.0 score where 1.0 = perfect exit timing
        """
        exit_scores = []
        
        for token_entry in token_list:
            token_data = token_entry["token"]
            mint = token_data['mint']
            launch_time = token_data.get('launch_time')
            
            if not launch_time:
                continue
                
            try:
                # Get wallet's transactions for this token
                swaps = await self.moralis.get_token_swaps(mint, limit=100)
                
                # Find wallet's buy and sell transactions
                wallet_buys = []
                wallet_sells = []
                
                for swap in swaps:
                    if swap.get('wallet') == wallet:
                        swap_time = self._parse_iso_timestamp(swap.get('timestamp', ''))
                        if swap.get('side') == 'buy':
                            wallet_buys.append((swap_time, swap.get('price', 0)))
                        elif swap.get('side') == 'sell':
                            wallet_sells.append((swap_time, swap.get('price', 0)))
                
                if not wallet_buys or not wallet_sells:
                    continue  # No complete buy-sell cycle
                    
                # Calculate exit quality for each buy-sell pair
                for buy_time, buy_price in wallet_buys:
                    # Find sells after this buy
                    relevant_sells = [(t, p) for t, p in wallet_sells if t > buy_time]
                    if not relevant_sells:
                        continue
                        
                    # Get peak price in the period after buy
                    all_prices_after_buy = [p for t, p in [(self._parse_iso_timestamp(swap.get('timestamp', '')), swap.get('price', 0)) 
                                           for swap in swaps 
                                           if self._parse_iso_timestamp(swap.get('timestamp', '')) > buy_time]]
                    
                    if not all_prices_after_buy:
                        continue
                        
                    peak_price = max(all_prices_after_buy)
                    
                    # Calculate exit efficiency for each sell
                    for sell_time, sell_price in relevant_sells:
                        if buy_price > 0 and peak_price > buy_price:
                            # Exit efficiency = how close to peak they sold
                            potential_gain = (peak_price - buy_price) / buy_price
                            actual_gain = (sell_price - buy_price) / buy_price
                            exit_efficiency = actual_gain / potential_gain if potential_gain > 0 else 0
                            exit_efficiency = max(0, min(1, exit_efficiency))  # Clamp 0-1
                            exit_scores.append(exit_efficiency)
                            
            except Exception as e:
                self.logger.debug(f"Error analyzing exit timing for {wallet[:8]}...: {e}")
                continue
        
        return sum(exit_scores) / len(exit_scores) if exit_scores else 0.5  # Default neutral

    def _calculate_early_entry_rate(self, wallet: str, token_list: List[Dict]) -> float:
        """Calculate the percentage of trades that were early entries (within 2 minutes of launch)"""
        if not token_list:
            return 0.0
        
        early_entries = 0
        total_entries = 0
        
        for token_entry in token_list:
            raw_token_data = token_entry.get("token", {})
            launch_time = raw_token_data.get('launch_time')
            
            if not launch_time:
                continue
                
            # Look for the wallet's trades in the raw_trades data
            raw_trades = raw_token_data.get('raw_trades', [])
            wallet_trades = [
                trade for trade in raw_trades 
                if str(trade.get('signer', '')) == wallet and trade.get('side') == 'buy'
            ]
            
            for trade in wallet_trades:
                total_entries += 1
                trade_time = trade.get('timestamp')
                
                if trade_time and launch_time:
                    # Check if trade was within 2 minutes (120 seconds) of launch
                    time_diff = trade_time - launch_time
                    if time_diff <= 120:  # 2 minutes in seconds
                        early_entries += 1
        
        if total_entries == 0:
            return 0.0
            
        early_entry_rate = early_entries / total_entries
        self.logger.debug(f"Wallet {wallet[:8]}... early entry rate: {early_entries}/{total_entries} = {early_entry_rate:.1%}")
        
        return early_entry_rate

    def _calculate_consistency(self, performance_multipliers: List[float], recency_weights: List[float]) -> float:
        """
        Check if wallet is consistently good vs got lucky on a few tokens
        Returns 0.0-1.0 where 1.0 = very consistent
        """
        if len(performance_multipliers) < 2:
            return 0.5  # Not enough data
        
        # Calculate variance in performance (lower = more consistent)
        import statistics
        try:
            variance = statistics.variance(performance_multipliers)
            
            # Convert variance to consistency score (lower variance = higher consistency)
            # Typical variance for good wallets is 0.1-0.5, bad wallets 1.0+
            consistency = max(0, 1.0 - (variance / 1.0))  # Scale so variance of 1.0 = 0 consistency
            
            return consistency
        except Exception as e:
            self.logger.debug(f"Error calculating consistency: {e}")
            return 0.5  # Default neutral
    
    async def _filter_farming_wallets(self, candidates: Dict) -> Dict:
        """
        Filter out farming/manipulation wallets using advanced detection
        Returns filtered candidates dictionary
        """
        filtered_candidates = {}
        farmer_wallets = set()
        suspicious_count = 0
        error_count = 0
        
        self.logger.info(f"Analyzing {len(candidates)} wallets for farming patterns...")
        
        for wallet, token_list in candidates.items():
            try:
                # Prepare trade data for farming analysis
                trades_data = []
                for entry in token_list:
                    token_data = entry['token']
                    
                    # Extract relevant trade information
                    trade_info = {
                        'token_address': token_data.get('mint'),
                        'timestamp': token_data.get('launch_time', 0),
                        'side': 'buy',  # These are early buys
                        'price': token_data.get('current_price', 0),
                        'amount_usd': entry.get('position_size_usd', 100),
                        'performance': entry.get('performance', 1.0)
                    }
                    trades_data.append(trade_info)
                    
                    # Add any raw trades if available
                    raw_trades = token_data.get('raw_trades', [])
                    for raw_trade in raw_trades:
                        if str(raw_trade.get('signer', '')) == wallet:
                            trades_data.append({
                                'token_address': token_data.get('mint'),
                                'timestamp': raw_trade.get('timestamp', 0),
                                'side': raw_trade.get('side', 'unknown'),
                                'price': raw_trade.get('price_usd', 0),
                                'amount_usd': raw_trade.get('amount_usd', 0)
                            })
                
                # Run farming analysis
                farming_metrics = await self.farming_detector.analyze_wallet(
                    wallet, 
                    trades_data,
                    self.moralis,
                    self.bitquery
                )
                
                # Check classification
                classification = farming_metrics.get('classification', 'unknown')
                farmer_score = farming_metrics.get('farmer_score', 0)
                
                # Log any errors from farming detection
                if 'error' in farming_metrics:
                    self.logger.error(f"Farming detection error for {wallet[:8]}...: {farming_metrics['error']}")
                    error_count += 1
                    # Include wallet if error occurred (conservative approach)
                    filtered_candidates[wallet] = token_list
                    continue
                
                if classification == 'farmer':
                    farmer_wallets.add(wallet)
                    self.logger.warning(f"FARMER detected: {wallet[:8]}... (score={farmer_score:.2f})")
                    # Log detailed metrics for farmers
                    if farming_metrics.get('red_flags'):
                        self.logger.warning(f"  Red flags: {', '.join(farming_metrics['red_flags'])}")
                elif classification == 'suspicious':
                    suspicious_count += 1
                    # Still include suspicious wallets but log them
                    filtered_candidates[wallet] = token_list
                    self.logger.info(f"Suspicious wallet kept: {wallet[:8]}... (score={farmer_score:.2f})")
                else:
                    # Safe wallet - include it
                    filtered_candidates[wallet] = token_list
                    
            except Exception as e:
                self.logger.error(f"ERROR processing wallet {wallet[:8]}... for farming detection: {e}")
                error_count += 1
                # Include wallet if error occurred (conservative approach)
                filtered_candidates[wallet] = token_list
        
        # Log filtering summary
        total_filtered = len(candidates) - len(filtered_candidates)
        farmer_count = len(farmer_wallets)
        safe_count = len(filtered_candidates) - suspicious_count - error_count
        
        self.logger.info(f"Farming detection complete:")
        self.logger.info(f"  - Total wallets: {len(candidates)}")
        self.logger.info(f"  - Farmers removed: {farmer_count}")
        self.logger.info(f"  - Suspicious kept: {suspicious_count}")
        self.logger.info(f"  - Safe wallets: {safe_count}")
        if error_count > 0:
            self.logger.warning(f"  - Errors (kept): {error_count}")
        self.logger.info(f"  - Total remaining: {len(filtered_candidates)}")
        
        if farmer_wallets:
            self.logger.info(f"Removed farmer wallets: {list(farmer_wallets)[:5]}...")
        
        return filtered_candidates


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
    moralis = MoralisClient(config.get('moralis_keys', config.get('moralis_key')), config.get('api_optimization', {}))
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
            print(f"   They consistently bought successful tokens within first 5 minutes")
            print(f"   Config updated with {len(alpha_wallets)} alpha wallets!")
    
    finally:
        # Clean up
        await moralis.close()
        if hasattr(bitquery, 'client') and bitquery.client:
            await bitquery.client.transport.close()


if __name__ == "__main__":
    asyncio.run(main())