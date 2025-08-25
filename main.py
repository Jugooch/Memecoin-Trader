#!/usr/bin/env python3
"""
Automated Memecoin Trading Bot
Follows the implementation plan for turning $100 into $100K

COST-EFFECTIVE DESIGN:
- Uses FREE Solana RPC (no expensive subscriptions needed)
- Relies on Bitquery + Moralis APIs for intelligence
- Waits for alpha wallet confirmation (quality over speed)
- Makes fewer, smarter trades rather than racing milliseconds
"""

import asyncio
import logging
import os
import yaml
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.clients.realtime_client import RealtimeClient
from src.clients.moralis_client import MoralisClient
from src.clients.pumpfun_client import PumpFunClient
from src.core.wallet_tracker import WalletTracker
from src.core.trading_engine import TradingEngine
from src.core.database import Database
from src.core.wallet_rotation_manager import WalletRotationManager
from src.core.safety_checks import SafetyChecker
from src.core.risk_manager import AdaptiveRiskManager
from src.utils.logger_setup import setup_logging
from src.utils.config_loader import load_config, validate_required_keys


@dataclass
class TradingConfig:
    rpc_endpoint: str
    bitquery_token: List[str]
    moralis_key: List[str]
    quicknode_endpoint: str
    quicknode_api_key: str
    watched_wallets: List[str]
    threshold_alpha_buys: int
    min_liquidity_usd: float
    time_window_sec: int
    max_trade_pct: float
    tp_multiplier: float
    stop_loss_pct: float
    paper_mode: bool
    initial_capital: float
    max_trades_per_day: int
    min_time_between_trades: int
    pump_fun_program: str
    pump_fun_api: str
    logging_level: str
    logging_file: str
    database_file: str
    # Fields with defaults must come last
    max_concurrent_positions: int = 3  # Add max concurrent positions
    notifications: Dict = None  # Add notifications section
    alpha_weight_min: float = 3.5  # Add alpha weight minimum threshold (conservative default)
    alpha_enhanced: Dict = None  # Add alpha enhanced configuration section
    risk_management: Dict = None  # Add risk management configuration section
    min_confidence: float = 50.0  # Add minimum confidence threshold


class MemecoinTradingBot:
    def __init__(self, config_path: str = "config.yml"):
        self.config = self._load_config(config_path)
        self.logger = setup_logging(self.config.logging_level, self.config.logging_file)
        
        self.realtime_client = RealtimeClient(self._get_realtime_config())
        self.moralis = MoralisClient(self.config.moralis_key)
        
        # QuickNode is optional - only for automated trading
        self.pumpfun = None
        if hasattr(self.config, 'quicknode_endpoint') and self.config.quicknode_endpoint:
            self.pumpfun = PumpFunClient(self.config.quicknode_endpoint, self.config.quicknode_api_key)
        
        # Pass config dict to WalletTracker for Bayesian scoring parameters
        wallet_config = {
            'beta_prior_alpha': getattr(self.config, 'beta_prior_alpha', 3),
            'beta_prior_beta': getattr(self.config, 'beta_prior_beta', 5),
            'alpha_weight_min': getattr(self.config, 'alpha_weight_min', 3.5),
            'require_one_wallet_pge_55': getattr(self.config, 'require_one_wallet_pge_55', False),
            'alpha_enhanced': getattr(self.config, 'alpha_enhanced', {})
        }
        self.wallet_tracker = WalletTracker(self.config.watched_wallets, config=wallet_config)
        self.trading_engine = TradingEngine(self.config, moralis_client=self.moralis)
        self.database = Database(self.config.database_file)
        
        # Initialize wallet rotation manager (will get discord notifier and realtime client later)
        self.wallet_rotation_manager = WalletRotationManager(
            self.wallet_tracker, self.realtime_client.bitquery_client, self.moralis, self.database, config_path,
            realtime_client=self.realtime_client
        )
        
        # Monitoring state
        self._pumpportal_monitoring = False
        
        # Initialize safety checker and risk manager (Phase 4)
        self.safety_checker = SafetyChecker()
        self.risk_manager = AdaptiveRiskManager(
            pnl_store=self.trading_engine.pnl_store if hasattr(self.trading_engine, 'pnl_store') else None,
            config={'initial_capital': self.config.initial_capital}
        )
        
        self.running = False
        self.trades_today = 0
        self.last_trade_time = None
        self.current_capital = self.config.initial_capital
        
        # Tracking counters for summary logging
        self.tokens_processed = 0
        self.alpha_checks_performed = 0
        
        # In-memory trade history for summaries (to avoid database issues)
        self.recent_trades = []  # List of trade dicts with timestamps
        self.trade_history_duration = 604800  # Keep 7 days of trade history (7 * 24 * 60 * 60)
        
        # Token safety tracking
        self.token_safety_cache = {}  # Cache safety scores to avoid re-checking
        
        # Execution latency tracking
        self.execution_latencies = []  # Store recent latencies for monitoring
        self.latency_history_duration = 3600  # Keep 1 hour of latency data
        
        # Smart token cache with different TTLs based on processing outcome
        self.processed_tokens = {}  # Now stores {token: {'timestamp': time, 'status': str, 'reason': str}}
        self.token_cache_ttls = {
            'traded': 600,           # Successfully traded: 10 minutes
            'evaluated': 300,        # Fully evaluated but not traded: 5 minutes  
            'no_alpha': 60,          # No alpha signal yet: 1 minute (quick retry)
            'failed_filters': 180,   # Failed basic filters: 3 minutes
            'rate_limited': 30,      # Rate limited during check: 30 seconds
            'error': 120             # Error during processing: 2 minutes
        }
        self.last_token_cleanup = time.time()

    def _load_config(self, config_path: str) -> TradingConfig:
        # Use shared config loader
        config_data = load_config(config_path)
        
        # Validate required configuration
        validate_required_keys(config_data)
        
        # Extract moralis keys (already normalized by config_loader)
        moralis_keys = config_data.get('moralis_keys', [config_data.get('moralis_key', '')])
        
        # Extract bitquery tokens (already normalized by config_loader)
        bitquery_tokens = config_data.get('bitquery_tokens', [config_data.get('bitquery_token', '')])
        
        return TradingConfig(
            rpc_endpoint=config_data['rpc_endpoint'],
            bitquery_token=bitquery_tokens,  # Now passes list of tokens
            moralis_key=moralis_keys,  # Now passes list of keys
            quicknode_endpoint=config_data['quicknode_endpoint'],
            quicknode_api_key=config_data['quicknode_api_key'],
            watched_wallets=config_data['watched_wallets'],
            threshold_alpha_buys=config_data['threshold_alpha_buys'],
            min_liquidity_usd=config_data['min_liquidity_usd'],
            time_window_sec=config_data['time_window_sec'],
            max_trade_pct=config_data['max_trade_pct'],
            tp_multiplier=config_data['tp_multiplier'],
            stop_loss_pct=config_data['stop_loss_pct'],
            paper_mode=config_data.get('trading_mode', 'simulation') == 'simulation',
            initial_capital=config_data['trading']['initial_capital'],
            max_trades_per_day=config_data['trading']['max_trades_per_day'],
            min_time_between_trades=config_data['trading']['min_time_between_trades'],
            max_concurrent_positions=config_data['trading'].get('max_concurrent_positions', 3),
            pump_fun_program=config_data['pump_fun']['program_address'],
            pump_fun_api=config_data['pump_fun']['api_endpoint'],
            logging_level=config_data['logging']['level'],
            logging_file=config_data['logging']['file'],
            database_file=config_data['database']['file'],
            notifications=config_data.get('notifications', {}),  # Add notifications section
            alpha_weight_min=config_data.get('alpha_weight_min', 3.5),  # Add alpha weight threshold
            alpha_enhanced=config_data.get('alpha_enhanced', {}),  # Add alpha enhanced section
            risk_management=config_data.get('risk_management', {'enabled': True}),  # Add risk management section
            min_confidence=config_data.get('min_confidence', 50.0)  # Add minimum confidence threshold
        )

    def _get_realtime_config(self) -> Dict:
        """Get configuration for realtime client"""
        # Load the full config data again
        config_data = load_config('config.yml')
        
        return {
            'bitquery_tokens': config_data.get('bitquery_tokens'),
            'pumpportal': config_data.get('pumpportal', {}),
            'realtime_source': config_data.get('realtime_source', 'pumpportal')
        }

    async def start(self):
        """Start the trading bot"""
        self.logger.info("Starting Memecoin Trading Bot")
        self.logger.info(f"Paper mode: {self.config.paper_mode}")
        self.logger.info(f"Initial capital: ${self.config.initial_capital}")
        
        self.running = True
        
        # Send startup notification to Discord
        if self.trading_engine.notifier:
            await self.trading_engine.notifier.send_text(
                f"🚀 **Memecoin Trading Bot Started**\n"
                f"Mode: {'📝 Paper Trading' if self.config.paper_mode else '💸 Live Trading'}\n"
                f"Starting Capital: ${self.config.initial_capital}\n"
                f"Watching {len(self.config.watched_wallets)} alpha wallets"
            )
            
            # Pass discord notifier to rotation manager
            self.wallet_rotation_manager.discord_notifier = self.trading_engine.notifier
        
        # Initialize database
        await self.database.initialize()
        
        # Initialize realtime client
        await self.realtime_client.initialize()
        self.logger.info(f"Realtime client initialized with source: {self.realtime_client.get_source()}")
        
        # Start monitoring tasks
        tasks = [
            self.manage_active_positions(),
            self.daily_reset_task(),
            self.periodic_summary_task(),
            self.heartbeat_task(),
            self.wallet_rotation_manager.start_rotation_loop()
        ]
        
        # Use unified stream for PumpPortal or separate for Bitquery
        realtime_source = getattr(self.config, 'realtime_source', 'pumpportal')
        if realtime_source == 'pumpportal':
            tasks.append(self.monitor_pumpportal_events())
        else:
            tasks.append(self.monitor_new_tokens())
        
        await asyncio.gather(*tasks)

    async def monitor_new_tokens(self):
        """Monitor for new token launches via Bitquery"""
        self.logger.info("Starting token launch monitoring")
        
        async for token_event in self.realtime_client.subscribe_token_launches():
            if not self.running:
                break
                
            try:
                await self.process_new_token(token_event)
            except Exception as e:
                self.logger.error(f"Error processing token {token_event.get('mint', 'unknown')}: {e}")

    async def monitor_pumpportal_events(self):
        """Monitor unified PumpPortal stream for both token launches and trades with auto-reconnect"""
        if self._pumpportal_monitoring:
            self.logger.warning("PumpPortal monitoring already running, skipping")
            return
            
        self._pumpportal_monitoring = True
        self.logger.info("Starting unified PumpPortal monitoring (tokens + trades)")
        
        try:
            while self.running:  # Add reconnection loop
                try:
                    # Check if PumpPortal client exists
                    if not self.realtime_client.pumpportal_client:
                        self.logger.error("PumpPortal client not available!")
                        await asyncio.sleep(60)  # Wait before retry
                        continue
                    
                    self.logger.info("Subscribing to PumpPortal all events stream...")
                    
                    event_count = 0
                    # Use the unified stream from PumpPortal, passing our CURRENT watched wallets
                    # This ensures we always use the latest wallet list after rotation
                    watched_wallets = list(self.wallet_tracker.watched_wallets)
                    self.logger.info(f"Using {len(watched_wallets)} current watched wallets for PumpPortal subscription")
                    async for event in self.realtime_client.pumpportal_client.subscribe_all_events(watched_wallets):
                        event_count += 1
                        if event_count <= 3:  # Log first few events
                            self.logger.info(f"Received PumpPortal event #{event_count}: {event.get('event_type', 'unknown')}")
                        if not self.running:
                            break
                            
                        try:
                            event_type = event.get('event_type')
                            
                            if event_type == 'token_launch':
                                # Process as new token
                                mint = event.get('mint', 'unknown')
                                self.logger.debug(f"Processing token launch: {mint[:8]}...")
                                await self.process_new_token(event)
                                
                            elif event_type == 'trade':
                                # Process as trade for alpha detection
                                mint = event.get('mint')
                                trader = event.get('buyer') or event.get('seller')
                                is_buy = event.get('buyer') is not None
                                timestamp = event.get('timestamp')
                                
                                if mint and trader and is_buy:
                                    # Check if this is from an alpha wallet
                                    if trader in self.wallet_tracker.watched_wallets:
                                        # Record this alpha wallet buy for real-time detection
                                        self.wallet_tracker.record_realtime_alpha_buy(trader, mint, timestamp)
                                        self.logger.info(f"REALTIME ALPHA: {trader[:8]}... bought {mint[:8]}... (via PumpPortal)")
                                        
                                        # Check if we have enough alpha signals to process this token
                                        current_alpha_buyers = self.wallet_tracker.get_realtime_alpha_buyers(
                                            mint, 
                                            self.config.time_window_sec
                                        )
                                        
                                        # If we hit threshold from real-time data alone, process the token
                                        if len(current_alpha_buyers) >= self.config.threshold_alpha_buys:
                                            if mint not in self.processed_tokens:
                                                self.logger.info(f"Alpha threshold reached for {mint[:8]}... from real-time data, processing token")
                                                # Create a token event for processing
                                                token_event = {
                                                    'mint': mint,
                                                    'deployer': event.get('deployer', 'unknown'),
                                                    'name': event.get('name', ''),
                                                    'symbol': event.get('symbol', ''),
                                                    'timestamp': timestamp
                                                }
                                                await self.process_new_token(token_event)
                                    else:
                                        # Log non-alpha trades for debugging (limited)
                                        if event_count <= 10:
                                            self.logger.debug(f"Non-alpha trade: {trader[:8]}... -> {mint[:8]}...")
                            
                        except Exception as e:
                            self.logger.error(f"Error processing PumpPortal event: {e}")
                    
                except (ConnectionError, OSError, Exception) as e:
                    self.logger.error(f"PumpPortal connection error: {e}")
                    self.logger.info("Will attempt to reconnect in 30 seconds...")
                    await asyncio.sleep(30)  # Wait before reconnection attempt
                    continue
                
        finally:
            self._pumpportal_monitoring = False
            
        self.logger.info("PumpPortal monitoring stopped")

    async def process_new_token(self, token_event: Dict):
        """Process a newly launched token"""
        # Start latency tracking
        token_detected_time = time.time()
        
        mint_address = token_event['mint']
        deployer = token_event['deployer']
        
        # Clean up old processed tokens periodically
        self._cleanup_processed_tokens()
        
        # Check if token is in cache and if it should be skipped
        if self._should_skip_token(mint_address):
            return
        
        # Only log tokens we're seriously considering (reduces noise)
        self.logger.debug(f"New token detected: {mint_address}")
        self.tokens_processed += 1
        
        # Check trading limits
        if not self._can_trade():
            self.logger.debug("Trading limits reached, skipping")
            return
        
        # NEW STRATEGY: Check alpha activity FIRST from real-time cache
        # This avoids calling Moralis for every token
        initial_alpha_buyers = self.wallet_tracker.get_realtime_alpha_buyers(
            mint_address, 
            self.config.time_window_sec
        )
        
        # If no alpha wallets are buying from real-time data, skip expensive Moralis calls
        if len(initial_alpha_buyers) == 0:
            self.logger.debug(f"No alpha activity detected for {mint_address[:8]}... in real-time cache, skipping Moralis checks")
            self._record_token_status(mint_address, 'no_alpha', 'No initial alpha signal')
            return
        
        self.logger.info(f"Initial alpha signal: {len(initial_alpha_buyers)} wallets detected for {mint_address[:8]}... from real-time data")
        
        # Token maturity check - wait a bit for Moralis to index the token
        token_age = time.time() - token_detected_time
        min_token_age = getattr(self.config, 'min_token_age_seconds', 10)  # Default 10 seconds
        
        if token_age < min_token_age:
            wait_time = min_token_age - token_age
            self.logger.info(f"Token {mint_address[:8]}... is too new ({token_age:.1f}s), waiting {wait_time:.1f}s for Moralis indexing")
            await asyncio.sleep(wait_time)
        
        # Now check Moralis, but be more lenient with liquidity for alpha-signaled tokens
        try:
            # If Moralis is rate limited, skip this token entirely (safer approach)
            if self.moralis.rate_limited:
                self.logger.debug(f"Moralis rate limited - skipping {mint_address[:8]}...")
                self._record_token_status(mint_address, 'rate_limited', 'Moralis API rate limited')
                return
                
            liquidity = await self.moralis.get_token_liquidity(mint_address)
            
            # For alpha-signaled tokens, we can accept zero liquidity (new tokens)
            # The alpha signal is our primary filter now
            if not liquidity:
                self.logger.info(f"No liquidity data yet for alpha-signaled token {mint_address[:8]}... - proceeding with alpha signal alone")
                liquidity = {'total_liquidity_usd': 0, 'pools': [], 'pool_count': 0}
            
            # Try to get metadata, but don't block on it for new tokens
            metadata = await self.moralis.get_token_metadata(mint_address)
            if not metadata:
                self.logger.debug(f"No metadata yet for {mint_address[:8]}... (proceeding with alpha signal)")
                metadata = {}  # Empty dict for safe access
            
            # Modified filter check - more lenient for alpha-signaled tokens
            if not self._passes_filters_with_alpha(metadata, liquidity, deployer, len(initial_alpha_buyers)):
                # Only log failures at debug level to reduce noise
                self.logger.debug(f"Token {mint_address} failed filters despite alpha signal")
                self._record_token_status(mint_address, 'failed_filters', 'Failed safety/liquidity filters')
                return
            
            # Get wallet status for enhanced logging
            active_wallets = self.wallet_tracker.get_active_wallets()
            total_wallets = len(self.wallet_tracker.watched_wallets)
            
            # If we get here, token has alpha signal - now it's worth detailed checking
            metadata_status = "with metadata" if metadata else "metadata-pending"
            liquidity_status = f"${liquidity.get('total_liquidity_usd', 0):,.0f}" if liquidity.get('total_liquidity_usd', 0) > 0 else "pending"
            self.logger.info(f"Alpha-signaled token {mint_address[:8]}... proceeding to detailed check (liquidity: {liquidity_status}, {metadata_status})")
                
        except Exception as e:
            # Only log 404s at debug level since they're common for new tokens
            if "404" in str(e) or "not found" in str(e).lower():
                self.logger.debug(f"Token {mint_address} metadata not available (likely too new)")
                self._record_token_status(mint_address, 'no_alpha', 'Metadata not ready')  # Allow quick retry
            else:
                self.logger.error(f"Failed to get token data for {mint_address}: {e}")
                self._record_token_status(mint_address, 'error', str(e))
            return
        
        # Track alpha wallet activity with enhanced analysis
        self.alpha_checks_performed += 1
        alpha_check_start_time = time.time()
        
        alpha_analysis = await self.wallet_tracker.check_alpha_activity_detailed(
            mint_address, 
            self.config.time_window_sec,
            self.moralis,
            self.config.threshold_alpha_buys
        )
        
        alpha_check_end_time = time.time()
        alpha_check_duration = alpha_check_end_time - alpha_check_start_time
        
        # Check if we should execute trade based on confidence and thresholds
        confidence_score = alpha_analysis['confidence_score']
        investment_multiplier = alpha_analysis['investment_multiplier']
        alpha_wallets = alpha_analysis['alpha_wallets']
        wallet_tiers = alpha_analysis['wallet_tiers']
        
        # Enhanced logging with timing and deduplication info
        if len(alpha_wallets) > 0:
            tier_summary = ', '.join([f"{w[:8]}({t})" for w, t in wallet_tiers.items()])
            self.logger.debug(f"Alpha check completed in {alpha_check_duration:.1f}s: {len(alpha_wallets)}/{self.config.threshold_alpha_buys} wallets detected, confidence: {confidence_score:.1f}, wallets: [{tier_summary}]")
        else:
            self.logger.debug(f"Alpha check completed in {alpha_check_duration:.1f}s: no alpha wallets found for {mint_address[:8]}...")
        
        # Use configurable confidence requirements
        min_confidence = self.config.min_confidence
        
        # Ensure we have distinct wallets (not just one wallet buying multiple times)
        distinct_wallets = len(set(alpha_wallets))
        min_distinct_wallets = self.config.threshold_alpha_buys  # Use configured value directly
        
        # Phase 4.2: Check adaptive risk management before proceeding
        can_trade, block_reason = self.risk_manager.can_trade()
        if not can_trade:
            self.logger.info(f"Trade blocked by risk management: {block_reason}")
            return
        
        # Get current risk-adjusted parameters
        if self.config.risk_management.get('enabled', True):
            risk_params = self.risk_manager.get_trading_params()
        else:
            # Use aggressive config values when risk management is disabled
            risk_params = {
                'min_weight': self.config.alpha_weight_min,
                'min_wallets': self.config.alpha_enhanced.get('min_independent_wallets', 1),
                'risk_level': 'aggressive'
            }
        
        # Use weighted voting from Phase 2 (check meets_threshold from alpha_analysis)
        weighted_threshold_passed = alpha_analysis.get('meets_threshold', False)
        total_weight = alpha_analysis.get('total_weight', 0)
        
        # Additional quality checks with dynamic thresholds
        signal_quality_passed = (
            weighted_threshold_passed and
            confidence_score >= min_confidence and 
            distinct_wallets >= risk_params['min_wallets'] and  # Dynamic based on P&L
            total_weight >= risk_params['min_weight']  # Dynamic weighted voting threshold
        )
        
        if signal_quality_passed:
            # NEW: Check entry timing - reject if too late after first alpha buy
            if 'alpha_wallets' in alpha_analysis and alpha_wallets:
                # Get timestamps of alpha buys from realtime cache
                buy_timestamps = []
                for wallet in alpha_wallets:
                    if mint_address in self.wallet_tracker.realtime_trades_cache:
                        for trade_wallet, trade_time in self.wallet_tracker.realtime_trades_cache[mint_address]:
                            if trade_wallet == wallet:
                                buy_timestamps.append(trade_time)
                                break
                
                if buy_timestamps:
                    first_alpha_buy = min(buy_timestamps)
                    time_since_first_alpha = time.time() - first_alpha_buy
                    max_entry_delay = getattr(self.config, 'max_entry_delay_seconds', 180)  # 3 minutes default
                    
                    if time_since_first_alpha > max_entry_delay:
                        self.logger.warning(f"ENTRY TOO LATE: {time_since_first_alpha:.0f}s after first alpha "
                                          f"(max: {max_entry_delay}s) for {mint_address[:8]}...")
                        self._record_token_status(mint_address, 'failed_filters', f'Entry too late: {time_since_first_alpha:.0f}s')
                        return  # Skip this trade - momentum likely gone
            
            # Check token safety before trading (reuse swap data from alpha check)
            alpha_swaps_data = alpha_analysis.get('last_swaps_data', [])
            safety_check = await self.check_token_safety(mint_address, metadata, liquidity, alpha_swaps_data)
            
            if not safety_check['safe']:
                rug_score = safety_check['rug_score']
                warnings = safety_check['warnings']
                self.logger.warning(f"Token {mint_address[:8]} failed safety check: "
                                  f"Score: {rug_score}, Warnings: {warnings}")
                self._record_token_status(mint_address, 'failed_filters', f'Safety score: {rug_score}')
                return  # Skip this trade
            
            # NEW: Hard block on price extension
            # The extension check is done in check_token_safety and adds to rug_score
            # We need to explicitly check the extension_guard result
            if alpha_swaps_data and len(alpha_swaps_data) > 0:
                current_price = None
                recent_prices = [swap.get('price', 0) for swap in alpha_swaps_data[-5:] if swap.get('price', 0) > 0]
                if recent_prices:
                    current_price = recent_prices[-1]
                
                extension_result = self.safety_checker.check_price_extension(
                    mint_address, alpha_swaps_data, current_price
                )
                
                if extension_result.get('is_extended', False):
                    percentile = extension_result.get('percentile_rank', 0) * 100
                    self.logger.warning(f"PRICE EXTENSION BLOCK: Price at {percentile:.0f}th percentile "
                                      f"(>{getattr(self.config, 'price_extension_percentile', 90)}th) for {mint_address[:8]}...")
                    self._record_token_status(mint_address, 'failed_filters', f'Price extended: {percentile:.0f}th percentile')
                    return  # Hard block - don't buy at peaks
            
            # ENHANCED: Comprehensive trade decision logging with all new checks
            tier_summary = ', '.join([f"{addr[:8]}({tier})" for addr, tier in wallet_tiers.items()])
            wallet_details = [f"{addr[:8]}(tier={tier})" for addr, tier in wallet_tiers.items()]
            
            # Show weighted voting details
            wallet_weights = alpha_analysis.get('wallet_weights', {})
            weight_details = [f"{addr[:8]}({weight:.2f})" for addr, weight in wallet_weights.items()]
            
            # Get timing information for logging
            entry_timing_info = ""
            if buy_timestamps:
                time_since_first = time.time() - min(buy_timestamps)
                time_spread = max(buy_timestamps) - min(buy_timestamps) if len(buy_timestamps) > 1 else 0
                entry_timing_info = f"entry_delay={time_since_first:.0f}s spread={time_spread:.0f}s"
            
            # Get price extension info
            extension_info = ""
            if 'extension_result' in locals():
                percentile = extension_result.get('percentile_rank', 0) * 100
                extension_info = f"price_percentile={percentile:.0f}th"
            
            self.logger.info(f"✅ TRADE APPROVED: mint={mint_address[:8]}... "
                           f"weight={total_weight:.2f} confidence={confidence_score:.1f} "
                           f"risk_level={risk_params['risk_level']} wallets=[{', '.join(weight_details)}] "
                           f"safety={safety_check['rug_score']} mult={investment_multiplier:.1f}x "
                           f"distinct={distinct_wallets} liquidity=${liquidity.get('usd', 0):,.0f} "
                           f"{entry_timing_info} {extension_info}")
            
            # Track execution latency
            trade_executed_time = time.time()
            total_latency = trade_executed_time - token_detected_time
            alpha_latency = alpha_check_end_time - alpha_check_start_time
            
            # Record latency metrics
            latency_record = {
                'total_latency': total_latency,
                'alpha_check_latency': alpha_latency,
                'token': mint_address[:8],
                'timestamp': trade_executed_time
            }
            
            self.execution_latencies.append(latency_record)
            self._cleanup_old_latencies()
            
            # Warn if latency is too high
            if total_latency > 3.0:  # More than 3 seconds
                self.logger.warning(f"High execution latency: {total_latency:.1f}s "
                                  f"(alpha: {alpha_latency:.1f}s) for {mint_address[:8]}")
            
            # Record that we're following these wallets' trades
            self.wallet_tracker.record_trade_follow(alpha_wallets, mint_address, confidence_score)
            
            # Mark as evaluated (will be upgraded to 'traded' if trade executes)
            self._record_token_status(mint_address, 'evaluated', f'Confidence: {confidence_score:.1f}')
            
            await self.execute_trade(mint_address, metadata, liquidity, 
                                   confidence_score, investment_multiplier, wallet_tiers)
        else:
            # ENHANCED: Detailed rejection logging with all failure points
            rejection_reasons = []
            if not weighted_threshold_passed:
                rejection_reasons.append(f"weighted_threshold_failed(weight={total_weight:.2f}<{risk_params['min_weight']})")
            if confidence_score < min_confidence:
                rejection_reasons.append(f"low_confidence({confidence_score:.1f}<{min_confidence})")
            if distinct_wallets < risk_params['min_wallets']:
                rejection_reasons.append(f"insufficient_wallets({distinct_wallets}<{risk_params['min_wallets']})")
            if total_weight < risk_params['min_weight']:
                rejection_reasons.append(f"low_weight({total_weight:.2f}<{risk_params['min_weight']})")
            
            # Check for specific failure reasons from alpha analysis
            if alpha_analysis.get('temporal_clustering_failure'):
                time_spread = alpha_analysis.get('time_spread', 0)
                rejection_reasons.append(f"temporal_clustering({time_spread:.0f}s)")
            if alpha_analysis.get('independence_failure'):
                rejection_reasons.append("insufficient_independent_wallets")
            
            self.logger.info(f"❌ TRADE REJECTED: mint={mint_address[:8]}... "
                           f"reasons=[{', '.join(rejection_reasons)}] "
                           f"confidence={confidence_score:.1f} wallets={len(alpha_wallets)} "
                           f"distinct={distinct_wallets} weight={total_weight:.2f}")
            
            # Record as evaluated but not traded
            self._record_token_status(mint_address, 'evaluated', f"Rejected: {', '.join(rejection_reasons)}")

    def _extract_liquidity_usd(self, liquidity: Dict) -> float:
        """Extract liquidity USD value from various possible field names"""
        return (liquidity.get('total_liquidity_usd') or
                liquidity.get('liquidity_usd') or
                liquidity.get('usd_value') or 0.0)
    
    def _can_trade(self) -> bool:
        """Check if we can execute a new trade with improved throttling"""
        # Check daily trade limit
        if self.trades_today >= self.config.max_trades_per_day:
            self.logger.debug(f"Daily trade limit reached: {self.trades_today}")
            return False
        
        # Check minimum time between trades
        if self.last_trade_time:
            time_since_last = time.time() - self.last_trade_time
            # Use configured value directly - no hardcoded minimum for aggressive trading
            min_interval = self.config.min_time_between_trades
            if time_since_last < min_interval:
                self.logger.debug(f"Too soon since last trade: {time_since_last:.0f}s < {min_interval}s")
                return False
        
        # Check max concurrent positions
        # Use configured max concurrent positions
        max_concurrent = self.config.max_concurrent_positions
        current_positions = len(self.trading_engine.active_positions)
        if current_positions >= max_concurrent:
            self.logger.debug(f"Max concurrent positions reached: {current_positions}/{max_concurrent}")
            return False
        
        # Cool-down after stop loss or time stop (3 minutes)
        if self.recent_trades:
            last_sells = [t for t in reversed(self.recent_trades) if t.get('action') == 'SELL']
            if last_sells:
                last_sell = last_sells[0]
                exit_reason = last_sell.get('exit_reason', '')
                if exit_reason in ('STOP_LOSS', 'TIME_STOP'):
                    cooldown_time = 180  # 3 minutes
                    time_since_loss = time.time() - last_sell['timestamp']
                    if time_since_loss < cooldown_time:
                        self.logger.debug(f"Cooling down after {exit_reason}: {time_since_loss:.0f}s < {cooldown_time}s")
                        return False
        
        return True

    def _passes_filters(self, metadata: Dict, liquidity: Dict, deployer: str) -> bool:
        """Apply token filters"""
        # Check liquidity requirement (use unified extraction method)
        liquidity_usd = self._extract_liquidity_usd(liquidity)
        if liquidity_usd < self.config.min_liquidity_usd:
            return False
            
        # Skip tokens with suspicious names (basic spam filter) - only if metadata available
        if metadata:  # Only check spam filter if we have metadata
            name = metadata.get('name', '').lower()
            symbol = metadata.get('symbol', '').lower()
            spam_keywords = ['test', 'fake', 'scam', 'rugpull', 'honeypot']
            if any(keyword in name or keyword in symbol for keyword in spam_keywords):
                return False
        
        # Add deployer blacklist check here if needed
        
        return True
    
    def _passes_filters_with_alpha(self, metadata: Dict, liquidity: Dict, deployer: str, alpha_count: int) -> bool:
        """Apply token filters with alpha signal consideration"""
        liquidity_usd = self._extract_liquidity_usd(liquidity)
        
        # For strong alpha signals (3+ wallets), be more lenient with liquidity
        if alpha_count >= 3:
            # Accept zero liquidity for very new tokens with strong alpha signal
            min_liquidity_override = 0
        elif alpha_count >= 2:
            # Reduce liquidity requirement by 50% for moderate alpha signal
            min_liquidity_override = self.config.min_liquidity_usd * 0.5
        else:
            # Use normal liquidity requirement
            min_liquidity_override = self.config.min_liquidity_usd
        
        if liquidity_usd < min_liquidity_override:
            return False
            
        # Skip tokens with suspicious names (basic spam filter) - only if metadata available
        if metadata:  # Only check spam filter if we have metadata
            name = metadata.get('name', '').lower()
            symbol = metadata.get('symbol', '').lower()
            spam_keywords = ['test', 'fake', 'scam', 'rugpull', 'honeypot']
            if any(keyword in name or keyword in symbol for keyword in spam_keywords):
                return False
        
        # Add deployer blacklist check here if needed
        
        return True

    def _cleanup_processed_tokens(self):
        """Remove old processed tokens based on their status-specific TTLs"""
        current_time = time.time()
        
        # Only cleanup every 60 seconds to avoid overhead
        if current_time - self.last_token_cleanup < 60:
            return
            
        # Remove tokens based on their status-specific TTL
        expired_tokens = []
        for token, cache_data in list(self.processed_tokens.items()):
            status = cache_data.get('status', 'error')
            timestamp = cache_data.get('timestamp', 0)
            ttl = self.token_cache_ttls.get(status, 300)  # Default 5 min
            
            if current_time - timestamp > ttl:
                expired_tokens.append(token)
                del self.processed_tokens[token]
            
        if expired_tokens:
            # Log summary by status
            status_counts = {}
            for token in expired_tokens:
                if token in self.processed_tokens:
                    status = self.processed_tokens[token].get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
            
            self.logger.debug(f"Cleaned up {len(expired_tokens)} tokens: {status_counts}")
        
        # Check for inactive alpha wallets while we're doing cleanup
        newly_inactive = self.wallet_tracker.check_inactive_wallets()
        if newly_inactive:
            self.logger.info(f"Marked {len(newly_inactive)} wallets as inactive: {[w[:8]+'...' for w in newly_inactive[:3]]}")
            
        self.last_token_cleanup = current_time
    
    def _should_skip_token(self, mint_address: str) -> bool:
        """Check if token should be skipped based on smart cache"""
        if mint_address not in self.processed_tokens:
            return False
            
        cache_data = self.processed_tokens[mint_address]
        status = cache_data.get('status', 'error')
        timestamp = cache_data.get('timestamp', 0)
        ttl = self.token_cache_ttls.get(status, 300)
        
        current_time = time.time()
        if current_time - timestamp > ttl:
            # Cache expired, allow reprocessing
            del self.processed_tokens[mint_address]
            return False
            
        # Still in cache, skip
        time_left = ttl - (current_time - timestamp)
        self.logger.debug(f"Token {mint_address[:8]}... in cache (status={status}, {time_left:.0f}s left)")
        return True
    
    def _record_token_status(self, mint_address: str, status: str, reason: str = ''):
        """Record token processing status for smart caching"""
        self.processed_tokens[mint_address] = {
            'timestamp': time.time(),
            'status': status,
            'reason': reason
        }
        self.logger.debug(f"Token {mint_address[:8]}... cached as {status}: {reason}")
    
    def _cleanup_old_trades(self):
        """Remove trades older than trade_history_duration from memory"""
        current_time = time.time()
        cutoff_time = current_time - self.trade_history_duration
        
        # Keep only trades within the retention period
        self.recent_trades = [
            trade for trade in self.recent_trades 
            if trade.get('timestamp', 0) > cutoff_time
        ]
    
    def _cleanup_old_latencies(self):
        """Remove old latency records from memory"""
        current_time = time.time()
        cutoff_time = current_time - self.latency_history_duration
        
        # Keep only latencies within the retention period
        self.execution_latencies = [
            latency for latency in self.execution_latencies 
            if latency.get('timestamp', 0) > cutoff_time
        ]
    
    async def check_token_safety(self, mint_address: str, metadata: Dict, liquidity: Dict, cached_swaps: List = None) -> Dict:
        """
        Enhanced token safety check using stream-based safety checker (Phase 4.1)
        Returns: {'safe': bool, 'rug_score': 0-100, 'warnings': []}
        """
        # Check cache first
        if mint_address in self.token_safety_cache:
            cache_entry = self.token_safety_cache[mint_address]
            if time.time() - cache_entry['timestamp'] < 300:  # 5 minute cache
                return cache_entry['result']
        
        warnings = []
        rug_score = 0  # 0 = safe, 100 = definite rug
        
        # Use enhanced safety checker if we have swap data
        if cached_swaps and len(cached_swaps) > 0:
            # Calculate our order size
            order_size = self.config.initial_capital * self.config.max_trade_pct
            
            # Get current price from recent trades for extension guard
            current_price = None
            if cached_swaps:
                recent_prices = [swap.get('price', 0) for swap in cached_swaps[-5:] if swap.get('price', 0) > 0]
                current_price = recent_prices[-1] if recent_prices else None
            
            # Perform comprehensive enhanced safety check
            safety_result = self.safety_checker.check_token_safety(
                mint_address, 
                order_size,
                cached_swaps,
                max_impact=getattr(self.config, 'safety', {}).get('max_price_impact', 0.008),
                current_price=current_price
            )
            
            # Convert to legacy format for compatibility
            if not safety_result['safe_to_trade']:
                rug_score += 50  # Major penalty for failing safety checks
                warnings.extend(safety_result['warnings'])
            
            # Add price impact to rug score
            impact_penalty = min(safety_result['price_impact'] * 1000, 30)  # Max 30 points
            rug_score += impact_penalty
            
            # Add extension guard penalty
            if safety_result.get('extension_guard', {}).get('is_extended', False):
                rug_score += 25  # Penalty for buying at price extension
                
            # Log enhanced safety details
            sellability = safety_result.get('sellability', {})
            extension = safety_result.get('extension_guard', {})
            self.logger.debug(f"Enhanced safety for {mint_address[:8]}...: "
                            f"Sellers: {sellability.get('unique_sellers', 0)}, "
                            f"Extension: {extension.get('percentile_rank', 0):.1%}, "
                            f"Impact: {safety_result['price_impact']:.2%}")
        
        try:
            # Check liquidity amount (basic check)
            liquidity_usd = self._extract_liquidity_usd(liquidity)
            if liquidity_usd < 1000:  # Less than $1k liquidity
                rug_score += 30
                warnings.append("Low liquidity (<$1k)")
            elif liquidity_usd < 5000:  # Less than $5k liquidity
                rug_score += 15
                warnings.append("Low liquidity (<$5k)")
            
            # Removed expensive holders API call from hot path
            # Instead, we use unique buyers from swap data we already fetched (free)
            if cached_swaps:
                # Count unique buyers from cached swap data
                unique_traders = set()
                for swap in cached_swaps:
                    if swap.get('wallet'):
                        unique_traders.add(swap.get('wallet'))
                
                trader_count = len(unique_traders)
                if trader_count < 20:  # Very few unique traders
                    rug_score += 20
                    warnings.append(f"Few unique traders ({trader_count})")
                elif trader_count < 50:  # Limited trader diversity
                    rug_score += 10
                    warnings.append(f"Limited trader diversity ({trader_count})")
                
                self.logger.debug(f"Used cached swap data to assess trader diversity: {trader_count} unique traders")
            else:
                # No penalty if we don't have swap data - avoid expensive API call
                self.logger.debug("No cached swap data available for trader diversity check")
            
            # TODO: Add LP lock and mint authority checks when available
            # Currently not available from Moralis API - would need:
            # - LP lock status (is liquidity locked?)
            # - Mint authority status (has mint been revoked?)
            # These are critical safety checks for pump.fun tokens
            
            # Check recent transaction patterns for honeypot behavior
            try:
                # Use cached swaps from alpha check if available, otherwise fetch new data
                if cached_swaps:
                    swaps = cached_swaps
                    self.logger.debug(f"Using cached swap data for safety check on {mint_address[:8]}... (API call saved)")
                else:
                    swaps = await self.moralis.get_token_swaps(mint_address, limit=50)
                
                buy_count = sum(1 for s in swaps if s.get('to_token') == mint_address)
                sell_count = sum(1 for s in swaps if s.get('from_token') == mint_address)
                
                if buy_count > 0 and sell_count == 0:
                    rug_score += 50
                    warnings.append("No sells detected (possible honeypot)")
                elif sell_count < buy_count * 0.1:  # Less than 10% sells vs buys
                    rug_score += 30
                    warnings.append("Very few sells vs buys (honeypot risk)")
                
            except Exception as e:
                self.logger.debug(f"Could not check swaps for {mint_address}: {e}")
            
            # Calculate final safety
            is_safe = rug_score < 70  # Conservative threshold
            
            result = {
                'safe': is_safe,
                'rug_score': min(rug_score, 100),
                'warnings': warnings
            }
            
            # Cache result
            self.token_safety_cache[mint_address] = {
                'result': result,
                'timestamp': time.time()
            }
            
            self.logger.info(f"Token safety check for {mint_address[:8]}: "
                           f"Score: {rug_score}, Safe: {is_safe}, Warnings: {len(warnings)}")
            
            return result
        except Exception as e:
            self.logger.error(f"Error checking token safety for {mint_address}: {e}")
            return {'safe': False, 'rug_score': 100, 'warnings': ['Safety check failed']}
            
        except Exception as e:
            self.logger.error(f"Error checking token safety for {mint_address}: {e}")
            return {'safe': False, 'rug_score': 100, 'warnings': ['Safety check failed']}

    async def execute_trade(self, mint_address: str, metadata: Dict, liquidity: Dict, 
                          confidence_score: float = 50, investment_multiplier: float = 1.0, 
                          wallet_tiers: Dict = None):
        """Execute a trade on the token with scaled investment based on alpha wallet quality"""
        # Calculate base trade amount
        base_trade_amount = self.current_capital * self.config.max_trade_pct
        
        # Scale investment based on alpha wallet tiers
        trade_amount = base_trade_amount * investment_multiplier
        
        self.logger.info(f"Executing trade on {mint_address}, amount: ${trade_amount}")
        
        try:
            # Get token symbol from metadata
            symbol = metadata.get('symbol', 'UNKNOWN')
            
            # Execute buy order
            result = await self.trading_engine.buy_token(
                mint_address, 
                trade_amount,
                self.config.paper_mode,
                symbol=symbol,
                confidence_score=confidence_score
            )
            
            if result['success']:
                # Create trade record with enhanced analytics
                trade_record = {
                    'mint': mint_address,
                    'action': 'BUY',
                    'amount': trade_amount,
                    'base_amount': base_trade_amount,
                    'price': result['price'],
                    'sol_amount': result.get('sol_amount', 0),
                    'tokens_amount': result.get('tokens_received', 0),
                    'timestamp': time.time(),  # Use unix timestamp for in-memory
                    'paper_mode': self.config.paper_mode,
                    'confidence_score': confidence_score,
                    'investment_multiplier': investment_multiplier,
                    'wallet_tiers': wallet_tiers or {},
                    'metadata': {
                        'symbol': metadata.get('symbol', 'UNKNOWN'),
                        'name': metadata.get('name', 'Unknown Token')
                    }
                }
                
                # Add to in-memory trade history
                self.recent_trades.append(trade_record)
                self._cleanup_old_trades()  # Clean up old trades from memory
                
                # Try to record in database (but don't fail if it doesn't work)
                try:
                    db_record = trade_record.copy()
                    db_record['timestamp'] = datetime.now()  # Database expects datetime
                    await self.database.record_trade(db_record)
                except Exception as db_error:
                    self.logger.warning(f"Failed to save trade to database: {db_error}")
                    # Continue anyway - we have it in memory
                
                self.trades_today += 1
                self.last_trade_time = time.time()
                
                # Mark token as traded in cache
                self._record_token_status(mint_address, 'traded', f'Invested ${trade_amount:.2f}')
                
                # Start monitoring for exit conditions
                asyncio.create_task(self.monitor_position(mint_address, result, metadata))
                
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")

    async def monitor_position(self, mint_address: str, entry_data: Dict, metadata: Dict = None):
        """Monitor position using sophisticated exit strategy from trading_engine"""
        symbol = metadata.get('symbol', 'UNKNOWN') if metadata else 'UNKNOWN'
        
        # Ensure position exists in trading_engine
        if mint_address not in self.trading_engine.active_positions:
            self.logger.error(f"Position {mint_address} not found in trading_engine!")
            return
            
        position = self.trading_engine.active_positions[mint_address]
        
        self.logger.info(f"🎯 SMART MONITORING {mint_address[:8]}... "
                        f"Entry: ${position.entry_price:.8f}, "
                        f"TP: ${position.tp_price:.8f}, "
                        f"SL: ${position.sl_price:.8f}")
        
        # Monitor until position is fully closed in trading_engine
        while mint_address in self.trading_engine.active_positions:
            try:
                # Get current price and update position state
                current_price = await self.moralis.get_current_price(mint_address, fresh=True)
                
                if current_price <= 0:
                    self.logger.warning(f"Invalid price for {mint_address}, retrying...")
                    await asyncio.sleep(5)
                    continue
                
                # Update peak price and gain tracking in position
                if current_price > position.peak_price:
                    position.peak_price = current_price
                    position.high_gain_peak = max(position.high_gain_peak, 
                                                 ((current_price / position.entry_price) - 1) * 100)
                
                # 🚀 USE SOPHISTICATED EXIT LOGIC
                exit_result = await self.trading_engine.check_exit_conditions(mint_address, current_price)
                
                if exit_result:
                    exit_reason, sell_percentage = exit_result
                    
                    current_gain = ((current_price / position.entry_price) - 1) * 100
                    
                    self.logger.info(f"🎯 EXIT TRIGGERED: {mint_address[:8]}... "
                                   f"Reason: {exit_reason}, "
                                   f"Amount: {sell_percentage*100:.0f}%, "
                                   f"Price: ${current_price:.8f}, "
                                   f"Gain: {current_gain:+.1f}%")
                    
                    # Execute the sophisticated exit
                    sell_result = await self.trading_engine.sell_token(
                        mint_address, 
                        sell_percentage, 
                        self.config.paper_mode,
                        symbol=symbol,
                        exit_reason=exit_reason
                    )
                    
                    if sell_result.get('success'):
                        # Record trade outcome for wallet performance tracking (on full exits)
                        if sell_percentage >= 0.99:  # Full exit
                            profit_pct = sell_result.get('profit_pct', 0)
                            self.wallet_tracker.record_trade_outcome(mint_address, profit_pct)
                            self.logger.info(f"✅ Position CLOSED: {mint_address[:8]}... via {exit_reason}")
                            break
                        else:
                            remaining_tokens = position.amount if mint_address in self.trading_engine.active_positions else 0
                            self.logger.info(f"📊 Partial exit: {sell_percentage*100:.0f}% sold, "
                                          f"Remaining: {remaining_tokens:.0f} tokens")
                    else:
                        self.logger.error(f"❌ Exit failed: {sell_result.get('error')}")
                
                # Log position status every 30 seconds for monitoring
                if int(time.time()) % 30 == 0:
                    hold_time = (datetime.now() - position.entry_time).total_seconds()
                    current_pnl = ((current_price / position.entry_price) - 1) * 100
                    self.logger.debug(f"📊 {mint_address[:8]}... "
                                    f"Hold: {hold_time:.0f}s, "
                                    f"P&L: {current_pnl:+.1f}%, "
                                    f"Peak: {position.high_gain_peak:.1f}%, "
                                    f"TP1: {'✓' if position.tp1_hit_time else '○'}, "
                                    f"BE: {'✓' if position.break_even_armed else '○'}")
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring position {mint_address}: {e}")
                await asyncio.sleep(5)
        
        self.logger.info(f"🏁 Position monitoring ended for {mint_address[:8]}...")
    
    # Note: _execute_partial_exit removed - now using trading_engine.sell_token directly
    # All exit logic is centralized in trading_engine.check_exit_conditions

    async def manage_active_positions(self):
        """Manage all active trading positions"""
        while self.running:
            try:
                # Check for stale positions, update PnL, etc.
                await asyncio.sleep(10)
            except Exception as e:
                self.logger.error(f"Error managing positions: {e}")

    async def daily_reset_task(self):
        """Reset daily counters"""
        while self.running:
            await asyncio.sleep(86400)  # 24 hours
            self.trades_today = 0
            self.logger.info("Daily trade counter reset")
    
    async def diagnostic_check(self):
        """Run diagnostic checks on the realtime connection"""
        self.logger.info("=== DIAGNOSTIC CHECK ===")
        
        try:
            # Get diagnostic info
            diag_info = await self.realtime_client.get_diagnostic_info()
            self.logger.info(f"Realtime client diagnostic: {diag_info}")
            
            # Check connection status
            is_connected = self.realtime_client.is_connected()
            self.logger.info(f"Realtime client connected: {is_connected}")
            
            # Test a simple message receive (with timeout)
            self.logger.info("Testing message reception...")
            message_received = False
            
            try:
                async def test_messages():
                    nonlocal message_received
                    async for token_data in self.realtime_client.subscribe_token_launches():
                        self.logger.info(f"Test message received: {token_data}")
                        message_received = True
                        break
                
                # Run test with 10 second timeout
                await asyncio.wait_for(test_messages(), timeout=10.0)
                
            except asyncio.TimeoutError:
                self.logger.warning("No messages received within 10 seconds - this indicates a subscription issue")
                
                # Try to get more detailed connection info
                if hasattr(self.realtime_client, 'pumpportal_client') and self.realtime_client.pumpportal_client:
                    conn_info = await self.realtime_client.pumpportal_client.get_connection_info()
                    self.logger.info(f"Detailed PumpPortal connection info: {conn_info}")
            
        except Exception as e:
            self.logger.error(f"Diagnostic check failed: {e}")
        
        self.logger.info("=== END DIAGNOSTIC ===")

    async def heartbeat_task(self):
        """Show periodic heartbeat status instead of constant WebSocket spam"""
        last_ws_activity = time.time()
        ws_active = True
        
        # Run diagnostic check after 60 seconds
        diagnostic_run = False
        
        while self.running:
            await asyncio.sleep(30)  # Every 30 seconds
            
            # Check if we're still receiving data
            current_time = time.time()
            if self.tokens_processed > 0 or self.alpha_checks_performed > 0:
                last_ws_activity = current_time
                ws_active = True
            elif current_time - last_ws_activity > 60:  # No activity for 60 seconds
                ws_active = False
            
            # Run diagnostic check after 60 seconds of no activity
            if not diagnostic_run and current_time - last_ws_activity > 60:
                self.logger.warning("No activity detected for 60 seconds - running diagnostic check...")
                await self.diagnostic_check()
                diagnostic_run = True
            
            # Only show heartbeat if no other activity
            if ws_active and self.tokens_processed == 0:
                self.logger.debug("WebSocket active - monitoring for new tokens...")
            elif not ws_active:
                self.logger.warning("No WebSocket activity detected - connection may be inactive")
    
    async def periodic_summary_task(self):
        """Log periodic summary to show the bot is working"""
        last_tokens_processed = 0
        last_alpha_checks = 0
        
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Calculate activity since last summary
            tokens_this_period = self.tokens_processed - last_tokens_processed
            alpha_checks_this_period = self.alpha_checks_performed - last_alpha_checks
            
            # Get recent trades from the last 5 minutes from in-memory storage
            try:
                # Filter trades from last 5 minutes
                five_min_ago = time.time() - 300
                period_trades = [
                    trade for trade in self.recent_trades
                    if trade.get('timestamp', 0) > five_min_ago
                ]
                
                # Prepare enhanced trade summary
                trade_summary = ""
                if period_trades:
                    buy_trades = [t for t in period_trades if t.get('action') == 'BUY']
                    sell_trades = [t for t in period_trades if t.get('action') == 'SELL']
                    
                    total_bought = sum(t.get('sol_amount', 0) for t in buy_trades)
                    total_sold = sum(t.get('sol_amount', 0) for t in sell_trades)
                    profits = sum(t.get('profit', 0) for t in period_trades if t.get('profit') is not None)
                    
                    # Calculate enhanced metrics
                    avg_confidence = sum(t.get('confidence_score', 0) for t in buy_trades) / max(len(buy_trades), 1)
                    avg_multiplier = sum(t.get('investment_multiplier', 1.0) for t in buy_trades) / max(len(buy_trades), 1)
                    
                    # Count wallet tiers involved
                    tier_counts = {'S': 0, 'A': 0, 'B': 0, 'C': 0}
                    for trade in buy_trades:
                        for tier in trade.get('wallet_tiers', {}).values():
                            tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    
                    trade_details = []
                    for trade in period_trades[:3]:  # Show up to 3 most recent trades
                        # Get symbol from metadata or directly from trade
                        metadata = trade.get('metadata', {})
                        if isinstance(metadata, str):
                            try:
                                import json
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                        token_name = metadata.get('symbol') or trade.get('symbol', 'Unknown')
                        trade_type = trade.get('action', 'unknown').lower()
                        amount = trade.get('sol_amount', 0)
                        profit = trade.get('profit', None)
                        
                        if trade_type == 'buy':
                            usd_amount = trade.get('amount', 0)
                            confidence = trade.get('confidence_score', 0)
                            multiplier = trade.get('investment_multiplier', 1.0)
                            tiers = list(trade.get('wallet_tiers', {}).values())
                            tier_str = f"[{','.join(tiers)}]" if tiers else ""
                            trade_details.append(f"  • Bought {token_name} for ${usd_amount:.2f} "
                                               f"(conf: {confidence:.0f}, {multiplier:.1f}x) {tier_str}")
                        elif trade_type == 'sell':
                            usd_amount = trade.get('amount', 0)
                            profit_str = f" ({profit:+.2f}% profit)" if profit is not None else ""
                            trade_details.append(f"  • Sold {token_name} for ${usd_amount:.2f}{profit_str}")
                    
                    # Enhanced trade summary
                    trade_summary = f"\nTrade Activity:\n" + "\n".join(trade_details)
                    
                    if len(buy_trades) > 0:
                        tier_summary = ", ".join([f"{tier}:{count}" for tier, count in tier_counts.items() if count > 0])
                        trade_summary += f"\n  Avg Confidence: {avg_confidence:.0f}, Avg Multiplier: {avg_multiplier:.1f}x"
                        trade_summary += f"\n  Alpha Tiers: {tier_summary}"
                    
                    if profits != 0:
                        trade_summary += f"\n  Period P&L: {profits:+.2f}%"
                
                # Add latency information
                recent_latencies = [l for l in self.execution_latencies if l['timestamp'] > five_min_ago]
                if recent_latencies:
                    avg_total_latency = sum(l['total_latency'] for l in recent_latencies) / len(recent_latencies)
                    avg_alpha_latency = sum(l['alpha_check_latency'] for l in recent_latencies) / len(recent_latencies)
                    trade_summary += f"\n  Avg Latency: {avg_total_latency:.1f}s (alpha: {avg_alpha_latency:.1f}s)"
                
                # Add active wallet information and check for recycling needs
                active_wallets = self.wallet_tracker.get_active_wallets()
                inactive_wallets = self.wallet_tracker.get_inactive_wallets()
                total_wallets = len(self.wallet_tracker.watched_wallets)
                
                # Get deduplication stats
                dedup_stats = self.wallet_tracker.get_deduplication_stats()
                
                # Build summary message
                if tokens_this_period > 0 or alpha_checks_this_period > 0 or period_trades:
                    summary = f"5min Summary: {tokens_this_period} tokens scanned, " \
                             f"{alpha_checks_this_period} alpha checks, " \
                             f"{len(period_trades)} trades executed, " \
                             f"${self.current_capital:.2f} capital"
                    
                    # Add wallet status
                    summary += f"\n  Alpha Wallets: {len(active_wallets)}/{total_wallets} active, {len(inactive_wallets)} inactive"
                    
                    # ENHANCED: Add detailed wallet performance summary
                    perf_summary = self.wallet_tracker.get_performance_summary()
                    if perf_summary['total_trades'] > 0:
                        summary += f"\n  Performance: {perf_summary['overall_win_rate']:.1%} win rate, " \
                                  f"{perf_summary['total_trades']} total trades tracked"
                        
                        # Add tier performance breakdown
                        try:
                            tier_stats = self.wallet_tracker.get_tier_performance_stats()
                            tier_summary = []
                            for tier in ['S', 'A', 'B', 'C']:
                                stats = tier_stats.get(tier, {})
                                if stats.get('count', 0) > 0:
                                    tier_summary.append(f"{tier}:{stats['count']}w({stats['win_rate']:.0%})")
                            if tier_summary:
                                summary += f"\n  Wallet Tiers: {', '.join(tier_summary)}"
                        except Exception as e:
                            self.logger.debug(f"Error getting tier stats: {e}")
                    
                    # Check if rotation is due soon
                    rotation_status = self.wallet_rotation_manager.get_rotation_status()
                    time_until_rotation = rotation_status['time_until_next_rotation']
                    if time_until_rotation < 600:  # Less than 10 minutes
                        summary += f"\n  🔄 Wallet rotation in {int(time_until_rotation/60)} minutes"
                    
                    # Add deduplication savings if significant
                    if dedup_stats['deduped_checks'] > 0:
                        summary += f"\n  API Savings: {dedup_stats['deduped_checks']} duplicate calls avoided ({dedup_stats['savings_pct']:.1f}%)"
                    
                    if trade_summary:
                        summary += trade_summary
                    
                    self.logger.info(summary)
                    
                    # Send Discord summary if there was significant activity
                    if tokens_this_period > 10 or period_trades:
                        await self.trading_engine.send_summary()
                        
                else:
                    self.logger.info(f"Bot running - monitoring pump.fun launches. Alpha wallets: {len(active_wallets)}/{total_wallets} active")
                
                # Check if we need to trigger alpha wallet recycling
                await self._check_wallet_recycling_needs(len(active_wallets))
                    
            except Exception as e:
                self.logger.error(f"Error generating trade summary: {e}")
                # Fallback to simple summary with basic wallet info
                try:
                    active_count = len(self.wallet_tracker.get_active_wallets())
                    total_count = len(self.wallet_tracker.watched_wallets)
                    self.logger.info(f"5min Summary: {tokens_this_period} tokens scanned, "
                                   f"{alpha_checks_this_period} alpha checks, "
                                   f"{self.trades_today} trades today, "
                                   f"${self.current_capital:.2f} capital, "
                                   f"{active_count}/{total_count} wallets active")
                except:
                    self.logger.info(f"5min Summary: {tokens_this_period} tokens scanned, "
                                   f"{alpha_checks_this_period} alpha checks, "
                                   f"{self.trades_today} trades today, "
                                   f"${self.current_capital:.2f} capital")
                
            # Reset deduplication stats after logging
            self.wallet_tracker.reset_dedup_stats()
            
            last_tokens_processed = self.tokens_processed
            last_alpha_checks = self.alpha_checks_performed
    
    async def _check_wallet_recycling_needs(self, active_wallet_count: int):
        """Check if we need to trigger alpha wallet discovery due to low active wallet count"""
        min_active_threshold = 25  # Trigger recycling if we have < 25 active wallets
        critical_threshold = 15    # Critical level for immediate action
        
        if active_wallet_count < critical_threshold:
            self.logger.warning(f"CRITICAL: Only {active_wallet_count} active alpha wallets remaining! Triggering immediate discovery...")
            
            # Send Discord alert for critical wallet shortage
            if self.trading_engine.notifier:
                await self.trading_engine.notifier.send_error_notification(
                    f"CRITICAL: Only {active_wallet_count} active alpha wallets remaining!",
                    {"trigger": "critical_wallet_shortage", "action": "starting_discovery"}
                )
            
            await self._trigger_alpha_discovery("critical")
        elif active_wallet_count < min_active_threshold:
            # Check if we've triggered discovery recently
            current_time = time.time()
            last_discovery = getattr(self, '_last_discovery_trigger', 0)
            
            # Only trigger discovery once every 2 hours to avoid spam
            if current_time - last_discovery > 7200:  # 2 hours
                self.logger.warning(f"Low active wallet count ({active_wallet_count}). Triggering alpha wallet discovery...")
                await self._trigger_alpha_discovery("low_count")
                self._last_discovery_trigger = current_time
            else:
                self.logger.debug(f"Active wallet count low ({active_wallet_count}) but discovery triggered recently")
    
    async def _trigger_alpha_discovery(self, trigger_reason: str):
        """Trigger alpha wallet discovery process"""
        try:
            self.logger.info(f"Starting alpha wallet discovery (reason: {trigger_reason})...")
            
            # Log discovery start (no Discord notification to avoid spam)
            
            # Import and run discovery
            from src.discovery.alpha_discovery_v2 import ProvenAlphaFinder
            
            # Create discovery instance
            finder = ProvenAlphaFinder(
                bitquery=self.realtime_client.bitquery_client,
                moralis=self.moralis, 
                database=self.database
            )
            
            # Run discovery
            discovery_start = time.time()
            new_wallets = await finder.discover_alpha_wallets()
            discovery_duration = time.time() - discovery_start
            
            if new_wallets:
                # Add new wallets to the tracker
                old_count = len(self.wallet_tracker.watched_wallets)
                
                # Bulk add wallets with proper activity initialization
                for wallet in new_wallets:
                    if wallet not in self.wallet_tracker.watched_wallets:
                        self.wallet_tracker.add_watched_wallet(wallet)
                
                new_count = len(self.wallet_tracker.watched_wallets)
                added_count = new_count - old_count
                
                # Verify wallets are active after adding
                active_wallets = self.wallet_tracker.get_active_wallets()
                self.logger.info(f"Post-discovery wallet status: {len(active_wallets)}/{new_count} active")
                
                self.logger.info(f"Alpha discovery complete: added {added_count} new wallets (total: {new_count})")
                
                # Log success (no Discord notification to avoid spam)
                self.logger.info(f"Alpha discovery notification disabled to reduce Discord spam")
                
                # Save to database
                await finder._save_discovered_wallets(new_wallets)
                
                # Update config file with score-aware selection
                try:
                    await self._update_config_with_new_wallets(finder, new_wallets)
                except Exception as e:
                    self.logger.warning(f"Could not update config file: {e}")
                
                # Update PumpPortal subscriptions with new wallet list
                try:
                    self.logger.info("Updating PumpPortal subscriptions after alpha discovery...")
                    current_wallets = list(self.wallet_tracker.watched_wallets)
                    await self.realtime_client.update_wallet_subscriptions(current_wallets)
                    self.logger.info("PumpPortal subscriptions updated successfully")
                except Exception as e:
                    self.logger.error(f"Failed to update PumpPortal subscriptions after discovery: {e}")
                    
            else:
                self.logger.warning("Alpha discovery found no new wallets")
                
                # Log that no new wallets were added (no Discord notification to avoid spam)
                self.logger.info(f"Alpha discovery found no wallets - Discord notification disabled")
                
        except Exception as e:
            self.logger.error(f"Error during alpha wallet discovery: {e}")
            
            # Send error notification to Discord
            if self.trading_engine.notifier:
                await self.trading_engine.notifier.send_error_notification(
                    f"Alpha wallet discovery failed: {str(e)}",
                    {"trigger_reason": trigger_reason, "module": "alpha_discovery"}
                )
    
    async def _update_config_with_new_wallets(self, finder, new_wallets: List[str]):
        """Update config file with score-aware wallet selection"""
        config_path = "config/config.yml"
        
        try:
            import os
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Get all wallet scores from the database for intelligent selection
                all_wallets = set(config.get('watched_wallets', [])) | set(new_wallets)
                
                # If we have too many wallets (>100), use score-based selection
                if len(all_wallets) > 100:
                    # Get scored wallet data from database
                    try:
                        # This is a simplified approach - in a real implementation, 
                        # you'd query the database for wallet scores
                        scored_wallets = []
                        for wallet in all_wallets:
                            stats = self.wallet_tracker.get_wallet_stats(wallet)
                            score = stats.get('avg_profit_pct', 0) * stats.get('win_rate', 0.5)
                            scored_wallets.append((wallet, score))
                        
                        # Sort by score and take top 100
                        scored_wallets.sort(key=lambda x: x[1], reverse=True)
                        top_wallets = [w[0] for w in scored_wallets[:100]]
                        
                        self.logger.info(f"Selected top 100 wallets by performance score (from {len(all_wallets)} candidates)")
                        
                    except Exception as e:
                        # Fallback to simple approach if scoring fails
                        self.logger.warning(f"Score-based selection failed ({e}), using simple approach")
                        top_wallets = list(all_wallets)[:100]
                        
                    config['watched_wallets'] = top_wallets
                else:
                    config['watched_wallets'] = list(all_wallets)
                
                # Write new config
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.logger.info(f"Updated config with {len(config['watched_wallets'])} wallets")
            
        except Exception as e:
            self.logger.warning(f"Could not update config file: {e}")

    async def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot")
        self.running = False
        
        # Stop wallet rotation
        self.wallet_rotation_manager.stop_rotation()
        
        # Close realtime client
        try:
            await self.realtime_client.close()
        except Exception as e:
            self.logger.error(f"Error closing realtime client: {e}")
        
        # Close Moralis client
        try:
            if self.moralis:
                await self.moralis.close()
        except Exception as e:
            self.logger.error(f"Error closing Moralis client: {e}")
        
        # Close database
        try:
            if self.database:
                await self.database.close()
        except Exception as e:
            self.logger.error(f"Error closing database: {e}")
        
        # Send shutdown notification to Discord
        try:
            if self.trading_engine.notifier:
                summary = self.trading_engine.pnl_store.get_summary()
                await self.trading_engine.notifier.send_text(
                    f"🛑 **Memecoin Trading Bot Stopped**\n"
                    f"Final Equity: ${summary['equity']:.2f}\n"
                    f"Total Trades: {summary['total_trades']}\n"
                    f"Win Rate: {summary['win_rate']:.1f}%"
                )
        except Exception as e:
            self.logger.error(f"Error sending shutdown notification: {e}")


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Memecoin Trading Bot")
    parser.add_argument(
        '--config', 
        default='config.yml',
        help='Configuration file path (default: config.yml)'
    )
    parser.add_argument(
        '--bot-name',
        default=None,
        help='Bot name for logging (auto-detected from config if not provided)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect bot name from config file
    bot_name = args.bot_name
    if not bot_name:
        if 'aggressive' in args.config.lower():
            bot_name = "AGGRESSIVE"
        else:
            bot_name = "CONSERVATIVE"
    
    print(f"🤖 Starting {bot_name} Bot with config: {args.config}")
    
    bot = MemecoinTradingBot(config_path=args.config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
        print(f"🛑 {bot_name} Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())