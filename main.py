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
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.clients.bitquery_client import BitqueryClient
from src.clients.moralis_client import MoralisClient
from src.clients.pumpfun_client import PumpFunClient
from src.core.wallet_tracker import WalletTracker
from src.core.trading_engine import TradingEngine
from src.core.database import Database
from src.utils.logger_setup import setup_logging


@dataclass
class TradingConfig:
    rpc_endpoint: str
    bitquery_token: str
    moralis_key: str
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


class MemecoinTradingBot:
    def __init__(self, config_path: str = "config.yml"):
        self.config = self._load_config(config_path)
        self.logger = setup_logging(self.config.logging_level, self.config.logging_file)
        
        self.bitquery = BitqueryClient(self.config.bitquery_token)
        self.moralis = MoralisClient(self.config.moralis_key)
        
        # QuickNode is optional - only for automated trading
        self.pumpfun = None
        if hasattr(self.config, 'quicknode_endpoint') and self.config.quicknode_endpoint:
            self.pumpfun = PumpFunClient(self.config.quicknode_endpoint, self.config.quicknode_api_key)
        
        self.wallet_tracker = WalletTracker(self.config.watched_wallets)
        self.trading_engine = TradingEngine(self.config)
        self.database = Database(self.config.database_file)
        
        self.running = False
        self.trades_today = 0
        self.last_trade_time = None
        self.current_capital = self.config.initial_capital
        
        # Tracking counters for summary logging
        self.tokens_processed = 0
        self.alpha_checks_performed = 0
        
        # Token deduplication to avoid processing same token multiple times
        self.processed_tokens = {}  # Now stores {token: timestamp} for cleanup
        self.token_cache_duration = 1800  # Remember processed tokens for 30 minutes
        self.last_token_cleanup = time.time()

    def _load_config(self, config_path: str) -> TradingConfig:
        # Check both old and new config locations
        if not os.path.exists(config_path):
            config_path = os.path.join('config', config_path)
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Handle both single key and multiple keys format
        moralis_keys = config_data.get('moralis_keys', [])
        if not moralis_keys:
            # Fallback to single key format
            moralis_keys = [config_data['moralis_key']]
        
        return TradingConfig(
            rpc_endpoint=config_data['rpc_endpoint'],
            bitquery_token=config_data['bitquery_token'],
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
            pump_fun_program=config_data['pump_fun']['program_address'],
            pump_fun_api=config_data['pump_fun']['api_endpoint'],
            logging_level=config_data['logging']['level'],
            logging_file=config_data['logging']['file'],
            database_file=config_data['database']['file']
        )

    async def start(self):
        """Start the trading bot"""
        self.logger.info("Starting Memecoin Trading Bot")
        self.logger.info(f"Paper mode: {self.config.paper_mode}")
        self.logger.info(f"Initial capital: ${self.config.initial_capital}")
        
        self.running = True
        
        # Initialize database
        await self.database.initialize()
        
        # Start monitoring tasks
        await asyncio.gather(
            self.monitor_new_tokens(),
            self.manage_active_positions(),
            self.daily_reset_task(),
            self.periodic_summary_task(),
            self.heartbeat_task()
        )

    async def monitor_new_tokens(self):
        """Monitor for new token launches via Bitquery"""
        self.logger.info("Starting token launch monitoring")
        
        async for token_event in self.bitquery.subscribe_token_launches():
            if not self.running:
                break
                
            try:
                await self.process_new_token(token_event)
            except Exception as e:
                self.logger.error(f"Error processing token {token_event.get('mint', 'unknown')}: {e}")

    async def process_new_token(self, token_event: Dict):
        """Process a newly launched token"""
        mint_address = token_event['mint']
        deployer = token_event['deployer']
        
        # Clean up old processed tokens periodically
        self._cleanup_processed_tokens()
        
        # Skip if we've already processed this token recently
        if mint_address in self.processed_tokens:
            return
        
        # Add to processed dict with timestamp
        self.processed_tokens[mint_address] = time.time()
        
        # Only log tokens we're seriously considering (reduces noise)
        self.logger.debug(f"New token detected: {mint_address}")
        self.tokens_processed += 1
        
        # Check trading limits
        if not self._can_trade():
            self.logger.debug("Trading limits reached, skipping")
            return
        
        # Get token metadata and liquidity from Moralis
        try:
            # If Moralis is rate limited, skip this token entirely (safer approach)
            if self.moralis.rate_limited:
                self.logger.debug(f"Moralis rate limited - skipping {mint_address[:8]}... (safety first)")
                return
                
            metadata = await self.moralis.get_token_metadata(mint_address)
            liquidity = await self.moralis.get_token_liquidity(mint_address)
            
            # If we can't get proper data, skip the token (no guessing)
            if not metadata or not liquidity:
                self.logger.debug(f"Incomplete data for {mint_address[:8]}... - skipping")
                return
            
            if not self._passes_filters(metadata, liquidity, deployer):
                # Only log failures at debug level to reduce noise
                self.logger.debug(f"Token {mint_address} failed filters")
                return
            
            # If we get here, token passed basic filters - now it's worth logging
            self.logger.info(f"Token {mint_address[:8]}... passed initial filters - checking alpha activity")
                
        except Exception as e:
            # Only log 404s at debug level since they're common for new tokens
            if "404" in str(e) or "not found" in str(e).lower():
                self.logger.debug(f"Token {mint_address} metadata not available (likely too new)")
            else:
                self.logger.error(f"Failed to get token data for {mint_address}: {e}")
            return
        
        # Track alpha wallet activity
        self.alpha_checks_performed += 1
        alpha_signal = await self.wallet_tracker.check_alpha_activity(
            mint_address, 
            self.config.time_window_sec,
            self.config.threshold_alpha_buys,
            self.moralis
        )
        
        if alpha_signal:
            await self.execute_trade(mint_address, metadata, liquidity)

    def _can_trade(self) -> bool:
        """Check if we can execute a new trade"""
        if self.trades_today >= self.config.max_trades_per_day:
            return False
            
        if self.last_trade_time:
            time_since_last = time.time() - self.last_trade_time
            if time_since_last < self.config.min_time_between_trades:
                return False
                
        return True

    def _passes_filters(self, metadata: Dict, liquidity: Dict, deployer: str) -> bool:
        """Apply token filters"""
        # Check liquidity requirement (use total_liquidity_usd from new structure)
        liquidity_usd = liquidity.get('total_liquidity_usd', liquidity.get('usd_value', 0))
        if liquidity_usd < self.config.min_liquidity_usd:
            return False
            
        # Skip tokens with suspicious names (basic spam filter)
        name = metadata.get('name', '').lower()
        symbol = metadata.get('symbol', '').lower()
        spam_keywords = ['test', 'fake', 'scam', 'rugpull', 'honeypot']
        if any(keyword in name or keyword in symbol for keyword in spam_keywords):
            return False
            
        # Add deployer blacklist check here if needed
        
        return True

    def _cleanup_processed_tokens(self):
        """Remove old processed tokens to allow reprocessing and prevent memory bloat"""
        current_time = time.time()
        
        # Only cleanup every 10 minutes to avoid overhead
        if current_time - self.last_token_cleanup < 600:
            return
            
        # Remove tokens older than cache duration
        expired_tokens = [
            token for token, timestamp in self.processed_tokens.items()
            if current_time - timestamp > self.token_cache_duration
        ]
        
        for token in expired_tokens:
            del self.processed_tokens[token]
            
        if expired_tokens:
            self.logger.debug(f"Cleaned up {len(expired_tokens)} old processed tokens")
            
        self.last_token_cleanup = current_time

    async def execute_trade(self, mint_address: str, metadata: Dict, liquidity: Dict):
        """Execute a trade on the token"""
        trade_amount = self.current_capital * self.config.max_trade_pct
        
        self.logger.info(f"Executing trade on {mint_address}, amount: ${trade_amount}")
        
        try:
            # Execute buy order
            result = await self.trading_engine.buy_token(
                mint_address, 
                trade_amount,
                self.config.paper_mode
            )
            
            if result['success']:
                # Record trade
                await self.database.record_trade({
                    'mint': mint_address,
                    'action': 'BUY',
                    'amount': trade_amount,
                    'price': result['price'],
                    'sol_amount': result.get('sol_amount', 0),
                    'tokens_amount': result.get('tokens_received', 0),
                    'timestamp': datetime.now(),
                    'paper_mode': self.config.paper_mode,
                    'metadata': {
                        'symbol': metadata.get('symbol', 'UNKNOWN'),
                        'name': metadata.get('name', 'Unknown Token')
                    }
                })
                
                self.trades_today += 1
                self.last_trade_time = time.time()
                
                # Start monitoring for exit conditions
                asyncio.create_task(self.monitor_position(mint_address, result, metadata))
                
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")

    async def monitor_position(self, mint_address: str, entry_data: Dict, metadata: Dict = None):
        """Monitor position for take profit and stop loss"""
        entry_price = entry_data['price']
        tp_price = entry_price * self.config.tp_multiplier
        sl_price = entry_price * self.config.stop_loss_pct
        peak_price = entry_price
        symbol = metadata.get('symbol', 'UNKNOWN') if metadata else 'UNKNOWN'
        
        self.logger.info(f"Monitoring position {mint_address}, TP: {tp_price}, SL: {sl_price}")
        
        while True:
            try:
                current_price = await self.moralis.get_current_price(mint_address)
                
                if current_price > peak_price:
                    peak_price = current_price
                
                # Take profit condition
                if current_price >= tp_price:
                    sell_result = await self.trading_engine.sell_token(mint_address, 0.5, self.config.paper_mode)
                    if sell_result.get('success'):
                        # Record the sell trade with profit/loss
                        await self.database.record_trade({
                            'mint': mint_address,
                            'action': 'SELL',
                            'amount': sell_result.get('usd_amount', 0),
                            'price': current_price,
                            'sol_amount': sell_result.get('sol_amount', 0),
                            'tokens_amount': sell_result.get('tokens_sold', 0),
                            'timestamp': datetime.now(),
                            'paper_mode': self.config.paper_mode,
                            'profit': sell_result.get('profit', 0),
                            'profit_pct': sell_result.get('profit_pct', 0),
                            'exit_reason': 'TAKE_PROFIT',
                            'metadata': {
                                'symbol': symbol,
                                'type': 'sell'
                            }
                        })
                    self.logger.info(f"Take profit executed for {mint_address}")
                    tp_price = current_price * 1.5  # Adjust for remaining position
                
                # Stop loss condition
                # Only use trailing stop if we're in profit (peak > entry * 1.1)
                if current_price <= sl_price or (peak_price > entry_price * 1.1 and current_price <= peak_price * 0.85):
                    sell_result = await self.trading_engine.sell_token(mint_address, 1.0, self.config.paper_mode)
                    if sell_result.get('success'):
                        # Record the sell trade with profit/loss
                        await self.database.record_trade({
                            'mint': mint_address,
                            'action': 'SELL',
                            'amount': sell_result.get('usd_amount', 0),
                            'price': current_price,
                            'sol_amount': sell_result.get('sol_amount', 0),
                            'tokens_amount': sell_result.get('tokens_sold', 0),
                            'timestamp': datetime.now(),
                            'paper_mode': self.config.paper_mode,
                            'profit': sell_result.get('profit', 0),
                            'profit_pct': sell_result.get('profit_pct', 0),
                            'exit_reason': 'STOP_LOSS',
                            'metadata': {
                                'symbol': symbol,
                                'type': 'sell'
                            }
                        })
                    self.logger.info(f"Stop loss executed for {mint_address}")
                    break
                
                await asyncio.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring position {mint_address}: {e}")
                await asyncio.sleep(5)

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
    
    async def heartbeat_task(self):
        """Show periodic heartbeat status instead of constant WebSocket spam"""
        last_ws_activity = time.time()
        ws_active = True
        
        while self.running:
            await asyncio.sleep(30)  # Every 30 seconds
            
            # Check if we're still receiving data
            current_time = time.time()
            if self.tokens_processed > 0 or self.alpha_checks_performed > 0:
                last_ws_activity = current_time
                ws_active = True
            elif current_time - last_ws_activity > 60:  # No activity for 60 seconds
                ws_active = False
            
            # Only show heartbeat if no other activity
            if ws_active and self.tokens_processed == 0:
                self.logger.debug("WebSocket active - monitoring for new tokens...")
            elif not ws_active:
                self.logger.warning("No WebSocket activity detected - checking connection...")
    
    async def periodic_summary_task(self):
        """Log periodic summary to show the bot is working"""
        last_tokens_processed = 0
        last_alpha_checks = 0
        
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Calculate activity since last summary
            tokens_this_period = self.tokens_processed - last_tokens_processed
            alpha_checks_this_period = self.alpha_checks_performed - last_alpha_checks
            
            # Get recent trades from the last 5 minutes
            try:
                recent_trades = await self.database.get_trade_history(limit=50)
                
                # Filter trades from last 5 minutes
                five_min_ago = time.time() - 300
                period_trades = []
                for t in recent_trades:
                    timestamp = t.get('timestamp', 0)
                    # Convert timestamp to float if it's a string
                    if isinstance(timestamp, str):
                        try:
                            timestamp = float(timestamp)
                        except (ValueError, TypeError):
                            timestamp = 0
                    if timestamp > five_min_ago:
                        period_trades.append(t)
                
                # Prepare trade summary
                trade_summary = ""
                if period_trades:
                    total_bought = sum(t.get('sol_amount', 0) for t in period_trades if t.get('action') == 'BUY')
                    total_sold = sum(t.get('sol_amount', 0) for t in period_trades if t.get('action') == 'SELL')
                    profits = sum(t.get('profit', 0) for t in period_trades if t.get('profit') is not None)
                    
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
                            trade_details.append(f"  • Bought {token_name} for ${usd_amount:.2f}")
                        elif trade_type == 'sell':
                            usd_amount = trade.get('amount', 0)
                            profit_str = f" ({profit:+.2f}% profit)" if profit is not None else ""
                            trade_details.append(f"  • Sold {token_name} for ${usd_amount:.2f}{profit_str}")
                    
                    trade_summary = f"\nTrade Activity:\n" + "\n".join(trade_details)
                    if profits != 0:
                        trade_summary += f"\n  Period P&L: {profits:+.2f}%"
                
                # Build summary message
                if tokens_this_period > 0 or alpha_checks_this_period > 0 or period_trades:
                    summary = f"5min Summary: {tokens_this_period} tokens scanned, " \
                             f"{alpha_checks_this_period} alpha checks, " \
                             f"{len(period_trades)} trades executed, " \
                             f"${self.current_capital:.2f} capital"
                    
                    if trade_summary:
                        summary += trade_summary
                    
                    self.logger.info(summary)
                else:
                    self.logger.info("Bot running - monitoring pump.fun launches and alpha wallets...")
                    
            except Exception as e:
                self.logger.error(f"Error generating trade summary: {e}")
                # Fallback to simple summary
                self.logger.info(f"5min Summary: {tokens_this_period} tokens scanned, "
                               f"{alpha_checks_this_period} alpha checks, "
                               f"{self.trades_today} trades today, "
                               f"${self.current_capital:.2f} capital")
                
            last_tokens_processed = self.tokens_processed
            last_alpha_checks = self.alpha_checks_performed

    async def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot")
        self.running = False


async def main():
    bot = MemecoinTradingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
        print("Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())