"""
Moralis-based position monitor to replace Bitquery WebSocket price tracking
Uses polling instead of streaming to avoid expensive API credit consumption
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MoralisPositionMonitor:
    """
    Moralis-based position monitor that replaces Bitquery WebSocket streams
    Uses REST API polling for price updates instead of real-time streams
    """
    
    def __init__(self, trading_engine, config, moralis_client):
        """
        Initialize monitor with trading engine integration
        
        Args:
            trading_engine: The bot's trading engine
            config: Bot configuration
            moralis_client: Existing MoralisClient with key rotation
        """
        self.trading_engine = trading_engine
        self.config = config
        self.moralis_client = moralis_client
        
        if not self.moralis_client:
            raise ValueError("MoralisClient is required for position monitoring")
            
        # Position tracking (matches BitqueryPositionMonitor interface)
        self.monitored_positions = {}  # mint_address -> position_info
        self.price_cache = {}  # mint_address -> current_price
        
        # Polling control
        self.running = False
        self.polling_tasks = {}  # mint_address -> asyncio.Task
        self.poll_interval = getattr(config, 'price_poll_interval', 2.0)  # Default 2 seconds
        
        # Performance tracking
        self.last_heartbeat_log = {}  # mint_address -> timestamp
        self.heartbeat_interval = 60  # Log heartbeat every 60 seconds
        
        logger.info("MoralisPositionMonitor initialized")
    
    async def start(self):
        """Start the position monitor"""
        self.running = True
        logger.info("Moralis position monitor started")
    
    async def add_position_for_monitoring(self, mint_address: str, metadata: Dict = None):
        """
        Add a position for real-time price monitoring
        Replaces the logic in monitor_position() from main.py
        
        Args:
            mint_address: Token mint address
            metadata: Token metadata (symbol, etc.)
        """
        symbol = metadata.get('symbol', 'UNKNOWN') if metadata else 'UNKNOWN'
        
        # Check if position exists in trading engine
        if mint_address not in self.trading_engine.active_positions:
            logger.error(f"Position {mint_address} not found in trading_engine!")
            return
        
        position = self.trading_engine.active_positions[mint_address]
        
        logger.info(f"MORALIS MONITORING {mint_address[:8]}... "
                   f"Entry: ${position.entry_price:.8f}, "
                   f"Symbol: {symbol}")
        
        # Add to our tracking
        self.monitored_positions[mint_address] = {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'start_time': time.time()
        }
        
        # Initialize price cache with entry price
        self.price_cache[mint_address] = position.entry_price
        
        # Start polling task for this token
        await self._start_polling_token(mint_address, symbol)
    
    async def _start_polling_token(self, mint_address: str, symbol: str):
        """Start polling price updates for a specific token"""
        # Don't start if already polling this token
        if mint_address in self.polling_tasks:
            logger.debug(f"Already polling {symbol}, skipping...")
            return
        
        # Create and start polling task
        task = asyncio.create_task(
            self._poll_token_price(mint_address, symbol)
        )
        self.polling_tasks[mint_address] = task
        
        logger.info(f"Started price polling for {symbol} every {self.poll_interval}s")
    
    async def _poll_token_price(self, mint_address: str, symbol: str):
        """Continuously poll price for a specific token until position is closed"""
        consecutive_failures = 0
        max_failures = 5
        
        while self.running and mint_address in self.monitored_positions:
            try:
                # Check if position still exists
                if mint_address not in self.trading_engine.active_positions:
                    logger.info(f"Position {symbol} closed, stopping price polling")
                    break
                
                # Get current price from Moralis (fresh=True for real-time monitoring)
                current_price = await self.moralis_client.get_current_price(mint_address, fresh=True)
                
                if current_price and current_price > 0:
                    # Update price cache
                    old_price = self.price_cache.get(mint_address, 0)
                    self.price_cache[mint_address] = current_price
                    
                    # Log price updates occasionally
                    if old_price > 0:
                        price_change = ((current_price / old_price) - 1) * 100
                        if abs(price_change) > 2:  # Log significant price changes
                            logger.debug(f"Price update {symbol}: ${current_price:.8f} ({price_change:+.1f}%)")
                    
                    # Check exit conditions immediately using existing trading engine logic
                    await self._check_exit_conditions_instantly(mint_address, current_price, symbol)
                    
                    # Log heartbeat periodically
                    await self._maybe_log_heartbeat(mint_address, symbol)
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                else:
                    consecutive_failures += 1
                    logger.warning(f"Failed to get price for {symbol} (attempt {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error(f"Max price fetch failures reached for {symbol}, stopping monitoring")
                        break
                
                # Wait for next poll
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Error polling price for {symbol}: {e} (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error(f"Max errors reached for {symbol}, stopping monitoring")
                    break
                
                # Wait before retrying
                await asyncio.sleep(self.poll_interval)
        
        # Cleanup when polling ends
        await self._cleanup_token_monitoring(mint_address, symbol)
    
    async def _check_exit_conditions_instantly(self, mint_address: str, current_price: float, symbol: str):
        """Check exit conditions when price updates - uses existing trading engine logic"""
        if mint_address not in self.trading_engine.active_positions:
            return
        
        position = self.trading_engine.active_positions[mint_address]
        
        # Update peak price and gain tracking in position
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.high_gain_peak = max(position.high_gain_peak, 
                                         ((current_price / position.entry_price) - 1) * 100)
        
        # Use existing sophisticated exit logic from trading engine
        exit_result = await self.trading_engine.check_exit_conditions(mint_address, current_price)
        
        if exit_result:
            exit_reason, sell_percentage = exit_result
            current_gain = ((current_price / position.entry_price) - 1) * 100
            
            logger.info(f"MORALIS EXIT: {symbol} {exit_reason} "
                       f"{sell_percentage*100:.0f}% @ ${current_price:.8f} "
                       f"({current_gain:+.1f}%)")
            
            # Execute the exit using existing trading engine
            sell_result = await self.trading_engine.sell_token(
                mint_address, sell_percentage, self.config.paper_mode, 
                symbol=symbol, exit_reason=exit_reason
            )
            
            if sell_result.get("success"):
                logger.info(f"MORALIS SELL SUCCESS: {symbol} executed successfully")
            else:
                logger.error(f"MORALIS SELL FAILED: {symbol} - {sell_result.get('error')}")
    
    async def _maybe_log_heartbeat(self, mint_address: str, symbol: str):
        """Log position heartbeat periodically"""
        current_time = time.time()
        last_log = self.last_heartbeat_log.get(mint_address, 0)
        
        if current_time - last_log >= self.heartbeat_interval:
            await self._log_position_heartbeat(mint_address, symbol)
            self.last_heartbeat_log[mint_address] = current_time
    
    async def _log_position_heartbeat(self, mint_address: str, symbol: str):
        """Log position heartbeat with real-time stats"""
        try:
            if mint_address not in self.trading_engine.active_positions:
                return
            
            position = self.trading_engine.active_positions[mint_address]
            position_info = self.monitored_positions.get(mint_address, {})
            
            # Calculate hold time
            start_time = position_info.get('start_time', time.time())
            hold_time_seconds = time.time() - start_time
            hold_time_minutes = int(hold_time_seconds / 60)
            
            # Get current price and calculate P&L
            current_price = self.price_cache.get(mint_address)
            
            if current_price and current_price > 0:
                current_pnl = ((current_price / position.entry_price) - 1) * 100
                
                logger.info(
                    f"HEARTBEAT: {symbol} | Hold: {hold_time_minutes}m | "
                    f"P&L: {current_pnl:+.1f}% | Current: ${current_price:.8f} | "
                    f"Peak: {position.high_gain_peak:.1f}%"
                )
            else:
                logger.warning(f"HEARTBEAT: {symbol} | Hold: {hold_time_minutes}m | No price data")
                
        except Exception as e:
            logger.error(f"Error in position heartbeat: {e}")
    
    async def remove_position_monitoring(self, mint_address: str):
        """Remove a position from monitoring"""
        symbol = self.monitored_positions.get(mint_address, {}).get('symbol', 'UNKNOWN')
        logger.info(f"Removing monitoring for {symbol} ({mint_address[:8]}...)")
        
        await self._cleanup_token_monitoring(mint_address, symbol)
    
    async def _cleanup_token_monitoring(self, mint_address: str, symbol: str):
        """Clean up monitoring for a token"""
        # Cancel polling task
        if mint_address in self.polling_tasks:
            task = self.polling_tasks[mint_address]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.polling_tasks[mint_address]
        
        # Remove from tracking
        self.monitored_positions.pop(mint_address, None)
        self.price_cache.pop(mint_address, None)
        self.last_heartbeat_log.pop(mint_address, None)
        
        logger.info(f"Stopped monitoring {symbol}")
    
    def get_current_price(self, mint_address: str) -> Optional[float]:
        """Get current cached price for a token (matches BitqueryPositionMonitor interface)"""
        return self.price_cache.get(mint_address)
    
    def is_monitoring(self, mint_address: str) -> bool:
        """Check if a token is being monitored (matches BitqueryPositionMonitor interface)"""
        return mint_address in self.monitored_positions
    
    def get_monitored_positions(self) -> Dict:
        """Get all monitored positions (matches BitqueryPositionMonitor interface)"""
        return self.monitored_positions.copy()
    
    async def stop(self):
        """Stop the position monitor and clean up all polling tasks"""
        logger.info("Stopping Moralis position monitor...")
        self.running = False
        
        # Cancel all polling tasks
        for mint_address, task in list(self.polling_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear all data
        self.polling_tasks.clear()
        self.monitored_positions.clear()
        self.price_cache.clear()
        self.last_heartbeat_log.clear()
        
        logger.info("Moralis position monitor stopped")