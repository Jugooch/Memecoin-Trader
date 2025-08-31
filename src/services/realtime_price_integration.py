"""
Real-time Price Integration
Wires PumpPortal price monitoring into existing trading flow to replace Moralis polling
"""

import asyncio
import logging
from typing import Dict, Optional, Callable, Set
from datetime import datetime

from .pumpportal_price_monitor import PumpPortalPriceMonitor, PriceUpdate
from ..core.realtime_position_manager import RealtimePositionManager


class RealtimePriceIntegration:
    """
    Integration layer that connects PumpPortal price monitoring 
    with existing position management and trading systems
    """
    
    def __init__(self, 
                 pumpportal_config: Dict, 
                 position_manager: RealtimePositionManager,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.position_manager = position_manager
        
        # Initialize PumpPortal price monitor
        self.price_monitor = PumpPortalPriceMonitor(pumpportal_config, self.logger)
        
        # Track active positions for price monitoring
        self.monitored_positions: Set[str] = set()
        
        # Price-based callbacks (for stop loss, take profit, etc.)
        self.price_callbacks: Dict[str, Callable] = {}
        
        # Connection state
        self.is_connected = False
        
        # Wire up the price change handler
        self.price_monitor.on_price_change = self._handle_price_change
        self.price_monitor.on_trade = self._handle_trade_event
    
    async def initialize(self) -> bool:
        """Initialize the price integration system"""
        try:
            success = await self.price_monitor.initialize()
            if success:
                self.is_connected = True
                self.logger.info("✅ Real-time price integration initialized")
            return success
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize price integration: {e}")
            return False
    
    async def start_position_monitoring(self, 
                                      mint_address: str, 
                                      symbol: str,
                                      entry_price: float,
                                      price_callback: Optional[Callable] = None) -> bool:
        """
        Start monitoring a position for real-time price updates
        Called when entering a new trade
        
        Args:
            mint_address: Token mint to monitor
            symbol: Token symbol for logging
            entry_price: Entry price for P&L calculations
            price_callback: Optional callback for price-based decisions (stop loss, etc.)
        
        Returns:
            True if monitoring started successfully
        """
        try:
            if not self.is_connected:
                self.logger.warning("Price integration not connected, attempting to initialize...")
                if not await self.initialize():
                    return False
            
            # Add token to price monitoring
            self.price_monitor.add_token(mint_address, entry_price)
            self.monitored_positions.add(mint_address)
            
            # Store callback for price-based decisions
            if price_callback:
                self.price_callbacks[mint_address] = price_callback
            
            # Start monitoring if not already started
            if not self.price_monitor.is_monitoring:
                await self.price_monitor.start_monitoring()
            
            self.logger.info(f"📊 Started real-time price monitoring: {symbol} ({mint_address[:8]}...)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring {mint_address[:8]}...: {e}")
            return False
    
    async def stop_position_monitoring(self, mint_address: str, symbol: str = None):
        """
        Stop monitoring a position (called when position is closed)
        
        Args:
            mint_address: Token mint to stop monitoring
            symbol: Token symbol for logging (optional)
        """
        try:
            # Remove from monitoring
            self.price_monitor.remove_token(mint_address)
            self.monitored_positions.discard(mint_address)
            self.price_callbacks.pop(mint_address, None)
            
            symbol_display = symbol or mint_address[:8] + "..."
            self.logger.info(f"🛑 Stopped price monitoring: {symbol_display}")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping monitoring for {mint_address[:8]}...: {e}")
    
    def get_current_price(self, mint_address: str) -> Optional[float]:
        """
        Get current cached price for a token
        Replaces Moralis get_token_price calls with instant cached data
        """
        return self.price_monitor.get_current_price(mint_address)
    
    def get_monitoring_status(self) -> Dict:
        """Get status of price monitoring"""
        base_stats = self.price_monitor.get_monitoring_stats()
        
        return {
            **base_stats,
            'monitored_positions': len(self.monitored_positions),
            'active_callbacks': len(self.price_callbacks),
            'position_mints': list(self.monitored_positions)
        }
    
    async def _handle_price_change(self, mint: str, old_price: float, new_price: float, update: PriceUpdate):
        """Handle price changes for monitored positions"""
        try:
            # Only process if this is a monitored position
            if mint not in self.monitored_positions:
                return
            
            # Calculate price change
            if old_price > 0:
                change_pct = ((new_price - old_price) / old_price) * 100
            else:
                change_pct = 0.0
            
            # Log significant price movements (>2%)
            if abs(change_pct) >= 2.0:
                self.logger.info(
                    f"📈 PRICE ALERT: {mint[:8]}... "
                    f"${old_price:.8f} → ${new_price:.8f} "
                    f"({change_pct:+.2f}%)"
                )
            
            # Call position-specific callback if set (for stop loss, take profit, etc.)
            callback = self.price_callbacks.get(mint)
            if callback:
                try:
                    await callback(mint, old_price, new_price, change_pct, update)
                except Exception as e:
                    self.logger.error(f"Error in position callback for {mint[:8]}...: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling price change for {mint[:8]}...: {e}")
    
    async def _handle_trade_event(self, update: PriceUpdate):
        """Handle trade events (optional: could be used for market analysis)"""
        try:
            # Only process trades for monitored positions
            if update.mint not in self.monitored_positions:
                return
            
            # Could add logic here for:
            # - Volume analysis
            # - Trade pattern detection
            # - Market sentiment tracking
            
            self.logger.debug(
                f"🔄 Trade event: {update.mint[:8]}... "
                f"{update.trade_type.upper()} {update.sol_amount:.3f} SOL @ ${update.price:.8f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling trade event: {e}")
    
    async def close(self):
        """Close the price integration system"""
        try:
            await self.price_monitor.close()
            self.is_connected = False
            self.monitored_positions.clear()
            self.price_callbacks.clear()
            self.logger.info("🔌 Real-time price integration closed")
        except Exception as e:
            self.logger.error(f"Error closing price integration: {e}")


# Drop-in replacement for Moralis price provider
class MoralisReplacementProvider:
    """
    Drop-in replacement for Moralis price calls using PumpPortal real-time data
    Maintains the same interface as existing Moralis calls for easy integration
    """
    
    def __init__(self, price_integration: RealtimePriceIntegration):
        self.price_integration = price_integration
        self.logger = price_integration.logger
    
    async def get_token_price(self, mint_address: str, fresh: bool = False) -> Optional[float]:
        """
        Get token price - compatible with existing Moralis calls
        
        Args:
            mint_address: Token mint address
            fresh: Ignored (PumpPortal data is always fresh)
        
        Returns:
            Current token price or None
        """
        try:
            # Get price from real-time cache
            price = self.price_integration.get_current_price(mint_address)
            
            if price is None:
                # If not monitoring this token yet, start monitoring
                # This allows seamless integration with existing code
                self.logger.info(f"🔍 Auto-starting price monitoring for {mint_address[:8]}...")
                
                success = await self.price_integration.start_position_monitoring(
                    mint_address, 
                    mint_address[:8] + "...",  # Use truncated mint as symbol
                    0.0  # No entry price for general monitoring
                )
                
                if success:
                    # Give it a moment to get first price update
                    await asyncio.sleep(1.0)
                    price = self.price_integration.get_current_price(mint_address)
            
            if price is not None:
                self.logger.debug(f"💰 Real-time price for {mint_address[:8]}...: ${price:.8f}")
            
            return price
            
        except Exception as e:
            self.logger.error(f"Error getting price for {mint_address[:8]}...: {e}")
            return None
    
    async def get_token_price_realtime(self, mint_address: str) -> Optional[float]:
        """Alias for get_token_price (both provide real-time data)"""
        return await self.get_token_price(mint_address, fresh=True)


# Integration helper function
async def create_realtime_price_system(pumpportal_config: Dict, 
                                     position_manager: RealtimePositionManager,
                                     logger: Optional[logging.Logger] = None) -> tuple:
    """
    Create and initialize the complete real-time price monitoring system
    
    Returns:
        (price_integration, moralis_replacement) tuple
    """
    # Create integration layer
    price_integration = RealtimePriceIntegration(pumpportal_config, position_manager, logger)
    
    # Initialize
    success = await price_integration.initialize()
    if not success:
        raise Exception("Failed to initialize real-time price system")
    
    # Create Moralis replacement
    moralis_replacement = MoralisReplacementProvider(price_integration)
    
    return price_integration, moralis_replacement