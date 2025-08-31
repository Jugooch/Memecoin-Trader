"""
PumpPortal Real-time Price Monitor
Replaces Moralis polling with event-driven price updates
"""

import asyncio
import json
import logging
from typing import Dict, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass

from ..clients.pumpportal_client import PumpPortalClient


@dataclass
class PriceUpdate:
    mint: str
    price: float
    sol_amount: float
    token_amount: float
    trader: str
    trade_type: str  # 'buy' or 'sell'
    timestamp: datetime
    market_cap_sol: Optional[float] = None


class PumpPortalPriceMonitor:
    """
    Real-time price monitoring using PumpPortal WebSocket
    Replaces slow Moralis polling with fast event-driven updates
    """
    
    def __init__(self, pumpportal_config: Dict, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.client = PumpPortalClient(
            api_key=pumpportal_config.get('api_key'),
            ws_endpoint=pumpportal_config.get('ws_endpoint', 'wss://pumpportal.fun/api/data')
        )
        
        # Track monitored tokens
        self.monitored_tokens: Set[str] = set()
        self.current_prices: Dict[str, float] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        
        # Event handlers
        self.on_price_change: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        
        # Monitoring state
        self.is_monitoring = False
        self._monitoring_task = None
        
        # Stats
        self.total_updates = 0
        self.updates_per_token: Dict[str, int] = {}
    
    async def initialize(self) -> bool:
        """Initialize PumpPortal connection"""
        try:
            await self.client.initialize()
            self.logger.info("PumpPortal price monitor initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize PumpPortal price monitor: {e}")
            return False
    
    def add_token(self, mint_address: str, initial_price: float = 0.0):
        """Add a token to price monitoring"""
        # Check if this is a new token
        is_new = mint_address not in self.monitored_tokens
        
        self.monitored_tokens.add(mint_address)
        if initial_price > 0:
            self.current_prices[mint_address] = initial_price
        self.updates_per_token[mint_address] = 0
        
        self.logger.info(f"Added token {mint_address[:8]}... to price monitoring")
        
        # If monitoring is active and this is new, update subscription
        if is_new and self.is_monitoring:
            self._needs_subscription_update = True
    
    def remove_token(self, mint_address: str):
        """Remove a token from monitoring"""
        self.monitored_tokens.discard(mint_address)
        self.current_prices.pop(mint_address, None)
        self.last_trade_time.pop(mint_address, None)
        self.updates_per_token.pop(mint_address, None)
        
        self.logger.info(f"Removed token {mint_address[:8]}... from price monitoring")
    
    def get_current_price(self, mint_address: str) -> Optional[float]:
        """Get current cached price for a token"""
        return self.current_prices.get(mint_address)
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'is_monitoring': self.is_monitoring,
            'monitored_tokens': len(self.monitored_tokens),
            'total_updates': self.total_updates,
            'updates_per_token': dict(self.updates_per_token),
            'current_prices': dict(self.current_prices),
            'last_activity': max(self.last_trade_time.values()) if self.last_trade_time else None
        }
    
    async def start_monitoring(self):
        """Start real-time price monitoring"""
        if self.is_monitoring:
            self.logger.warning("Price monitoring already started")
            return
        
        if not self.monitored_tokens:
            self.logger.warning("No tokens to monitor - add tokens first")
            return
        
        self.is_monitoring = True
        self.logger.info(f"Starting price monitoring for {len(self.monitored_tokens)} tokens")
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop price monitoring"""
        self.is_monitoring = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Price monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            self.logger.info("Starting PumpPortal token trade monitoring...")
            
            # PRODUCTION APPROACH: Subscribe to ALL trades, filter client-side
            # This ensures continuous data flow when adding/removing tokens
            async for trade_event in self._subscribe_to_all_token_trades():
                if not self.is_monitoring:
                    break
                
                # Only process trades for tokens we're monitoring
                if trade_event.get('mint') in self.monitored_tokens:
                    # Process trade event
                    price_update = self._process_trade_event(trade_event)
                    if price_update:
                        await self._handle_price_update(price_update)
                    
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            self.is_monitoring = False
            raise
    
    async def _subscribe_to_all_token_trades(self):
        """
        PRODUCTION APPROACH: Subscribe to ALL token trades and filter client-side
        This ensures continuous data flow without reconnection when adding/removing tokens
        """
        try:
            self.logger.info("📊 Subscribing to ALL token trades (will filter client-side)")
            self.logger.info(f"   Currently monitoring {len(self.monitored_tokens)} tokens")
            
            # Subscribe to ALL token trades
            subscription_msg = {
                "method": "subscribeTokenTrade",
                "keys": []  # Empty array = ALL tokens
            }
            
            # Send subscription
            await self.client.websocket.send(json.dumps(subscription_msg))
            
            # Wait for and log the subscription response
            response = await self.client.websocket.recv()
            self.logger.info(f"📨 Subscription response: {response}")
            
            # Check if subscription was successful
            if "error" in response.lower() or "fail" in response.lower():
                self.logger.warning(f"⚠️ Subscription may have failed: {response}")
                # Fall back to specific token subscription
                raise Exception("All token subscription not supported, need to use specific tokens")
            
            # Track stats for debugging
            total_trades_received = 0
            trades_for_our_tokens = 0
            
            # Listen for ALL trade events
            async for message in self.client.websocket:
                try:
                    data = json.loads(message)
                    total_trades_received += 1
                    
                    # Parse as trade event
                    trade_event = self._parse_trade_message(data)
                    if trade_event:
                        mint = trade_event.get('mint')
                        
                        # Only yield trades for tokens we're monitoring
                        if mint in self.monitored_tokens:
                            trades_for_our_tokens += 1
                            
                            # Log stats occasionally
                            if trades_for_our_tokens % 10 == 1:
                                efficiency = (trades_for_our_tokens / total_trades_received * 100) if total_trades_received > 0 else 0
                                self.logger.debug(
                                    f"📊 Trade stats: {trades_for_our_tokens}/{total_trades_received} "
                                    f"({efficiency:.1f}% relevant)"
                                )
                            
                            yield trade_event
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing trade message: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in all token trades subscription: {e}")
            raise
    
    async def _subscribe_to_specific_tokens(self):
        """Subscribe to trades for specific monitored tokens"""
        try:
            if not self.monitored_tokens:
                self.logger.warning("No tokens to monitor")
                return
            
            # Convert set to list of token addresses
            token_list = list(self.monitored_tokens)
            self.logger.info(f"📊 Subscribing to trades for {len(token_list)} specific tokens")
            
            # Log the tokens we're monitoring for debugging
            for token in token_list[:3]:  # Show first 3
                self.logger.info(f"   Monitoring: {token[:8]}...")
            
            # Send the subscription with specific token addresses
            subscription_msg = {
                "method": "subscribeTokenTrade",
                "keys": token_list  # Array of token addresses to watch
            }
            
            # Send subscription directly
            await self.client.websocket.send(json.dumps(subscription_msg))
            self.logger.info(f"✅ Sent subscription for {len(token_list)} tokens")
            
            # Now listen for trade events on these specific tokens
            async for message in self.client.websocket:
                try:
                    data = json.loads(message)
                    
                    # Parse as trade event
                    trade_event = self._parse_trade_message(data)
                    if trade_event:
                        mint = trade_event.get('mint')
                        if mint in self.monitored_tokens:
                            self.logger.debug(f"📈 Received trade for monitored token: {mint[:8]}...")
                            yield trade_event
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing trade message: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error subscribing to specific tokens: {e}")
            raise
    
    def _parse_trade_message(self, data: Dict) -> Optional[Dict]:
        """Parse a trade message from PumpPortal"""
        try:
            # Check if this is a trade message
            if 'txType' in data and data['txType'] in ['buy', 'sell']:
                return {
                    'mint': data.get('mint'),
                    'trade_type': data.get('txType'),
                    'sol_amount': float(data.get('solAmount', 0)),
                    'token_amount': float(data.get('tokenAmount', 0)),
                    'trader': data.get('traderPublicKey'),
                    'signature': data.get('signature'),
                    'timestamp': datetime.now()
                }
            return None
        except Exception:
            return None
    
    def _process_trade_event(self, trade_event: Dict) -> Optional[PriceUpdate]:
        """Process a trade event into a price update"""
        try:
            mint = trade_event.get('mint')
            if not mint or mint not in self.monitored_tokens:
                return None
            
            # Extract trade data (already parsed in _parse_trade_message)
            sol_amount = float(trade_event.get('sol_amount', 0))
            token_amount = float(trade_event.get('token_amount', 0))
            trade_type = trade_event.get('trade_type', 'unknown')
            
            if sol_amount <= 0 or token_amount <= 0:
                return None
            
            # Calculate price
            price = sol_amount / token_amount
            
            trader = trade_event.get('buyer') or trade_event.get('seller', '')
            timestamp = datetime.now()
            
            # Parse timestamp if provided
            if 'timestamp' in trade_event:
                try:
                    timestamp_str = trade_event['timestamp']
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except Exception:
                    pass  # Use current time if parsing fails
            
            return PriceUpdate(
                mint=mint,
                price=price,
                sol_amount=sol_amount,
                token_amount=token_amount,
                trader=trader,
                trade_type=trade_type,
                timestamp=timestamp,
                market_cap_sol=trade_event.get('marketCapSol')
            )
            
        except Exception as e:
            self.logger.error(f"Error processing trade event: {e}")
            return None
    
    async def _handle_price_update(self, update: PriceUpdate):
        """Handle a price update"""
        mint = update.mint
        old_price = self.current_prices.get(mint, 0.0)
        new_price = update.price
        
        # Update cached data
        self.current_prices[mint] = new_price
        self.last_trade_time[mint] = update.timestamp
        self.total_updates += 1
        self.updates_per_token[mint] = self.updates_per_token.get(mint, 0) + 1
        
        # Calculate price change
        if old_price > 0:
            change_pct = ((new_price - old_price) / old_price) * 100
        else:
            change_pct = 0.0
        
        # Log significant price changes
        if abs(change_pct) >= 1.0 or self.updates_per_token[mint] <= 5:  # Log first 5 updates or >1% changes
            self.logger.info(
                f"💰 PRICE UPDATE: {mint[:8]}... "
                f"${old_price:.8f} → ${new_price:.8f} "
                f"({change_pct:+.2f}%) [{update.trade_type.upper()}]"
            )
        
        # Call price change handler if set
        if self.on_price_change:
            try:
                await self.on_price_change(mint, old_price, new_price, update)
            except Exception as e:
                self.logger.error(f"Error in price change handler: {e}")
        
        # Call trade handler if set
        if self.on_trade:
            try:
                await self.on_trade(update)
            except Exception as e:
                self.logger.error(f"Error in trade handler: {e}")
    
    async def close(self):
        """Close the price monitor"""
        await self.stop_monitoring()
        await self.client.close()
        self.logger.info("PumpPortal price monitor closed")


# Integration helper for main trading flow
class PumpPortalPriceProvider:
    """
    Wrapper that provides Moralis-compatible price interface using PumpPortal
    Drop-in replacement for Moralis polling in the main trading flow
    """
    
    def __init__(self, pumpportal_config: Dict, logger: Optional[logging.Logger] = None):
        self.monitor = PumpPortalPriceMonitor(pumpportal_config, logger)
        self.logger = logger or logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize the price provider"""
        return await self.monitor.initialize()
    
    async def get_token_price_realtime(self, mint_address: str) -> Optional[float]:
        """
        Get current price for a token (Moralis-compatible interface)
        This replaces the Moralis polling approach with real-time data
        """
        # If we're not monitoring this token yet, start monitoring
        if mint_address not in self.monitor.monitored_tokens:
            self.monitor.add_token(mint_address)
            
            # Start monitoring if not already started
            if not self.monitor.is_monitoring:
                await self.monitor.start_monitoring()
                
                # Give it a moment to get the first price update
                await asyncio.sleep(2.0)
        
        return self.monitor.get_current_price(mint_address)
    
    async def start_position_monitoring(self, mint_address: str, initial_price: float = 0.0):
        """Start monitoring a position (called when entering a trade)"""
        self.monitor.add_token(mint_address, initial_price)
        
        if not self.monitor.is_monitoring:
            await self.monitor.start_monitoring()
    
    async def stop_position_monitoring(self, mint_address: str):
        """Stop monitoring a position (called when exiting a trade)"""
        self.monitor.remove_token(mint_address)
    
    def set_price_change_handler(self, handler: Callable):
        """Set callback for price changes (for position management)"""
        self.monitor.on_price_change = handler
    
    async def close(self):
        """Close the price provider"""
        await self.monitor.close()