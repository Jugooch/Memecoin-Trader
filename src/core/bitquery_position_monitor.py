"""
Bitquery-based position monitor to replace PumpPortal price tracking
Integrates real-time WebSocket price monitoring with existing trading engine
All-in-one solution with direct WebSocket handling
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Callable, Optional
from datetime import datetime
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class PriceUpdate:
    """Real-time price update from Bitquery"""
    token_address: str
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    mean: float
    volume_usd: float

@dataclass
class Position:
    """Trading position to monitor"""
    token_address: str
    symbol: str
    entry_price: float
    quantity: float
    take_profit: float
    stop_loss: float
    side: str  # 'long' or 'short'

class BitqueryPositionMonitor:
    """
    Bitquery-based position monitor that integrates with existing bot architecture
    Replaces PumpPortal price tracking with Bitquery WebSocket streams
    All-in-one solution with direct WebSocket connection
    """
    
    def __init__(self, trading_engine, config, bitquery_client):
        """
        Initialize monitor with trading engine integration
        
        Args:
            trading_engine: The bot's trading engine
            config: Bot configuration
            bitquery_client: Existing BitqueryClient with token rotation
        """
        self.trading_engine = trading_engine
        self.config = config
        self.bitquery_client = bitquery_client
        
        if not self.bitquery_client:
            raise ValueError("BitqueryClient is required for position monitoring")
            
        # WebSocket connection will be handled by BitqueryClient with token rotation
        self.websocket = None
        self.running = False
        
        # Position tracking (replaces main.py's monitored_positions and price_cache)
        self.monitored_positions = {}  # mint_address -> position_info
        self.price_cache = {}  # mint_address -> current_price
        
        # TP/SL positions to monitor
        self.tp_sl_positions: Dict[str, Position] = {}
        
        # Track subscriptions
        self.active_subscriptions: Dict[str, str] = {}  # token_address -> subscription_id
        self.subscription_counter = 0
        
        # Performance tracking
        self.last_heartbeat_log = {}  # mint_address -> timestamp
        
        logger.info("BitqueryPositionMonitor initialized")
    
    async def start(self):
        """Start the position monitor"""
        await self._connect()
        
        # Start listening in background
        asyncio.create_task(self._listen())
        
        logger.info("🎯 Bitquery position monitor started")
    
    async def _connect(self):
        """Establish WebSocket connection using BitqueryClient's token rotation"""
        max_retries = len(self.bitquery_client.api_tokens) if hasattr(self.bitquery_client, 'api_tokens') else 5
        
        for attempt in range(max_retries):
            try:
                # Get current token from BitqueryClient with rotation
                token_index, api_token = self.bitquery_client._get_next_available_token()
                if not api_token:
                    if attempt == max_retries - 1:
                        raise Exception("No available Bitquery API tokens after trying all")
                    continue
                
                # Build WebSocket URL with current token
                ws_url = f"wss://streaming.bitquery.io/graphql?token={api_token}"
                
                logger.info(f"Connecting with token #{token_index} (attempt {attempt + 1}/{max_retries})...")
                
                self.websocket = await websockets.connect(
                    ws_url,
                    subprotocols=["graphql-transport-ws"],
                    additional_headers={"Content-Type": "application/json"}
                )
                
                # Send connection init
                await self.websocket.send(json.dumps({
                    "type": "connection_init",
                    "payload": {}
                }))
                
                # Wait for ack
                response = await self.websocket.recv()
                response_data = json.loads(response)
                
                if response_data.get("type") != "connection_ack":
                    raise Exception(f"Connection not acknowledged: {response_data}")
                
                logger.info(f"✅ Bitquery WebSocket connected with token #{token_index}")
                self.running = True
                return  # Success!
                
            except Exception as e:
                logger.error(f"Failed to connect to Bitquery (attempt {attempt + 1}): {e}")
                
                # Check if this is a token issue that should trigger rotation
                error_str = str(e).lower()
                if '402' in error_str or 'payment required' in error_str or '403' in error_str or 'forbidden' in error_str:
                    logger.warning(f"Token #{token_index} has payment/auth issues, marking and trying next...")
                    # Mark this token as having issues in the BitqueryClient
                    if hasattr(self.bitquery_client, 'token_stats') and token_index is not None:
                        if '402' in error_str or 'payment required' in error_str:
                            self.bitquery_client.token_stats[token_index]['payment_required'] = True
                        elif '403' in error_str or 'forbidden' in error_str:
                            self.bitquery_client.token_stats[token_index]['forbidden'] = True
                    
                    # Try next token immediately
                    continue
                else:
                    # Non-token error - wait a bit before retrying
                    if attempt < max_retries - 1:
                        logger.info(f"Non-token error, waiting 2s before retry...")
                        await asyncio.sleep(2)
                    continue
        
        # If we get here, all attempts failed
        raise Exception(f"Failed to connect to Bitquery after {max_retries} attempts with different tokens")
    
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
        
        logger.info(f"🎯 BITQUERY MONITORING {mint_address[:8]}... "
                   f"Entry: ${position.entry_price:.8f}, "
                   f"TP: ${position.tp_price:.8f}, "
                   f"SL: ${position.sl_price:.8f}")
        
        # Add to our tracking
        self.monitored_positions[mint_address] = {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'start_time': time.time()
        }
        
        # Initialize price cache
        self.price_cache[mint_address] = position.entry_price
        
        # Subscribe to Bitquery price updates for this token
        await self._subscribe_token(mint_address, symbol)
        
        # Add position for TP/SL tracking
        tp_sl_position = Position(
            token_address=mint_address,
            symbol=symbol,
            entry_price=position.entry_price,
            quantity=position.amount,
            take_profit=position.tp_price,
            stop_loss=position.sl_price,
            side='long'  # Assuming long positions for now
        )
        
        self.tp_sl_positions[mint_address] = tp_sl_position
        
        logger.info(f"⚡ Bitquery real-time tracking started: {symbol} ({mint_address[:8]}...)")
        
        # Start heartbeat tracking
        self.last_heartbeat_log[mint_address] = time.time()
    
    async def _subscribe_token(self, token_address: str, symbol: str = None):
        """
        Subscribe to real-time price updates for a token
        
        Args:
            token_address: Solana token mint address
            symbol: Optional token symbol for logging
        """
        if not self.websocket:
            await self._connect()
        
        # Generate subscription ID
        self.subscription_counter += 1
        sub_id = f"price_{self.subscription_counter}"
        
        # Build subscription query for this specific token
        query = f"""
        subscription {{
          Trading {{
            Pairs(
              where: {{
                Token: {{ 
                  Address: {{ is: "{token_address}" }}
                }}
                Price: {{ IsQuotedInUsd: true }}
                Market: {{
                  Network: {{ is: "Solana" }}
                }}
              }}
            ) {{
              Token {{ 
                Symbol 
                Address 
              }}
              Price {{ 
                Ohlc {{ 
                  Open 
                  High 
                  Low 
                  Close 
                }} 
                Average {{ 
                  Mean 
                }} 
              }}
              Volume {{ 
                Usd 
              }}
              Block {{ 
                Time 
              }}
            }}
          }}
        }}
        """
        
        # Send subscription
        await self.websocket.send(json.dumps({
            "id": sub_id,
            "type": "subscribe",
            "payload": {"query": query}
        }))
        
        self.active_subscriptions[token_address] = sub_id
        logger.info(f"📊 Subscribed to price updates for {symbol or token_address[:8]}...")
    
    async def _unsubscribe_token(self, token_address: str):
        """Unsubscribe from a token's price updates"""
        if token_address in self.active_subscriptions:
            sub_id = self.active_subscriptions[token_address]
            
            if self.websocket:
                await self.websocket.send(json.dumps({
                    "id": sub_id,
                    "type": "complete"
                }))
            
            del self.active_subscriptions[token_address]
            logger.info(f"Unsubscribed from {token_address[:8]}...")
    
    async def remove_position_monitoring(self, mint_address: str):
        """
        Remove position from monitoring
        Called when position is closed
        """
        if mint_address in self.monitored_positions:
            symbol = self.monitored_positions[mint_address]['symbol']
            
            # Unsubscribe from Bitquery
            await self._unsubscribe_token(mint_address)
            
            # Remove from tracking
            self.monitored_positions.pop(mint_address, None)
            self.price_cache.pop(mint_address, None)
            self.last_heartbeat_log.pop(mint_address, None)
            self.tp_sl_positions.pop(mint_address, None)
            
            logger.info(f"🔕 Bitquery tracking stopped: {symbol} ({mint_address[:8]}...)")
    
    async def _listen(self):
        """Main loop to listen for price updates"""
        logger.info("🎧 Listening for Bitquery price updates...")
        
        while self.running:
            try:
                if not self.websocket:
                    await self._connect()
                
                message = await self.websocket.recv()
                msg_data = json.loads(message)
                msg_type = msg_data.get("type")
                
                if msg_type == "next":  # Data message for graphql-transport-ws
                    await self._handle_price_message(msg_data)
                    
                elif msg_type == "error":
                    logger.error(f"Bitquery subscription error: {msg_data}")
                    
                elif msg_type == "ping":
                    # Send pong
                    await self.websocket.send(json.dumps({"type": "pong"}))
                    
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Bitquery WebSocket connection closed: {e}")
                # Check if this is due to token issues
                error_str = str(e).lower()
                if '402' in error_str or 'payment required' in error_str or '403' in error_str:
                    logger.warning("Connection closed due to token issue - will try with rotated token")
                await self._reconnect()
            except Exception as e:
                logger.error(f"Error in Bitquery listen loop: {e}")
                # Check if this is a token-related error
                error_str = str(e).lower()
                if '402' in error_str or 'payment required' in error_str or '403' in error_str or 'forbidden' in error_str:
                    logger.warning("API error indicates token issue - will reconnect with rotated token")
                    await self._reconnect()
                else:
                    await asyncio.sleep(1)
    
    async def _handle_price_message(self, msg_data):
        """Handle incoming price update message"""
        if not msg_data:
            logger.warning("Received None message data")
            return
            
        payload = msg_data.get("payload", {})
        data = payload.get("data", {})
        pairs = (data.get("Trading") or {}).get("Pairs", [])
        
        for pair in pairs:
            # Parse price update
            token = pair["Token"]
            price = pair["Price"]
            ohlc = price["Ohlc"]
            volume = pair["Volume"]
            timestamp = pair["Block"]["Time"]
            
            token_address = token["Address"]
            symbol = token["Symbol"]
            
            # Only process if we're monitoring this token
            if token_address not in self.monitored_positions:
                continue
            
            price_update = PriceUpdate(
                token_address=token_address,
                symbol=symbol,
                timestamp=datetime.fromisoformat(timestamp.replace('Z', '+00:00')),
                open=float(ohlc["Open"]),
                high=float(ohlc["High"]),
                low=float(ohlc["Low"]),
                close=float(ohlc["Close"]),
                mean=float(price["Average"]["Mean"]),
                volume_usd=float(volume["Usd"])
            )
            
            # Process price update
            await self._handle_price_update(price_update)
    
    async def _handle_price_update(self, price_update: PriceUpdate):
        """
        Handle real-time price updates from Bitquery
        Replaces the PumpPortal trade event processing logic
        """
        mint_address = price_update.token_address
        current_price = price_update.close
        symbol = price_update.symbol
        
        # Update price cache (replaces main.py's price_cache logic)
        old_price = self.price_cache.get(mint_address, 0.0)
        self.price_cache[mint_address] = current_price
        
        # Calculate change
        change_pct = ((current_price - old_price) / old_price * 100) if old_price > 0 else 0
        
        # Log significant price moves
        if abs(change_pct) >= 0.1:
            logger.debug(f"💰 {symbol}: ${old_price:.8f} → ${current_price:.8f} ({change_pct:+.2f}%)")
        
        if abs(change_pct) >= 5.0:
            logger.info(f"📈 MAJOR PRICE MOVE: {symbol} ${old_price:.8f} → ${current_price:.8f} ({change_pct:+.2f}%)")
        
        # Check TP/SL conditions
        await self._check_tp_sl(price_update)
        
        # Check exit conditions (trading engine's sophisticated logic)
        if mint_address in self.trading_engine.active_positions:
            await self._check_exit_conditions_instantly(mint_address, current_price, symbol)
        
        # Log heartbeat periodically
        await self._maybe_log_heartbeat(mint_address, symbol)
    
    async def _check_tp_sl(self, price_update: PriceUpdate):
        """Check if price update triggers TP or SL but respect buffer periods"""
        token_address = price_update.token_address
        
        if token_address not in self.tp_sl_positions:
            return
        
        position = self.tp_sl_positions[token_address]
        current_price = price_update.close
        
        # CRITICAL FIX: Check if trading engine position is in buffer period
        if token_address in self.trading_engine.active_positions:
            trading_position = self.trading_engine.active_positions[token_address]
            if trading_position.buffer_end_time and datetime.now() < trading_position.buffer_end_time:
                time_remaining = (trading_position.buffer_end_time - datetime.now()).total_seconds()
                logger.debug(f"🛡️ BITQUERY: Skipping TP/SL check for {position.symbol} - buffer period active ({time_remaining:.0f}s remaining)")
                return  # Skip TP/SL during buffer period
        
        # Check for TP/SL trigger
        triggered = False
        trigger_type = None
        
        if position.side == 'long':
            if current_price >= position.take_profit:
                triggered = True
                trigger_type = 'TAKE_PROFIT'
            elif current_price <= position.stop_loss:
                triggered = True
                trigger_type = 'STOP_LOSS'
        else:  # short
            if current_price <= position.take_profit:
                triggered = True
                trigger_type = 'TAKE_PROFIT'
            elif current_price >= position.stop_loss:
                triggered = True
                trigger_type = 'STOP_LOSS'
        
        if triggered:
            logger.info(f"🚨 BITQUERY {trigger_type} TRIGGERED for {position.symbol}!")
            logger.info(f"   Entry: ${position.entry_price:.8f}")
            logger.info(f"   Current: ${current_price:.8f}")
            logger.info(f"   Target: ${position.take_profit if trigger_type == 'TAKE_PROFIT' else position.stop_loss:.8f}")
            
            # Calculate P&L
            if position.side == 'long':
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
            
            logger.info(f"   P&L: {pnl_pct:+.2f}%")
            
            # Execute sell via trading engine
            sell_result = await self.trading_engine.sell_token(
                token_address,
                1.0,  # Sell 100% on TP/SL
                self.config.paper_mode,
                symbol=position.symbol,
                exit_reason=f"{trigger_type}_BITQUERY"
            )
            
            if sell_result.get("success"):
                logger.info(f"✅ BITQUERY {trigger_type} EXECUTED: {position.symbol}")
            else:
                logger.error(f"❌ BITQUERY {trigger_type} FAILED: {position.symbol} - {sell_result.get('error')}")
            
            # Clean up monitoring
            await self.remove_position_monitoring(token_address)
    
    async def _check_exit_conditions_instantly(self, mint_address: str, current_price: float, symbol: str):
        """
        Check exit conditions using trading engine (replaces main.py logic)
        This handles complex exit logic beyond simple TP/SL
        """
        if mint_address not in self.trading_engine.active_positions:
            return
        
        position = self.trading_engine.active_positions[mint_address]
        
        # Update peak price tracking
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.high_gain_peak = max(position.high_gain_peak, 
                                         ((current_price / position.entry_price) - 1) * 100)
        
        # Check trading engine's sophisticated exit logic
        exit_result = await self.trading_engine.check_exit_conditions(mint_address, current_price)
        
        if exit_result:
            exit_reason, sell_percentage = exit_result
            current_gain = ((current_price / position.entry_price) - 1) * 100
            
            logger.info(f"⚡ BITQUERY SMART EXIT: {symbol} {exit_reason} "
                       f"{sell_percentage*100:.0f}% @ ${current_price:.8f} "
                       f"({current_gain:+.1f}%)")
            
            # Execute sell
            sell_result = await self.trading_engine.sell_token(
                mint_address, sell_percentage, self.config.paper_mode,
                symbol=symbol, exit_reason=exit_reason
            )
            
            if sell_result.get("success"):
                logger.info(f"✅ BITQUERY SMART SELL: {symbol} executed successfully")
                
                # If full position sold, clean up monitoring
                if sell_percentage >= 1.0:
                    await self.remove_position_monitoring(mint_address)
            else:
                logger.error(f"❌ BITQUERY SMART SELL FAILED: {symbol} - {sell_result.get('error')}")
    
    async def _maybe_log_heartbeat(self, mint_address: str, symbol: str):
        """Log position heartbeat periodically"""
        current_time = time.time()
        last_heartbeat = self.last_heartbeat_log.get(mint_address, 0)
        
        # Log every 5 minutes
        if current_time - last_heartbeat >= 300:
            await self._log_position_heartbeat(mint_address, symbol)
            self.last_heartbeat_log[mint_address] = current_time
    
    async def _log_position_heartbeat(self, mint_address: str, symbol: str):
        """Log detailed position heartbeat with real-time stats"""
        try:
            if mint_address not in self.trading_engine.active_positions:
                return
            
            position = self.trading_engine.active_positions[mint_address]
            monitored_info = self.monitored_positions.get(mint_address, {})
            
            # Calculate hold time
            start_time = monitored_info.get('start_time', time.time())
            hold_time_seconds = time.time() - start_time
            hold_time_minutes = int(hold_time_seconds / 60)
            
            # Get current price and calculate P&L
            current_price = self.price_cache.get(mint_address, 0)
            
            if current_price and current_price > 0:
                current_pnl = ((current_price / position.entry_price) - 1) * 100
                high_gain_since_entry = position.high_gain_peak
                
                logger.info(f"💓 BITQUERY HEARTBEAT {symbol}: Hold={hold_time_minutes}m "
                           f"Current=${current_price:.8f} ({current_pnl:+.1f}%) "
                           f"Peak={high_gain_since_entry:+.1f}% "
                           f"TP=${position.tp_price:.8f} SL=${position.sl_price:.8f}")
            else:
                logger.warning(f"💓 BITQUERY HEARTBEAT {symbol}: Hold={hold_time_minutes}m "
                             f"(No current price data)")
                
        except Exception as e:
            logger.error(f"Error logging Bitquery heartbeat for {symbol}: {e}")
    
    async def _reconnect(self):
        """Reconnect and resubscribe to all tokens"""
        logger.info("Reconnecting to Bitquery...")
        
        # Store current subscriptions
        tokens_to_resubscribe = list(self.active_subscriptions.keys())
        self.active_subscriptions.clear()
        
        # Reconnect
        await self._connect()
        
        # Resubscribe
        for token_address in tokens_to_resubscribe:
            monitored_info = self.monitored_positions.get(token_address)
            symbol = monitored_info.get('symbol') if monitored_info else None
            await self._subscribe_token(token_address, symbol)
    
    def get_current_price(self, mint_address: str) -> Optional[float]:
        """Get current price from cache (replaces main.py's price_cache access)"""
        return self.price_cache.get(mint_address)
    
    def is_monitoring(self, mint_address: str) -> bool:
        """Check if token is being monitored"""
        return mint_address in self.monitored_positions
    
    def get_monitored_positions(self) -> Dict:
        """Get all monitored positions"""
        return self.monitored_positions.copy()
    
    async def stop(self):
        """Stop the position monitor"""
        self.running = False
        
        # Unsubscribe all
        for token_address in list(self.active_subscriptions.keys()):
            await self._unsubscribe_token(token_address)
        
        # Close connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        # Clean up
        self.monitored_positions.clear()
        self.price_cache.clear()
        self.last_heartbeat_log.clear()
        self.tp_sl_positions.clear()
        
        logger.info("Bitquery position monitor stopped")