"""
Pump Portal WebSocket client for real-time token launches and trades
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, AsyncGenerator, Optional
from datetime import datetime


class PumpPortalClient:
    def __init__(self, api_key: str, ws_endpoint: str):
        self.api_key = api_key
        self.ws_endpoint = ws_endpoint
        self.logger = logging.getLogger(__name__)
        
        # Track connection state
        self.websocket = None
        self.connected = False
        
        # Connection health tracking
        self.last_message_time = None
        self.connection_errors = 0
        self.max_connection_errors = 10
        
        # Prevent multiple concurrent recv operations
        self._recv_lock = asyncio.Lock()
        self._is_receiving = False
        
        # Store watched wallets for reconnection
        self.watched_wallets = []
        
        # Store our own trading wallet for position tracking
        self.self_wallet = None
        
    def add_self_wallet_monitoring(self, trading_wallet_address: str):
        """Add our own wallet to subscription for real-time position tracking"""
        self.self_wallet = trading_wallet_address
        self.logger.info(f"Added self-wallet monitoring: {trading_wallet_address[:8]}...")
        
    def handle_self_trade_event(self, trade_event: Dict) -> Optional[Dict]:
        """
        Parse trade events from our own wallet for immediate position updates.
        Returns formatted event for position manager.
        """
        try:
            # Determine if this is a buy or sell based on the event structure
            # The event has either 'buyer' or 'seller' field populated
            is_buy = trade_event.get('buyer') is not None
            
            return {
                'type': 'self_trade',
                'action': 'buy' if is_buy else 'sell',
                'mint': trade_event.get('mint'),
                'tokens_amount': float(trade_event.get('tokenAmount', 0)),
                'sol_amount': float(trade_event.get('solAmount', 0)),
                'price': float(trade_event.get('price', 0)),
                'tx_signature': trade_event.get('signature'),
                'timestamp': datetime.now(),
                'trader': trade_event.get('buyer') or trade_event.get('seller')
            }
        except Exception as e:
            self.logger.error(f"Error parsing self-trade event: {e}")
            return None
        
    async def initialize(self):
        """Initialize the WebSocket connection with proper cleanup"""
        try:
            # Close existing connection if any
            if self.websocket and not self.websocket.closed:
                try:
                    await self.websocket.close()
                    await asyncio.sleep(0.1)  # Brief wait for cleanup
                except:
                    pass  # Ignore cleanup errors
            
            # Reset receiving state
            self._is_receiving = False
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Memecoin-Trader/1.0"
            }
            
            self.logger.info(f"Connecting to Pump Portal: {self.ws_endpoint}")
            self.logger.info(f"Using API key: {self.api_key[:10]}...")
            
            # Use more robust connection parameters
            self.websocket = await websockets.connect(
                self.ws_endpoint,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                max_size=2**20,  # 1MB max message size
                max_queue=32     # Limit queued messages
            )
            
            self.connected = True
            self.logger.info("Successfully connected to Pump Portal WebSocket")
            
            # Test the connection by sending a ping
            try:
                await self.websocket.ping()
                self.logger.info("WebSocket ping successful")
            except Exception as ping_error:
                self.logger.warning(f"WebSocket ping failed: {ping_error}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Pump Portal: {e}")
            self.connected = False
            raise
    
    async def subscribe_all_events(self, watched_wallets: list = None, max_retries: int = 5) -> AsyncGenerator[Dict, None]:
        """Subscribe to both token launches and trades in a single stream with reconnection"""
        self.logger.info("subscribe_all_events called")
        
        # Store watched wallets for reconnection (use stored list if not provided)
        if watched_wallets is not None:
            self.watched_wallets = watched_wallets[:100]  # Limit to 100 wallets
        elif not self.watched_wallets:
            self.logger.warning("No watched wallets provided and none stored")
        
        # Prevent multiple instances from reading the same WebSocket
        async with self._recv_lock:
            if self._is_receiving:
                self.logger.warning("Another coroutine is already receiving messages, aborting")
                return
            
            self._is_receiving = True
            
            try:
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        if not self.connected or not self.websocket:
                            self.logger.info(f"WebSocket not connected (attempt {retry_count + 1}/{max_retries}), initializing...")
                            await self.initialize()
                        
                        # Subscribe to token launches
                        subscriptions = [
                            {"method": "subscribeNewToken"},
                        ]
                        
                        # Try different approaches for trade subscription
                        all_wallets = []
                        
                        # Add alpha wallets
                        if self.watched_wallets:
                            all_wallets.extend(self.watched_wallets)
                            
                        # Add our own wallet for position tracking
                        if self.self_wallet and self.self_wallet not in all_wallets:
                            all_wallets.append(self.self_wallet)
                        
                        if all_wallets:
                            # Subscribe to trades from alpha wallets + our wallet
                            subscriptions.append({"method": "subscribeAccountTrade", "keys": all_wallets[:100]})  # Limit to 100
                            alpha_count = len(self.watched_wallets) if self.watched_wallets else 0
                            self_count = 1 if self.self_wallet else 0
                            self.logger.info(f"Subscribing to trades from {alpha_count} alpha wallets + {self_count} self wallet")
                        else:
                            # Try to subscribe to all token trades
                            subscriptions.append({"method": "subscribeTokenTrade", "keys": []})
                            self.logger.info("Subscribing to all token trades")
                        
                        self.logger.info(f"Sending {len(subscriptions)} subscription messages...")
                        
                        for sub_msg in subscriptions:
                            try:
                                await self.websocket.send(json.dumps(sub_msg))
                                self.logger.info(f"Sent subscription: {sub_msg}")
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                self.logger.warning(f"Failed to send subscription: {e}")
                        
                        self.logger.info("Subscribed to both token launches and trades, waiting for messages...")
                        
                        message_count = 0
                        # Process all messages from the single WebSocket
                        async for message in self.websocket:
                            message_count += 1
                            self.last_message_time = asyncio.get_event_loop().time()
                            
                            if message_count <= 5:  # Log first 5 messages for debugging
                                self.logger.info(f"PumpPortal message #{message_count}: {message[:200]}...")
                            
                            # Reset error count on successful message
                            self.connection_errors = 0
                            
                            try:
                                data = json.loads(message)
                                
                                # Determine message type and parse accordingly
                                if 'txType' in data:
                                    tx_type = data.get('txType', '').lower()
                                    
                                    if tx_type == 'create':
                                        # Token creation event
                                        token_data = self._parse_pump_portal_message(data)
                                        if token_data:
                                            token_data['event_type'] = 'token_launch'
                                            self.logger.debug(f"Token launch: {token_data.get('symbol')} ({token_data.get('mint', '')[:8]}...)")
                                            yield token_data
                                            
                                    elif tx_type in ['buy', 'sell']:
                                        # Trade event
                                        trade_data = self._parse_pump_portal_trade(data)
                                        if trade_data:
                                            trade_data['event_type'] = 'trade'
                                            yield trade_data
                                else:
                                    # Try to parse as either type
                                    token_data = self._parse_pump_portal_message(data)
                                    if token_data:
                                        token_data['event_type'] = 'token_launch'
                                        yield token_data
                                    else:
                                        trade_data = self._parse_pump_portal_trade(data)
                                        if trade_data:
                                            trade_data['event_type'] = 'trade'
                                            yield trade_data
                                            
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Failed to parse message: {e}")
                            except Exception as e:
                                self.logger.error(f"Error processing message: {e}")
                                
                    except (websockets.exceptions.ConnectionClosed, 
                            websockets.exceptions.ConnectionClosedError,
                            ConnectionResetError, 
                            OSError) as e:
                        self.logger.warning(f"WebSocket connection lost: {e}")
                        self.connected = False
                        retry_count += 1
                        
                        if retry_count < max_retries:
                            wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30s
                            self.logger.info(f"Attempting reconnect in {wait_time}s (attempt {retry_count + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            self.logger.error(f"Max reconnection attempts ({max_retries}) reached")
                            raise
                            
                    except Exception as e:
                        # For the specific 'NoneType' resume_reading error, treat as connection issue
                        if "resume_reading" in str(e) or "NoneType" in str(e):
                            self.logger.warning(f"WebSocket transport error: {e}")
                            self.connected = False
                            retry_count += 1
                            
                            if retry_count < max_retries:
                                wait_time = min(2 ** retry_count, 30)
                                self.logger.info(f"Attempting reconnect after transport error in {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                self.logger.error(f"Max reconnection attempts ({max_retries}) reached after transport errors")
                                raise
                        else:
                            self.logger.error(f"Pump Portal WebSocket error: {e}")
                            self.connected = False
                            raise
                            
                # If we get here, all retries are exhausted
                self.logger.error("WebSocket connection could not be established after retries")
                raise ConnectionError("Failed to maintain WebSocket connection to PumpPortal")
                
            finally:
                self._is_receiving = False
    
    async def subscribe_token_launches(self) -> AsyncGenerator[Dict, None]:
        """Subscribe to new Pump.fun token creation events - matches Bitquery interface"""
        if not self.connected or not self.websocket:
            await self.initialize()
        
        try:
            # Try different subscription methods that Pump Portal might use
            subscription_messages = [
                {"method": "subscribeNewToken"},
                {"action": "subscribe", "type": "new_tokens"},
                {"type": "subscribe", "channel": "new_tokens"},
                {"subscribe": "new_tokens"},
                {"method": "subscribe", "params": {"channels": ["new_tokens"]}},
            ]
            
            # Try each subscription format
            for i, sub_msg in enumerate(subscription_messages):
                try:
                    await self.websocket.send(json.dumps(sub_msg))
                    self.logger.info(f"Sent subscription message #{i+1}: {sub_msg}")
                    await asyncio.sleep(0.1)  # Small delay between attempts
                except Exception as e:
                    self.logger.warning(f"Failed to send subscription #{i+1}: {e}")
            
            self.logger.info("All subscription attempts sent, waiting for messages...")
            
            # Add a timeout to detect if no messages are received
            message_count = 0
            last_message_time = asyncio.get_event_loop().time()
            
            # Create a timeout task to warn if no messages received
            async def timeout_warning():
                await asyncio.sleep(30)  # Wait 30 seconds
                if message_count == 0:
                    self.logger.warning("No messages received from Pump Portal WebSocket after 30 seconds")
                    self.logger.warning("This might indicate subscription format issues or API changes")
            
            # Start the timeout warning task
            timeout_task = asyncio.create_task(timeout_warning())
            
            try:
                async for message in self.websocket:
                    message_count += 1
                    current_time = asyncio.get_event_loop().time()
                    
                    # Cancel timeout warning after first message
                    if message_count == 1:
                        timeout_task.cancel()
                        self.logger.info("First message received, WebSocket appears to be working")
                    
                    # Log every message for debugging (at least initially)
                    if message_count <= 10:  # Log first 10 messages in detail
                        self.logger.info(f"Received message #{message_count}: {message}")
                    else:
                        self.logger.debug(f"Received message #{message_count}: {message[:200]}...")
                    
                    # Log time gaps between messages
                    if message_count > 1:
                        time_gap = current_time - last_message_time
                        if time_gap > 60:  # More than 1 minute gap
                            self.logger.warning(f"Long gap between messages: {time_gap:.1f} seconds")
                    last_message_time = current_time
                    
                    try:
                        data = json.loads(message)
                        self.logger.debug(f"Parsed JSON data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                        
                        # Parse Pump Portal message and convert to Bitquery-compatible format
                        token_data = self._parse_pump_portal_message(data)
                        if token_data:
                            self.logger.info(f"Successfully parsed token: {token_data.get('symbol', 'Unknown')} ({token_data.get('mint', '')[:8]}...)")
                            yield token_data
                        else:
                            self.logger.debug("Message did not contain valid token data")
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse message: {e}, Raw message: {message[:500]}")
                    except Exception as e:
                        self.logger.error(f"Error processing message: {e}, Data: {message[:200]}")
            
            finally:
                # Clean up timeout task
                if not timeout_task.done():
                    timeout_task.cancel()
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Pump Portal WebSocket connection closed")
            self.connected = False
            raise
        except Exception as e:
            self.logger.error(f"Pump Portal WebSocket error: {e}")
            self.connected = False
            raise
    
    async def subscribe_token_trades(self, mint_address: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """Subscribe to token trades (all tokens or specific mint)"""
        if not self.connected or not self.websocket:
            await self.initialize()
        
        try:
            # Try multiple subscription formats for trades
            if mint_address:
                subscription_messages = [
                    {"method": "subscribeTokenTrade", "keys": [mint_address]},
                    {"method": "subscribeAccountTrade"},  # Maybe this gets all trades?
                ]
            else:
                # Subscribe to all token trades - try different formats
                subscription_messages = [
                    {"method": "subscribeTokenTrade"},
                    {"method": "subscribeAccountTrade"},
                    {"method": "subscribeAllTrades"},  # Guessing this might work
                ]
            
            # Send all subscription attempts
            for msg in subscription_messages:
                try:
                    await self.websocket.send(json.dumps(msg))
                    self.logger.info(f"Sent trade subscription: {msg}")
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self.logger.warning(f"Failed to send trade subscription: {e}")
            
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Parse trade data
                    trade_data = self._parse_pump_portal_trade(data)
                    if trade_data:
                        yield trade_data
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse trade message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing trade message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Pump Portal trades WebSocket connection closed")
            self.connected = False
            raise
        except Exception as e:
            self.logger.error(f"Pump Portal trades WebSocket error: {e}")
            self.connected = False
            raise
    
    def _parse_pump_portal_message(self, data: Dict) -> Optional[Dict]:
        """Parse Pump Portal new token message into Bitquery-compatible format"""
        try:
            # Handle different message types from Pump Portal
            if not isinstance(data, dict):
                self.logger.debug(f"Message is not a dict: {type(data)}")
                return None
            
            # Log all available fields for debugging
            self.logger.debug(f"Available fields in message: {list(data.keys())}")
            self.logger.debug(f"Full message data: {data}")
                
            # Look for token creation data - try multiple possible field names
            mint_fields = ['mint', 'tokenAddress', 'token', 'mintAddress', 'address']
            mint = None
            for field in mint_fields:
                if field in data and data[field]:
                    mint = data[field]
                    self.logger.debug(f"Found mint address in field '{field}': {mint}")
                    break
            
            if not mint:
                self.logger.debug(f"No mint address found in fields: {mint_fields}")
                return None
            
            # Get token metadata with various possible field names
            name = data.get('name') or data.get('tokenName') or 'Unknown'
            symbol = data.get('symbol') or data.get('ticker') or ''
            creator = data.get('creator') or data.get('deployer') or data.get('user')
            
            # Get timestamp - use current time if not provided
            timestamp = data.get('timestamp') or data.get('blockTime')
            if timestamp:
                # Convert to ISO format if needed
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp).isoformat() + "Z"
            else:
                timestamp = datetime.utcnow().isoformat() + "Z"
            
            result = {
                'mint': mint,
                'name': name,
                'symbol': symbol,
                'deployer': creator,
                'timestamp': timestamp,
                'signature': data.get('signature') or data.get('txid') or '',
                'buyer': None,  # Not available in token creation event
                'buy_amount': 0  # Not applicable for token creation
            }
            
            self.logger.debug(f"Successfully created token data: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing Pump Portal message: {e}")
            self.logger.error(f"Message data was: {data}")
            return None
    
    def _parse_pump_portal_trade(self, data: Dict) -> Optional[Dict]:
        """Parse Pump Portal trade message into Bitquery-compatible format"""
        try:
            if not isinstance(data, dict):
                return None
            
            # Extract trade information
            mint = data.get('mint') or data.get('tokenAddress')
            if not mint:
                return None
            
            # Get trade details
            trader = data.get('traderPublicKey') or data.get('trader')
            signature = data.get('signature', '')
            timestamp = data.get('timestamp')
            
            # Convert timestamp
            if timestamp:
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp).isoformat() + "Z"
            else:
                timestamp = datetime.utcnow().isoformat() + "Z"
            
            # Determine if this is a buy or sell based on PumpPortal format
            # PumpPortal uses 'txType' field with values like 'buy', 'sell'
            tx_type = data.get('txType', '').lower()
            trade_type = data.get('tradeType', tx_type).lower()
            
            # Check multiple fields to determine if it's a buy
            is_buy = (trade_type == 'buy' or tx_type == 'buy' or 
                     'buy' in trade_type or 'buy' in tx_type)
            
            # Get amounts - PumpPortal may use different field names
            amount = data.get('tokenAmount') or data.get('vTokensInBondingCurve', 0)
            sol_amount = data.get('solAmount') or data.get('vSolInBondingCurve', 0)
            
            return {
                'mint': mint,
                'name': data.get('name', ''),
                'symbol': data.get('symbol', ''),
                'deployer': None,  # Not available in trade data
                'timestamp': timestamp,
                'signature': signature,
                'buyer': trader if is_buy else None,
                'seller': trader if not is_buy else None,
                'buy_amount': amount if is_buy else 0,
                'sell_amount': amount if not is_buy else 0,
                'sol_amount': sol_amount,
                'trade_type': 'buy' if is_buy else 'sell'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Pump Portal trade: {e}")
            return None
    
    async def update_wallet_subscriptions(self, new_wallets: list):
        """Update the wallet subscription list by forcing reconnection"""
        self.logger.info(f"Updating PumpPortal subscriptions with {len(new_wallets)} wallets")
        
        # Store the new wallet list
        self.watched_wallets = new_wallets[:100]  # Limit to 100 wallets
        
        # Force reconnection by setting connected = False
        # This will cause the monitoring loop to reconnect with new wallet list
        old_connected = self.connected
        self.connected = False
        
        # Try to close the websocket gracefully if it exists
        if self.websocket:
            try:
                self.logger.info("Forcing disconnection to update subscriptions...")
                await self.websocket.close()
            except Exception as e:
                self.logger.debug(f"WebSocket close error (expected): {e}")
        
        # Clear websocket reference
        self.websocket = None
        
        self.logger.info(f"PumpPortal subscription update triggered - will reconnect with {len(self.watched_wallets)} wallets")
        return True
    
    async def close(self):
        """Close the WebSocket connection"""
        self.connected = False
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")