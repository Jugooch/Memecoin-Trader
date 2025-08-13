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
        
    async def initialize(self):
        """Initialize the WebSocket connection"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Memecoin-Trader/1.0"
            }
            
            self.logger.info(f"Connecting to Pump Portal: {self.ws_endpoint}")
            self.logger.info(f"Using API key: {self.api_key[:10]}...")
            
            # Add more verbose connection logging
            # Use basic connection parameters for compatibility
            self.websocket = await websockets.connect(
                self.ws_endpoint,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10
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
            self.logger.error(f"Connection details: endpoint={self.ws_endpoint}, headers={headers}")
            raise
    
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
            if mint_address:
                # Subscribe to specific token trades
                subscription_message = {
                    "method": "subscribeTokenTrade",
                    "keys": [mint_address]
                }
            else:
                # Subscribe to all token trades
                subscription_message = {
                    "method": "subscribeTokenTrade"
                }
            
            await self.websocket.send(json.dumps(subscription_message))
            self.logger.info(f"Subscribed to token trades{f' for {mint_address[:8]}...' if mint_address else ''}")
            
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
            
            # Determine if this is a buy or sell
            # Adjust these fields based on actual Pump Portal format
            trade_type = data.get('tradeType', 'buy').lower()
            is_buy = trade_type == 'buy'
            
            amount = data.get('tokenAmount', 0)
            sol_amount = data.get('solAmount', 0)
            
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
                'trade_type': trade_type
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Pump Portal trade: {e}")
            return None
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.logger.info("Pump Portal WebSocket connection closed")
        self.connected = False
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        is_conn = self.connected and self.websocket and not self.websocket.closed
        if not is_conn and self.websocket:
            self.logger.debug(f"Connection check: connected={self.connected}, websocket_exists={bool(self.websocket)}, websocket_closed={self.websocket.closed if self.websocket else 'N/A'}")
        return is_conn
    
    async def get_connection_info(self) -> Dict:
        """Get detailed connection information for debugging"""
        info = {
            'connected': self.connected,
            'websocket_exists': bool(self.websocket),
            'websocket_closed': self.websocket.closed if self.websocket else None,
            'websocket_state': str(self.websocket.state) if self.websocket else None,
        }
        
        if self.websocket:
            try:
                # Try to get more connection details
                info['remote_address'] = str(self.websocket.remote_address) if hasattr(self.websocket, 'remote_address') else 'unknown'
                info['local_address'] = str(self.websocket.local_address) if hasattr(self.websocket, 'local_address') else 'unknown'
            except Exception as e:
                info['connection_info_error'] = str(e)
        
        return info