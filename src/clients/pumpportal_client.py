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
            self.websocket = await websockets.connect(
                self.ws_endpoint,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.connected = True
            self.logger.info("Successfully connected to Pump Portal WebSocket")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Pump Portal: {e}")
            raise
    
    async def subscribe_token_launches(self) -> AsyncGenerator[Dict, None]:
        """Subscribe to new Pump.fun token creation events - matches Bitquery interface"""
        if not self.connected or not self.websocket:
            await self.initialize()
        
        try:
            # Subscribe to new token events
            subscription_message = {
                "method": "subscribeNewToken"
            }
            await self.websocket.send(json.dumps(subscription_message))
            self.logger.info("Subscribed to new token events")
            
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Parse Pump Portal message and convert to Bitquery-compatible format
                    token_data = self._parse_pump_portal_message(data)
                    if token_data:
                        yield token_data
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
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
                return None
                
            # Look for token creation data
            # Adjust these fields based on actual Pump Portal message format
            mint = data.get('mint') or data.get('tokenAddress') or data.get('token')
            if not mint:
                return None
            
            # Get token metadata
            name = data.get('name', 'Unknown')
            symbol = data.get('symbol', '')
            creator = data.get('creator') or data.get('deployer')
            
            # Get timestamp - use current time if not provided
            timestamp = data.get('timestamp')
            if timestamp:
                # Convert to ISO format if needed
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp).isoformat() + "Z"
            else:
                timestamp = datetime.utcnow().isoformat() + "Z"
            
            # Return in same format as Bitquery for compatibility
            return {
                'mint': mint,
                'name': name,
                'symbol': symbol,
                'deployer': creator,
                'timestamp': timestamp,
                'signature': data.get('signature', ''),
                'buyer': None,  # Not available in token creation event
                'buy_amount': 0  # Not applicable for token creation
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Pump Portal message: {e}")
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
        return self.connected and self.websocket and not self.websocket.closed