"""
Unified real-time client that can use either Bitquery or Pump Portal for WebSocket feeds
"""

import asyncio
import logging
from typing import Dict, AsyncGenerator, Optional, Union, List
from .bitquery_client import BitqueryClient
from .pumpportal_client import PumpPortalClient


class RealtimeClient:
    def __init__(self, config: Dict):
        """
        Initialize with config that contains both Bitquery and PumpPortal credentials
        
        Config format:
        {
            'bitquery_tokens': [...],
            'pumpportal': {
                'api_key': '...',
                'ws_endpoint': 'wss://pumpportal.fun/api/data'
            },
            'realtime_source': 'pumpportal'  # or 'bitquery'
        }
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Determine which client to use
        self.source = config.get('realtime_source', 'pumpportal').lower()
        
        # Initialize clients
        self.bitquery_client = None
        self.pumpportal_client = None
        
        # Always initialize Bitquery client if tokens are available (needed for alpha discovery)
        bitquery_tokens = config.get('bitquery_tokens', config.get('bitquery_token'))
        if bitquery_tokens:
            self.bitquery_client = BitqueryClient(bitquery_tokens)
        
        if self.source == 'pumpportal' or config.get('pumpportal'):
            pumpportal_config = config.get('pumpportal', {})
            api_key = pumpportal_config.get('api_key')
            ws_endpoint = pumpportal_config.get('ws_endpoint', 'wss://pumpportal.fun/api/data')
            
            if api_key:
                self.pumpportal_client = PumpPortalClient(api_key, ws_endpoint)
        
        self.logger.info(f"Realtime client initialized with source: {self.source}")
    
    async def initialize(self):
        """Initialize the active client and bitquery client for alpha discovery"""
        # Initialize the active realtime source
        if self.source == 'bitquery' and self.bitquery_client:
            await self.bitquery_client.initialize()
        elif self.source == 'pumpportal' and self.pumpportal_client:
            await self.pumpportal_client.initialize()
        else:
            raise ValueError(f"Invalid realtime source '{self.source}' or missing credentials")
        
        # Always initialize Bitquery client if available (needed for alpha discovery)
        if self.bitquery_client and self.source != 'bitquery':
            try:
                await self.bitquery_client.initialize()
                self.logger.info("Bitquery client initialized for alpha discovery")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Bitquery client for alpha discovery: {e}")
    
    async def subscribe_token_launches(self) -> AsyncGenerator[Dict, None]:
        """Subscribe to token launches using the configured source"""
        self.logger.info(f"Starting token launch subscription via {self.source}")
        
        try:
            if self.source == 'bitquery' and self.bitquery_client:
                async for token_data in self.bitquery_client.subscribe_token_launches():
                    yield token_data
                    
            elif self.source == 'pumpportal' and self.pumpportal_client:
                async for token_data in self.pumpportal_client.subscribe_token_launches():
                    yield token_data
                    
            else:
                raise ValueError(f"No client available for source: {self.source}")
                
        except Exception as e:
            self.logger.error(f"Error in {self.source} subscription: {e}")
            
            # Try fallback to other source if available
            if self.source == 'pumpportal' and self.bitquery_client:
                self.logger.info("Falling back to Bitquery...")
                try:
                    await self.bitquery_client.initialize()
                    async for token_data in self.bitquery_client.subscribe_token_launches():
                        yield token_data
                except Exception as fallback_error:
                    self.logger.error(f"Fallback to Bitquery failed: {fallback_error}")
                    raise
            
            elif self.source == 'bitquery' and self.pumpportal_client:
                self.logger.info("Falling back to Pump Portal...")
                try:
                    await self.pumpportal_client.initialize()
                    async for token_data in self.pumpportal_client.subscribe_token_launches():
                        yield token_data
                except Exception as fallback_error:
                    self.logger.error(f"Fallback to Pump Portal failed: {fallback_error}")
                    raise
            else:
                raise
    
    async def subscribe_token_trades(self, mint_address: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """Subscribe to token trades (if supported by the source)"""
        self.logger.info(f"Starting token trade subscription via {self.source}")
        
        if self.source == 'pumpportal' and self.pumpportal_client:
            async for trade_data in self.pumpportal_client.subscribe_token_trades(mint_address):
                yield trade_data
        else:
            # Bitquery doesn't have a separate trades subscription - it's included in launches
            self.logger.warning(f"Trade subscription not supported for {self.source}")
            return
    
    async def close(self):
        """Close all client connections"""
        if self.bitquery_client:
            # Bitquery client doesn't have explicit close method, but clean up if needed
            if hasattr(self.bitquery_client, 'client') and self.bitquery_client.client:
                await self.bitquery_client.client.transport.close()
        
        if self.pumpportal_client:
            await self.pumpportal_client.close()
        
        self.logger.info("Realtime client connections closed")
    
    def get_source(self) -> str:
        """Get the current active source"""
        return self.source
    
    def is_connected(self) -> bool:
        """Check if the active client is connected"""
        if self.source == 'bitquery' and self.bitquery_client:
            # Bitquery doesn't expose connection status, assume connected if initialized
            return self.bitquery_client.client is not None
        elif self.source == 'pumpportal' and self.pumpportal_client:
            return self.pumpportal_client.is_connected()
        else:
            return False
    
    # Proxy historical data methods to Bitquery (since we're keeping it for alpha discovery)
    async def get_recent_token_launches(self, limit: int = 50, start_time: str = None, end_time: str = None) -> List[Dict]:
        """Get historical token launches via Bitquery"""
        if not self.bitquery_client:
            raise ValueError("Bitquery client not available for historical data")
        
        return await self.bitquery_client.get_recent_token_launches(limit, start_time, end_time)
    
    async def get_trades_windowed_paginated(self, start_iso: str, end_iso: str, page_limit: int = 3000, max_pages: int = 20) -> List[Dict]:
        """Get paginated historical trades via Bitquery"""
        if not self.bitquery_client:
            raise ValueError("Bitquery client not available for historical data")
        
        return await self.bitquery_client.get_trades_windowed_paginated(start_iso, end_iso, page_limit, max_pages)
    
    async def get_token_transactions(self, mint_address: str, limit: int = 100) -> List[Dict]:
        """Get token transactions via Bitquery"""
        if not self.bitquery_client:
            raise ValueError("Bitquery client not available for historical data")
        
        return await self.bitquery_client.get_token_transactions(mint_address, limit)
    
    def get_token_status(self) -> Dict:
        """Get Bitquery token status"""
        if not self.bitquery_client:
            return {"error": "Bitquery client not available"}
        
        return self.bitquery_client.get_token_status()
    
    async def get_diagnostic_info(self) -> Dict:
        """Get diagnostic information for troubleshooting"""
        info = {
            'active_source': self.source,
            'bitquery_available': bool(self.bitquery_client),
            'pumpportal_available': bool(self.pumpportal_client),
        }
        
        # Get connection status
        if self.source == 'pumpportal' and self.pumpportal_client:
            info['pumpportal_connected'] = self.pumpportal_client.is_connected()
            try:
                info['pumpportal_details'] = await self.pumpportal_client.get_connection_info()
            except Exception as e:
                info['pumpportal_error'] = str(e)
        
        if self.bitquery_client:
            info['bitquery_connected'] = hasattr(self.bitquery_client, 'client') and self.bitquery_client.client is not None
            info['bitquery_token_status'] = self.bitquery_client.get_token_status()
        
        return info